"""Frozen LANG v0 train-only quality audit. Never read sealed test or modify old Runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aster.corpus.pipeline import digest, json_bytes
from aster.training.dataset import load_training_tokenizer
from aster.training.lang_gate3 import _in_dir, _view_samples_no_test_read

VIEW_ID = "6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d"
TOKENIZER_ID = "37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513"
LEAN_GROUPS = frozenset({"article-lean-trust", "article-tex-lean"})
OUTPUT_DIR = "lang-v02-corpus-audit"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_checkout(root: Path, expected_commit: str) -> str:
    """Require the reviewed, clean checkout and pin the auditor source bytes."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("An exact 40-hex commit SHA is required")
    if root != ROOT.resolve():
        raise ValueError("--root must equal the executing runner checkout")
    def git(*args: str) -> bytes:
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except subprocess.CalledProcessError as exc:
            raise ValueError("Cannot attest Git checkout") from exc
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("Wrong Git checkout root")
    if git("rev-parse", "HEAD").decode().strip() != expected_commit:
        raise ValueError("Git HEAD differs from reviewed commit")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked checkout/index not clean")
    source_name = "scripts/audit_lang_v02_corpus.py"
    snapshot = (root / source_name).read_bytes()
    if snapshot != git("show", f"{expected_commit}:{source_name}"):
        raise ValueError("Auditor differs from reviewed source")
    return sha256(snapshot)


def marker_counts(value: str) -> dict[str, int]:
    return {
        "lean_case_insensitive_substring": len(re.findall("lean", value, re.IGNORECASE)),
        "bracket_3": value.count("[3]"),
        "numbered_brackets": len(re.findall(r"\[[0-9]{1,3}\]", value)),
        "code_fences": value.count(chr(96) * 3),
        "url_prefixes": len(re.findall(r"https?://|www\.", value, re.IGNORECASE)),
        "html_like_tags": len(re.findall(r"</?[A-Za-z][^<>]{0,128}>", value)),
        "control_characters": sum((ord(ch) < 32 and ch not in "\r\n\t") or ord(ch) == 127
                                  for ch in value),
    }


def shingles(value: str) -> set[str]:
    k = 64
    return {value[i:i+k] for i in range(len(value)-k+1)}


def near_duplicate_candidates(texts: list[tuple[str, str]]) -> list[dict]:
    """Exact 64-character shingle Jaccard: candidates, not semantic dedup."""
    indexed = [(record, shingles(value)) for record, value in texts]
    flagged = []
    for i, (a, left) in enumerate(indexed):
        for b, right in indexed[i+1:]:
            overlap = len(left & right)
            denom = len(left) + len(right) - overlap
            score = overlap / denom if denom else 0
            if score >= 0.20:
                flagged.append({"record_ids": [a, b], "jaccard": round(score, 6)})
    return flagged


def scan_train_view(view: Path, samples: list[dict], manifest: dict,
                    tokenizer, context: int, raw_sizes: dict[str, int]) -> dict:
    """Analyze verified one-time train snapshots. No dev/test file opens."""
    if type(context) is not int or context < 1:
        raise ValueError("Invalid frozen context length")
    files = manifest["files"]
    rows, texts = [], []
    for item in sorted(samples, key=lambda row: row["record_id"]):
        if item["split"] != "train":
            continue
        name = item["text_path"]
        if name != f"train/{item['record_id']}.txt":
            raise ValueError("Unexpected train text path")
        source = item["raw_path"].removeprefix("data/raw/pool/")
        if source not in raw_sizes:
            raise ValueError("Train source absent from frozen raw metadata")
        snapshot = _in_dir(view, name).read_bytes()
        if sha256(snapshot) != files.get(name) or sha256(snapshot) != item["text_sha256"]:
            raise ValueError("Train snapshot hash mismatch")
        value = snapshot.decode("utf-8", errors="strict")
        if not value.strip():
            raise ValueError("Empty selected train text")
        ids = tokenizer.encode(value, add_bos=True, add_eos=True)
        if len(ids) < 3 or ids[0] != tokenizer.bos_id or ids[-1] != tokenizer.eos_id:
            raise ValueError("Tokenizer BOS/EOS convention mismatch")
        if tokenizer.decode(ids) != value:
            raise ValueError("Tokenizer roundtrip mismatch")
        predicted = len(ids) - 1
        row = {
            "record_id": item["record_id"], "source": source,
            "group_id": item["group_id"], "leakage_group": item["leakage_group"],
            "domain": item["domain"],
            "lean_themed_source": item["group_id"] in LEAN_GROUPS,
            "raw_original_bytes": raw_sizes[source],
            "extracted_train_bytes": len(snapshot),
            "bpe_text_tokens": len(ids) - 2,
            "bpe_targets_including_eos": predicted,
            "training_windows": (predicted + context - 1) // context,
            "markers": marker_counts(value),
        }
        rows.append(row)
        texts.append((item["record_id"], value))
    if not rows:
        raise ValueError("No selected train documents")
    totals = Counter()
    domains = defaultdict(Counter)
    groups = defaultdict(Counter)
    for row in rows:
        for key in ("raw_original_bytes", "extracted_train_bytes", "bpe_text_tokens",
                    "bpe_targets_including_eos", "training_windows"):
            totals[key] += row[key]
            domains[row["domain"]][key] += row[key]
            groups[row["group_id"]][key] += row[key]
        for key, count in row["markers"].items():
            totals[key] += count
            domains[row["domain"]][key] += count
            groups[row["group_id"]][key] += count
    text_tokens = totals["bpe_text_tokens"]
    seen = Counter(item["text_sha256"] for item in samples if item["split"] == "train")
    return {
        "schema_version": "aster-lang-v02-corpus-private-0",
        "train_documents": len(rows), "totals": dict(totals),
        "by_domain": {k: dict(v) for k, v in sorted(domains.items())},
        "by_group": {k: dict(v) for k, v in sorted(groups.items())},
        "per_document_private": rows,
        "lean_source_text_token_share": sum(
            r["bpe_text_tokens"] for r in rows if r["lean_themed_source"]) / text_tokens,
        "largest_document_text_token_share": max(
            r["bpe_text_tokens"] for r in rows) / text_tokens,
        "exact_duplicate_pairs": sum(n*(n-1)//2 for n in seen.values()),
        "near_duplicate_candidates_private": near_duplicate_candidates(texts),
        "near_duplicate_limit": "64-character shingle Jaccard >= 0.20; manual review required",
    }


def audit(root: Path, expected_commit: str) -> dict:
    root = root.resolve()
    source_sha = verify_checkout(root, expected_commit)
    spec = json.loads((root / "configs/lang-v0-gate3.json").read_bytes())
    if (spec.get("training_view_id") != VIEW_ID
            or spec.get("tokenizer_artifact_id") != TOKENIZER_ID
            or spec.get("pilot_config", {}).get("context_length") != 128):
        raise ValueError("Frozen LANG view/tokenizer/context mismatch")
    view = root / "data/training" / VIEW_ID
    if view.is_symlink() or not view.is_dir():
        raise ValueError("Frozen training view missing or symlinked")
    # The existing verifier hashes train/dev and checks test metadata only.
    # No sealed test text bytes are ever opened.
    samples = _view_samples_no_test_read(view)
    inventory = json.loads((root / "configs/lang-v0-corpus.json").read_bytes())
    if inventory.get("expected_file_count") != 22 or inventory.get("expected_total_bytes") != 301231:
        raise ValueError("Unexpected frozen raw inventory")
    raw_sizes = {r["path"]: r["bytes"] for r in inventory["expected_files"]}
    if len(raw_sizes) != 22 or sum(raw_sizes.values()) != 301231:
        raise ValueError("Invalid frozen raw inventory")
    token_dir = root / "artifacts/tokenizers/lang-v0" / TOKENIZER_ID
    tokenizer, token_payload = load_training_tokenizer(
        token_dir, VIEW_ID, expected_artifact_id=TOKENIZER_ID)
    if tokenizer.vocab_size != spec["vocab_size"]:
        raise ValueError("Frozen tokenizer vocab mismatch")
    manifest_snapshot = (view / "manifest.json").read_bytes()
    if sha256(manifest_snapshot) != VIEW_ID:
        raise ValueError("Frozen manifest identity mismatch")
    report = scan_train_view(view, samples, json.loads(manifest_snapshot), tokenizer,
                             spec["pilot_config"]["context_length"], raw_sizes)
    report.update({
        "frozen_view_id": VIEW_ID, "frozen_tokenizer_id": TOKENIZER_ID,
        "tokenizer_payload_sha256": digest(json_bytes(token_payload)),
        "audited_commit": expected_commit, "auditor_sha256": source_sha,
        "sealed_test_text_read": False, "training_updates": 0,
        "limits": "Source topic tags are provenance, not semantic labels; raw inventory "
                  "counts differ from extracted train bytes. Marker counts are not proof of "
                  "noise or cause. One prose-only held-out dev group; no dialogue generalization.",
    })
    runs = root / "runs"
    if runs.is_symlink() or not runs.is_dir():
        raise ValueError("Run directory missing or symlinked")
    destination = runs / OUTPUT_DIR
    destination.mkdir(mode=0o700, exist_ok=False)
    output = destination / "private-report.json"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    return {
        "status": "train_only_corpus_audit_completed",
        "train_documents": report["train_documents"],
        "raw_original_bytes": report["totals"]["raw_original_bytes"],
        "extracted_train_bytes": report["totals"]["extracted_train_bytes"],
        "bpe_text_tokens": report["totals"]["bpe_text_tokens"],
        "training_windows": report["totals"]["training_windows"],
        "lean_source_text_token_pct": round(100*report["lean_source_text_token_share"], 3),
        "largest_document_text_token_pct": round(
            100*report["largest_document_text_token_share"], 3),
        "markers": {key: report["totals"][key] for key in marker_counts("")},
        "near_duplicate_candidate_pairs": len(report["near_duplicate_candidates_private"]),
        "sealed_test_text_read": False, "training_updates": 0,
        "private_result_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.expected_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
