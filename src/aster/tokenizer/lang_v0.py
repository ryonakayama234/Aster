"""LANG v0 train-only BPE build and tokenizer audit.

The tokenizer is fit only on selected training-view samples. Dev is audited without
fitting. Test content remains sealed and is never read by this module.
"""

from __future__ import annotations

import json
import math
import os
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.tokenizer.artifact import AsterTokenizer, save_tokenizer, train_aster_tokenizer
from aster.training.view import build_view


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_documents(rows: list[dict], tokenizer: AsterTokenizer, context_length: int) -> dict:
    bytes_total = 0
    chars_total = 0
    bpe_tokens_total = 0
    byte_tokens_total = 0
    lengths: list[int] = []
    windows_total = 0
    roundtrip_failures = 0

    for row in rows:
        text = row["text"]
        encoded = tokenizer.encode(text)
        byte_count = len(text.encode("utf-8"))
        char_count = len(text)
        token_count = len(encoded)

        bytes_total += byte_count
        chars_total += char_count
        bpe_tokens_total += token_count
        byte_tokens_total += byte_count
        lengths.append(token_count)
        windows_total += math.ceil(token_count / context_length) if token_count else 0

        if tokenizer.decode(encoded) != text:
            roundtrip_failures += 1

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "documents": len(rows),
        "bytes": bytes_total,
        "characters": chars_total,
        "byte_baseline": {
            "tokens": byte_tokens_total,
            "bytes_per_token": ratio(bytes_total, byte_tokens_total),
            "tokens_per_character": ratio(byte_tokens_total, chars_total),
        },
        "bpe": {
            "tokens": bpe_tokens_total,
            "bytes_per_token": ratio(bytes_total, bpe_tokens_total),
            "tokens_per_character": ratio(bpe_tokens_total, chars_total),
        },
        "document_bpe_tokens": {
            "min": min(lengths) if lengths else None,
            "median": quantile(lengths, 0.5),
            "p95": quantile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
        },
        "context_length": context_length,
        "nonoverlap_context_windows": windows_total,
        "roundtrip_failures": roundtrip_failures,
    }


def merge_audit(tokenizer: AsterTokenizer, limit: int = 20) -> dict:
    ranked = []
    for rank, (pair, new_id) in enumerate(tokenizer.model.merges[:limit], 1):
        token = tokenizer.model.vocab[new_id]
        ranked.append({
            "rank": rank,
            "left": pair[0],
            "right": pair[1],
            "new_id": new_id,
            "byte_length": len(token),
            "token_hex": token.hex(),
        })

    longest = []
    for pair, new_id in sorted(
        tokenizer.model.merges,
        key=lambda item: (-len(tokenizer.model.vocab[item[1]]), item[1]),
    )[:limit]:
        token = tokenizer.model.vocab[new_id]
        longest.append({
            "left": pair[0],
            "right": pair[1],
            "new_id": new_id,
            "byte_length": len(token),
            "token_hex": token.hex(),
        })

    return {
        "merge_count": len(tokenizer.model.merges),
        "first_merges": ranked,
        "longest_merges": longest,
    }


def load_audit_rows(view: Path, samples: list[dict]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    sealed_test_records = 0
    for sample in samples:
        split = sample["split"]
        if split == "test":
            sealed_test_records += 1
            continue
        if split not in {"train", "dev"}:
            raise ValueError(f"Unexpected selected split: {split}")
        text_path = Path(sample["text_path"])
        if text_path.is_absolute() or ".." in text_path.parts:
            raise ValueError("training-view text_path must be relative and contained")
        text_file = (view / text_path).resolve()
        if not text_file.is_relative_to(view.resolve()):
            raise ValueError("training-view text_path escaped view")
        rows.append({
            "record_id": sample["record_id"],
            "split": split,
            "domain": sample["domain"],
            "text": text_file.read_text(encoding="utf-8"),
        })
    return rows, sealed_test_records


def publish_artifact(
    root: Path,
    tokenizer: AsterTokenizer,
    *,
    view_id: str,
    recipe_sha256: str,
    split_contract_sha256: str | None,
    audit: dict,
) -> Path:
    base = root / "artifacts/tokenizers/lang-v0"
    base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".building-", dir=base) as temp:
        stage = Path(temp) / "artifact"
        stage.mkdir()
        save_tokenizer(tokenizer, stage)

        provenance = {
            "schema_version": "aster-tokenizer-provenance-0",
            "experiment": "lang-v0",
            "training_view_id": view_id,
            "training_recipe_sha256": recipe_sha256,
            "split_contract_sha256": split_contract_sha256,
            "fit_split": "train",
            "audited_splits": ["train", "dev"],
            "sealed_test_content_read": False,
            "target_vocab_size": tokenizer.manifest.target_vocab_size,
            "min_pair_frequency": tokenizer.manifest.min_pair_frequency,
        }
        (stage / "provenance.json").write_bytes(json_bytes(provenance))
        (stage / "audit.json").write_bytes(json_bytes(audit))

        files = {
            path.name: digest(path.read_bytes())
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        identity = {
            "schema_version": "aster-tokenizer-artifact-0",
            "training_view_id": view_id,
            "files": files,
        }
        identity_bytes = json_bytes(identity)
        artifact_id = digest(identity_bytes)
        (stage / "artifact.json").write_bytes(identity_bytes)
        destination = base / artifact_id

        if destination.exists():
            expected = {p.name for p in stage.iterdir() if p.is_file()}
            actual = {p.name for p in destination.iterdir() if p.is_file()}
            if expected != actual or any(
                (destination / name).read_bytes() != (stage / name).read_bytes()
                for name in expected
            ):
                raise ValueError("Existing tokenizer artifact differs; refusing overwrite")
            return destination

        os.rename(stage, destination)
        return destination


def run(
    root: Path,
    *,
    recipe_path: Path | None = None,
    target_vocab_size: int = 512,
    min_pair_frequency: int = 2,
    context_length: int = 128,
) -> tuple[Path, Path]:
    root = root.resolve()
    recipe_path = recipe_path or root / "configs/lang-v0-training.json"
    if not recipe_path.is_absolute():
        recipe_path = root / recipe_path

    runlog = RunLog(root, "lang_v0_tokenizer", {
        "recipe_path": str(recipe_path),
        "target_vocab_size": target_vocab_size,
        "min_pair_frequency": min_pair_frequency,
        "context_length": context_length,
    })
    try:
        view, training_run = build_view(root, recipe_path)
        view_id = view.name
        view_manifest = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
        samples = read_jsonl(view / "samples.jsonl")
        rows, sealed_test_records = load_audit_rows(view, samples)

        train_rows = [row for row in rows if row["split"] == "train"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
        if not train_rows:
            raise ValueError("LANG v0 tokenizer requires at least one train sample")
        if not dev_rows:
            raise ValueError("LANG v0 tokenizer requires a nonempty dev split")

        start = time.perf_counter()
        tokenizer = train_aster_tokenizer(
            [row["text"] for row in train_rows],
            target_vocab_size=target_vocab_size,
            min_pair_frequency=min_pair_frequency,
            name="AsterTokenizer",
            version="lang-v0",
        )
        elapsed = time.perf_counter() - start
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["split"]].append(row)
            grouped[f"{row['split']}:{row['domain']}"].append(row)

        metrics = {
            key: summarize_documents(value, tokenizer, context_length)
            for key, value in sorted(grouped.items())
        }
        if any(summary["roundtrip_failures"] for summary in metrics.values()):
            raise ValueError("Tokenizer round-trip failure detected")

        audit = {
            "schema_version": "aster-lang-tokenizer-audit-0",
            "experiment": "lang-v0",
            "training_view_id": view_id,
            "fit_split": "train",
            "audited_splits": ["train", "dev"],
            "sealed_test_records": sealed_test_records,
            "sealed_test_content_read": False,
            "target_vocab_size": target_vocab_size,
            "actual_bpe_vocab_size": len(tokenizer.model.vocab),
            "total_vocab_with_special_tokens": tokenizer.vocab_size,
            "min_pair_frequency": min_pair_frequency,
            "metrics": metrics,
            "merges": merge_audit(tokenizer),
        }

        artifact = publish_artifact(
            root,
            tokenizer,
            view_id=view_id,
            recipe_sha256=view_manifest["recipe_sha256"],
            split_contract_sha256=view_manifest.get("split_contract_sha256"),
            audit=audit,
        )

        resource_report = {
            "training_seconds": elapsed,
            "peak_rss_kib": peak_rss_kib,
            "training_view": str(view),
            "training_view_run": str(training_run),
            "tokenizer_artifact": str(artifact),
        }
        (runlog.path / "resource.json").write_bytes(json_bytes(resource_report))
        (runlog.path / "audit.json").write_bytes(json_bytes(audit))
        runlog.finish(
            "completed",
            tokenizer_artifact_id=artifact.name,
            training_view_id=view_id,
            training_seconds=elapsed,
            peak_rss_kib=peak_rss_kib,
        )
        return artifact, runlog.path
    except BaseException as error:
        runlog.finish("failed", error_type=type(error).__name__, error=str(error))
        raise
