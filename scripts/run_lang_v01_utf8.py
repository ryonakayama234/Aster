"""LANG v0.1: one private, frozen-checkpoint UTF-8 decoding diagnostic.

This is a NEW post-hoc decoding policy, not a rerun of Gate 4, a tuning
exercise, or a new training run. No corpus/weights/completions go to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

# Loaded only from the reviewed Git snapshot after verifying the checkout.
gate4 = None

def load_verified_module(snapshot: bytes, path: Path, name: str, package: str):
    """Execute exactly the Git-verified bytes, bypassing importlib's .pyc cache."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    sys.modules[name] = module
    try:
        exec(compile(snapshot, str(path), "exec"), module.__dict__)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


class PinnedAsterLoader(importlib.machinery.SourceFileLoader):
    """Source-only loader that checks Git bytes before compiling each module."""

    def __init__(self, fullname: str, path: str, root: Path, commit: str,
                 loaded_hashes: dict[str, str]):
        super().__init__(fullname, path)
        self.root, self.commit, self.loaded_hashes = root, commit, loaded_hashes

    def get_code(self, fullname):
        path = Path(self.path).resolve()
        relative = path.relative_to(self.root).as_posix()
        snapshot = path.read_bytes()
        try:
            committed = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.commit}:{relative}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise ImportError("Could not verify Aster source: " + relative) from exc
        if snapshot != committed:
            raise ImportError("Aster module differs from Git source: " + relative)
        self.loaded_hashes[relative] = hashlib.sha256(snapshot).hexdigest()
        # No get_data for .pyc / pycache: compile precisely the verified bytes.
        return self.source_to_code(snapshot, str(path))


class PinnedAsterFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path, commit: str, loaded_hashes: dict[str, str]):
        self.root, self.commit, self.loaded_hashes = root, commit, loaded_hashes

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "aster" and not fullname.startswith("aster."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            raise ImportError("Missing expected Aster module: " + fullname)
        # src/aster/inference, model, and training are legitimate namespace
        # packages without __init__.py; validate every search location.
        if spec.origin is None:
            locations = list(spec.submodule_search_locations or [])
            if not locations or any(
                not Path(location).resolve().is_relative_to(self.root / "src/aster")
                for location in locations
            ):
                raise ImportError("Namespace package escapes reviewed checkout: " + fullname)
            return spec
        source = Path(spec.origin).resolve()
        if source.suffix != ".py" or not source.is_relative_to(self.root / "src/aster"):
            raise ImportError("Unexpected Aster module outside reviewed checkout")
        spec.loader = PinnedAsterLoader(
            fullname, str(source), self.root, self.commit, self.loaded_hashes)
        return spec


def prepare_verified_imports(root: Path, commit: str, snapshots: dict[str, bytes],
                             loaded_hashes: dict[str, str]):
    """Bind Gate 4 and all imported local Aster dependencies to verified bytes."""
    if any(name == "aster" or name.startswith("aster.") for name in sys.modules):
        raise ValueError("Aster modules imported before source attestation")
    sys.meta_path.insert(0, PinnedAsterFinder(root, commit, loaded_hashes))
    gate4_mod = load_verified_module(
        snapshots["scripts/run_lang_v0_gate4.py"],
        root / "scripts/run_lang_v0_gate4.py", "run_lang_v0_gate4_verified", "")
    decoder_mod = load_verified_module(
        snapshots["src/aster/inference/utf8_greedy.py"],
        root / "src/aster/inference/utf8_greedy.py",
        "aster.inference._lang_v01_verified_utf8", "aster.inference")
    return gate4_mod, decoder_mod.generate_utf8_greedy


PRIVATE_OUTPUT = "lang-v01-utf8-private.json"
NEW_POLICY = "UTF-8 strict-prefix masked greedy; BOS excluded; EOS at scalar boundary; sliding context"
EXPECTED_OLD_COUNTS = {
    0: {"outputs": 6, "valid_utf8": 0, "repeated_trigram": 0,
        "train_32byte_match": 0, "eos_stopped": 1, "prompt_in_train": 0},
    200: {"outputs": 6, "valid_utf8": 0, "repeated_trigram": 6,
          "train_32byte_match": 0, "eos_stopped": 0, "prompt_in_train": 0},
}


def read_immutable_baseline(run: Path, report: dict) -> tuple[str, dict]:
    """Pin against the single original local Gate 4 run; never regenerate it."""
    path = run / gate4.PRIVATE_OUTPUT
    if path.is_symlink():
        raise ValueError("Baseline file symlink not allowed")
    snapshot = path.read_bytes()
    old = json.loads(snapshot)
    if (old.get("schema_version") != "aster-lang-gate4-private-0"
            or old.get("run_id") != gate4.RUN_ID
            or old.get("gate3_report_sha256") != gate4.REPORT_SHA256
            or old.get("policy") != gate4.POLICY
            or old.get("max_new_tokens") != gate4.MAX_NEW_TOKENS
            or old.get("prompts") != list(gate4.PROMPTS)):
        raise ValueError("Original frozen Gate 4 policy/identity mismatch")
    entries = old.get("checkpoints")
    if not isinstance(entries, list) or [x.get("step") for x in entries] != list(gate4.STEPS):
        raise ValueError("Original Gate 4 checkpoint steps mismatch")
    baseline: dict[str, dict] = {}
    by_step = {entry["step"]: entry for entry in report["checkpoint_evaluations"]}
    for entry in entries:
        step = entry["step"]
        if entry.get("checkpoint_sha256") != by_step[step]["checkpoint_sha256"]:
            raise ValueError("Gate 4 checkpoint hash differs from Gate 3")
        results = entry.get("results")
        if not isinstance(results, list) or len(results) != len(gate4.PROMPTS):
            raise ValueError("Original Gate 4 prompts/results missing")
        if any(r.get("prompt_index") != i or r.get("prompt") != prompt
               for i, (r, prompt) in enumerate(zip(results, gate4.PROMPTS))):
            raise ValueError("Original Gate 4 prompt ordering mismatch")
        fields = {
            "outputs": len(results),
            "valid_utf8": sum(r["utf8_valid"] is True for r in results),
            "repeated_trigram": sum(r["repeated_token_trigram"] is True for r in results),
            "train_32byte_match": sum(r["train_exact_32_byte_match"] is True for r in results),
            "eos_stopped": sum(r["ended_with_eos"] is True for r in results),
            "prompt_in_train": sum(r["prompt_in_train_window"] is True for r in results),
        }
        if fields != EXPECTED_OLD_COUNTS[step]:
            raise ValueError("Original Gate 4 aggregate counters differ from frozen observation")
        baseline[str(step)] = fields
    return hashlib.sha256(snapshot).hexdigest(), baseline


SOURCE_FILES = (
    "scripts/run_lang_v01_utf8.py",
    "scripts/run_lang_v0_gate4.py",
    "src/aster/inference/utf8_greedy.py",
)


def verify_reviewed_checkout(root: Path, expected_commit: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Require a clean exact Git revision; hash exactly its audited source bytes.

    The expected reviewed/merged commit is supplied in the *user's invocation*
    (not inferred from their checkout). The source hash is checked against Git's
    immutable blobs and stored in the private result for post-hoc auditing.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("An exact 40-hex reviewed commit SHA is required")

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except subprocess.CalledProcessError as exc:
            raise ValueError("Cannot verify reviewed Git sources") from exc

    top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root:
        raise ValueError("Specified root is not the exact checkout root")
    head = git("rev-parse", "HEAD").decode().strip()
    if head != expected_commit:
        raise ValueError("Checkout HEAD differs from exact reviewed commit")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked working tree/index is dirty; preserve private changes")

    verified: dict[str, str] = {}
    source_snapshots: dict[str, bytes] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked audited source: " + relative)
        actual = path.read_bytes()
        committed = git("show", f"{expected_commit}:{relative}")
        if actual != committed:
            raise ValueError("Audited source differs from Git blob: " + relative)
        verified[relative] = hashlib.sha256(actual).hexdigest()
        source_snapshots[relative] = actual
    return verified, source_snapshots


def contains_controls(s: str) -> bool:
    """Flag controls excluding printable spacing of TAB/LF/CR."""
    return any((ord(c) < 32 and c not in "\t\n\r") or 0x7f <= ord(c) < 0xa0 for c in s)


def audit(root: Path, expected_commit: str) -> dict:
    root = root.resolve()
    if root != ROOT.resolve():
        raise ValueError("The --root checkout must match the executing runner's own checkout")
    source_hashes, source_snapshots = verify_reviewed_checkout(root, expected_commit)
    global gate4
    gate4, generate = prepare_verified_imports(
        root, expected_commit, source_snapshots, source_hashes)
    run = root / "runs" / gate4.RUN_ID
    if run.is_symlink() or run.resolve().parent != (root / "runs").resolve():
        raise ValueError("Unexpected frozen Run location")
    output = run / PRIVATE_OUTPUT
    if output.exists() or output.is_symlink():
        raise FileExistsError("LANG v0.1 private result already exists; never overwrite")

    report = gate4.read_frozen_report(run)
    baseline_sha, baseline = read_immutable_baseline(run, report)
    spec, tokenizer, windows, preflight = gate4.preflight(
        root, root / "configs/lang-v0-gate3.json")
    if (report.get("training_view_id") != preflight["view_id"]
            or report.get("tokenizer_artifact_id") != spec["tokenizer_artifact_id"]
            or report.get("pilot_config") != spec["pilot_config"]
            or report.get("preflight") != preflight
            or preflight.get("sealed_test_text_read") is not False):
        raise ValueError("Frozen Gate 3 report/preflight mismatch")
    torch.set_num_threads(spec["pilot_config"]["threads"])
    torch.use_deterministic_algorithms(True)

    experiment = gate4.load_json(run / "experiment.json")
    summary = gate4.load_json(run / "run.json")
    bundle = gate4.load_json(run / "training-bundle.json")
    if (experiment.get("config") != spec["pilot_config"]
            or experiment.get("tokenizer_id") != preflight["tokenizer_payload_sha256"]
            or experiment.get("tokenizer_artifact_id") != spec["tokenizer_artifact_id"]
            or summary.get("status") != "completed" or summary.get("run_id") != gate4.RUN_ID
            or summary.get("inputs", {}).get("tokenizer_artifact_id") != spec["tokenizer_artifact_id"]
            or bundle.get("status") != "completed" or bundle.get("run_id") != gate4.RUN_ID
            or bundle.get("experiment") != experiment):
        raise ValueError("Frozen Run provenance mismatch")
    gate4.verify_scoring_dependencies(experiment, str(torch.__version__), sys.version)
    gate4.verify_run_sources(root, spec, experiment.get("code_sha256"))
    observations = gate4.validate_run_observations(
        bundle.get("observations"), experiment, summary, bundle)
    train_chunks = gate4.verified_training_chunks(
        root, spec, tokenizer, len(windows["train"]))

    pairs: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for step in gate4.STEPS:
        record = next(x for x in report["checkpoint_evaluations"] if x["step"] == step)
        obs = next(x for x in observations if x["step"] == step)
        model, tok = gate4.verify_checkpoint(
            run, record, obs, experiment, spec, preflight["tokenizer_payload_sha256"])
        generated_results = []
        for i, prompt in enumerate(gate4.PROMPTS):
            prompt_ids = tok.encode(prompt, add_bos=True)
            generation = generate(model, tok, prompt_ids, gate4.MAX_NEW_TOKENS)
            raw = gate4.token_bytes(tok, generation.token_ids)
            try:
                continuation = raw.decode("utf-8")
                valid_utf8 = True
            except UnicodeDecodeError:
                continuation = raw.decode("utf-8", errors="replace")
                valid_utf8 = False
            if valid_utf8 != (generation.pending_bytes == b""):
                raise AssertionError("Streaming UTF-8 state and final decode disagree")
            generated_results.append({
                "prompt_index": i, "prompt": prompt, "continuation": continuation,
                "utf8_valid": valid_utf8, "pending_utf8_hex": generation.pending_bytes.hex(),
                "stop_reason": generation.stop_reason,
                "generated_tokens": len(generation.token_ids),
                "ended_with_eos": generation.stop_reason == "eos",
                "contains_control": contains_controls(continuation),
                "repeated_token_trigram": gate4.has_repeated_trigram(generation.token_ids),
                "train_exact_32_byte_match": gate4.has_long_training_match(raw, train_chunks),
                "prompt_in_train_window": any(prompt.encode("utf-8") in c for c in train_chunks),
            })
        pairs.append({"step": step, "checkpoint_sha256": record["checkpoint_sha256"],
                      "results": generated_results})
        counts[str(step)] = {
            "outputs": len(generated_results),
            "valid_utf8": sum(r["utf8_valid"] for r in generated_results),
            "incomplete_utf8_at_budget": sum(r["stop_reason"] == "budget" and
                                             bool(r["pending_utf8_hex"]) for r in generated_results),
            "control_characters": sum(r["contains_control"] for r in generated_results),
            "repeated_trigram": sum(r["repeated_token_trigram"] for r in generated_results),
            "train_32byte_match": sum(r["train_exact_32_byte_match"] for r in generated_results),
            "eos_stopped": sum(r["ended_with_eos"] for r in generated_results),
            "prompt_in_train": sum(r["prompt_in_train_window"] for r in generated_results),
        }
    private = {
        "schema_version": "aster-lang-v01-utf8-private-0",
        "run_id": gate4.RUN_ID, "gate3_report_sha256": gate4.REPORT_SHA256,
        "gate4_baseline_sha256": baseline_sha, "baseline_counts": baseline,
        "reviewed_source_commit": expected_commit,
        "verified_source_sha256": source_hashes,
        "new_policy": NEW_POLICY, "max_new_tokens": gate4.MAX_NEW_TOKENS,
        "prompts": list(gate4.PROMPTS), "checkpoints": pairs,
        "limitations": "Post-hoc decoding-only intervention, not Gate 4 PASS. "
                       "UTF-8 validity is technical correctness, not Japanese fluency. "
                       "Pending bytes on budget cutoff are not silently repaired.",
    }
    gate4.write_private_once(output, private)
    return {
        "status": "utf8_decoding_diagnostic_completed",
        "run_id": gate4.RUN_ID, "gate3_report_sha256": gate4.REPORT_SHA256,
        "gate4_baseline_sha256": baseline_sha,
        "reviewed_source_commit": expected_commit,
        "verified_source_sha256": source_hashes,
        "paired_prompts": len(gate4.PROMPTS),
        "new_policy": NEW_POLICY, "max_new_tokens": gate4.MAX_NEW_TOKENS,
        "original_gate4_by_step": baseline, "utf8_constrained_by_step": counts,
        "training_updates": 0, "sealed_test_text_read": False,
        "private_result_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-commit", required=True,
                        help="Exact 40-hex reviewed/merged commit SHA, never inferred from HEAD")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.expected_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
