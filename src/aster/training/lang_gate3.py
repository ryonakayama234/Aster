"""LANG v0 Gate 3: frozen, token-aligned held-out scoring (no training).

Preflight and evaluation never read sealed test text. The unigram reference is
fit only on the exact train-window *targets*, including EOS but excluding BOS.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

from aster.corpus.pipeline import digest, json_bytes
from aster.model.checkpoint import load_checkpoint
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.dataset import Window, collate, load_training_tokenizer, prepare_windows, tokenizer_payload
from aster.training.pretrain import TrainConfig

VIEW_ID = "6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d"
TOKENIZER_ARTIFACT_ID = "37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513"
CONFIG_FILE = "configs/tinylm-pilot-v0.json"
CONFIG_SHA256 = "445df549ccd4e62359248354b48dacf6796ce11cedf9e5839f59cc96971e83de"
EXPECTED_STEPS = (0, 50, 100, 150, 200)
EXPECTED_TRAIN_WINDOWS = 358
SCHEMA = "aster-lang-gate3-evaluation-0"


class TokenTotals:
    """Accumulate NLL by true target position; never score padding."""

    def __init__(self) -> None:
        self.domains: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"targets": 0, "nll_sum": 0.0, "text_targets": 0,
                     "text_bytes": 0, "text_nll_sum": 0.0,
                     "eos_targets": 0, "eos_nll_sum": 0.0}
        )
        self.windows = 0

    def add(self, window: Window, nll: Sequence[float], tokenizer: AsterTokenizer) -> None:
        targets = window.ids[1:]
        if len(nll) != len(targets) or not targets:
            raise ValueError("One NLL is required per real target")
        row = self.domains[window.domain]
        for token_id, loss in zip(targets, nll):
            value = float(loss)
            if not math.isfinite(value) or value < -1e-5:
                raise ValueError("Invalid target NLL")
            if token_id == tokenizer.bos_id:
                raise ValueError("BOS must not occur as a prediction target")
            if not 0 <= token_id < tokenizer.vocab_size:
                raise ValueError("Unknown target token ID")
            row["targets"] += 1
            row["nll_sum"] += value
            if token_id == tokenizer.eos_id:
                row["eos_targets"] += 1
                row["eos_nll_sum"] += value
            else:
                if token_id not in tokenizer.model.vocab:
                    raise ValueError("Unexpected non-text special target")
                size = len(tokenizer.model.vocab[token_id])
                if size == 0:
                    raise ValueError("Empty byte token")
                row["text_targets"] += 1
                row["text_bytes"] += size
                row["text_nll_sum"] += value
        self.windows += 1

    def result(self) -> dict:
        def summarize(row: dict[str, float | int]) -> dict:
            targets, byte_count = int(row["targets"]), int(row["text_bytes"])
            return {
                **row,
                "nll": float(row["nll_sum"]) / targets if targets else None,
                "text_bits_per_byte": (
                    float(row["text_nll_sum"]) / (math.log(2) * byte_count)
                    if byte_count else None
                ),
            }

        total: dict[str, float | int] = {
            key: sum(row[key] for row in self.domains.values())
            for key in ("targets", "nll_sum", "text_targets", "text_bytes",
                        "text_nll_sum", "eos_targets", "eos_nll_sum")
        }
        return {
            **summarize(total), "windows": self.windows,
            "by_domain": {name: summarize(row) for name, row in sorted(self.domains.items())},
        }


def unigram_counts(train: Sequence[Window], tokenizer: AsterTokenizer) -> list[int]:
    counts = [0] * tokenizer.vocab_size
    if not train:
        raise ValueError("No train windows")
    for window in train:
        for token_id in window.ids[1:]:
            if token_id == tokenizer.bos_id or not 0 <= token_id < tokenizer.vocab_size:
                raise ValueError("Unexpected train target")
            counts[token_id] += 1
    return counts


def score_unigram(
    train: Sequence[Window], dev: Sequence[Window], tokenizer: AsterTokenizer,
) -> tuple[dict, str]:
    counts = unigram_counts(train, tokenizer)
    denominator = sum(counts) + tokenizer.vocab_size
    losses = [math.log(denominator) - math.log(n + 1) for n in counts]
    result = TokenTotals()
    for window in dev:
        result.add(window, [losses[token_id] for token_id in window.ids[1:]], tokenizer)
    # Hash counts, not private training text; never use dev targets for fitting.
    return result.result(), digest(json_bytes({
        "schema_version": "aster-unigram-laplace-1",
        "vocab_size": tokenizer.vocab_size, "train_target_counts": counts,
    }))


@torch.no_grad()
def score_model(
    model: torch.nn.Module, dev: Sequence[Window], tokenizer: AsterTokenizer, batch_size: int,
) -> dict:
    if batch_size < 1:
        raise ValueError("Invalid batch size")
    model.eval()
    result = TokenTotals()
    for start in range(0, len(dev), batch_size):
        batch = dev[start:start + batch_size]
        x, y = collate(batch, tokenizer.eos_id)
        logits = model(x)
        loss = F.cross_entropy(
            logits.transpose(1, 2), y, reduction="none", ignore_index=-100
        )
        for i, window in enumerate(batch):
            length = len(window.ids) - 1
            result.add(window, loss[i, :length].tolist(), tokenizer)
    return result.result()


def preflight(root: Path) -> dict:
    root = root.resolve()
    config_path = root / CONFIG_FILE
    config_raw = config_path.read_bytes()
    if digest(config_raw) != CONFIG_SHA256:
        raise ValueError("Frozen Gate 3 pilot config digest mismatch")
    config = TrainConfig(**json.loads(config_raw))
    if (config.mode != "pilot" or config.steps != 200 or config.eval_every != 50
            or config.context_length != 128 or config.batch_size != 8
            or (config.width, config.heads, config.layers, config.seed) != (64, 4, 2, 42)):
        raise ValueError("Gate 3 model/training parameters were changed")
    view = root / "data/training" / VIEW_ID
    tok_path = root / "artifacts/tokenizers/lang-v0" / TOKENIZER_ARTIFACT_ID
    if not view.is_dir() or not tok_path.is_dir():
        raise FileNotFoundError("Pinned local WSL training view or tokenizer is missing")
    tokenizer, payload = load_training_tokenizer(
        tok_path, VIEW_ID, expected_artifact_id=TOKENIZER_ARTIFACT_ID
    )
    if tokenizer.vocab_size != 514:
        raise ValueError("Gate 3 vocabulary drift")
    # prepare_windows verifies train/dev hashes and cross-split metadata but
    # deliberately does NOT read any sealed test text bytes.
    splits = prepare_windows(view, tokenizer, config.context_length)
    if len(splits["train"]) != EXPECTED_TRAIN_WINDOWS or not splits["dev"]:
        raise ValueError("Frozen train/dev window inventory mismatch")
    if {w.record_id for w in splits["train"]} & {w.record_id for w in splits["dev"]}:
        raise ValueError("Train/dev record identity overlap")
    return {
        "view": view, "tokenizer_path": tok_path,
        "config": config, "tokenizer": tokenizer, "tokenizer_payload": payload,
        "train": splits["train"], "dev": splits["dev"],
    }


def preflight_report(root: Path) -> dict:
    prepared = preflight(root)
    return {
        "schema_version": SCHEMA, "mode": "preflight_only",
        "training_view_id": VIEW_ID, "tokenizer_artifact_id": TOKENIZER_ARTIFACT_ID,
        "config_sha256": CONFIG_SHA256,
        "tokenizer_payload_sha256": digest(json_bytes(prepared["tokenizer_payload"])),
        "view_manifest_sha256": digest((prepared["view"] / "manifest.json").read_bytes()),
        "train_windows": len(prepared["train"]), "dev_windows": len(prepared["dev"]),
        "train_targets": sum(len(w.ids) - 1 for w in prepared["train"]),
        "dev_targets": sum(len(w.ids) - 1 for w in prepared["dev"]),
        "sealed_test_text_read": False, "model_trained": False,
    }


def evaluate_completed_run(root: Path, training_run: Path) -> Path:
    """Post-hoc *measurement* only: never retrain, inspect test, or select checkpoints."""
    root = root.resolve()
    prepared = preflight(root)
    run = training_run.resolve()
    if run.parent != (root / "runs").resolve() or not run.is_dir():
        raise ValueError("Expected an existing immediate runs/<run_id> directory")

    def load_json(filename: str) -> dict:
        value = json.loads((run / filename).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Malformed training evidence: " + filename)
        return value

    summary = load_json("run.json")
    experiment = load_json("experiment.json")
    bundle = load_json("training-bundle.json")
    config = prepared["config"]
    tok_id = digest(json_bytes(prepared["tokenizer_payload"]))
    if (summary.get("status") != "completed"
            or bundle.get("status") != "completed"
            or bundle.get("reload_exact_match") is not True
            or experiment.get("view_id") != VIEW_ID
            or experiment.get("tokenizer_id") != tok_id
            or experiment.get("config") != asdict(config)
            or bundle.get("experiment") != experiment):
        raise ValueError("Training run provenance/complete-reload gate failed")
    observations = bundle.get("observations")
    if not isinstance(observations, list) or [o.get("step") for o in observations] != list(EXPECTED_STEPS):
        raise ValueError("Expected exactly the frozen five checkpoints")
    if any(o.get("dev") is None for o in observations):
        raise ValueError("Missing dev evaluation in pilot")
    base, counts_digest = score_unigram(prepared["train"], prepared["dev"], prepared["tokenizer"])
    checkpoints = []
    for obs in observations:
        step = obs["step"]
        filename = f"checkpoint-{step:06d}.pt"
        if obs.get("checkpoint") != filename:
            raise ValueError("Checkpoint name mismatch")
        checkpoint = run / filename
        expected_hash = obs.get("checkpoint_sha256")
        actual_hash = digest(checkpoint.read_bytes())
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError("Checkpoint hash mismatch")
        model, loaded_tokenizer, state = load_checkpoint(checkpoint, tok_id)
        metadata = state.get("metadata", {})
        if (metadata.get("step") != step or metadata.get("view_id") != VIEW_ID
                or metadata.get("tokenizer_id") != tok_id
                or metadata.get("config") != asdict(config)
                or loaded_tokenizer.vocab_size != prepared["tokenizer"].vocab_size):
            raise ValueError("Checkpoint model/training lineage mismatch")
        measured = score_model(model, prepared["dev"], prepared["tokenizer"], config.batch_size)
        if measured["targets"] != base["targets"] or measured["text_bytes"] != base["text_bytes"]:
            raise ValueError("Inconsistent evaluation target accounting")
        checkpoints.append({
            "step": step, "checkpoint_sha256": actual_hash,
            "train_nll": obs["train"]["loss"], "dev_nll_in_training": obs["dev"]["loss"],
            "tokens_seen": obs["tokens_seen"], "heldout": measured,
        })
    report = {
        "schema_version": SCHEMA, "mode": "fixed_pilot_analysis",
        "training_run_id": run.name,
        "source_code_sha256": experiment.get("code_sha256"),
        "evaluation_code_sha256": digest(Path(__file__).read_bytes()),
        "training_view_id": VIEW_ID, "tokenizer_artifact_id": TOKENIZER_ARTIFACT_ID,
        "tokenizer_payload_sha256": tok_id, "config_sha256": CONFIG_SHA256,
        "sealed_test_used": False,
        "train_windows": len(prepared["train"]), "dev_windows": len(prepared["dev"]),
        "unigram": {"fit": "train-target-only-plus-one", "counts_sha256": counts_digest,
                    "heldout": base},
        "checkpoints": checkpoints,
        "train_seconds": summary.get("seconds"), "peak_rss_kib": summary.get("peak_rss_kib"),
        "reload_exact_match": True,
        "interpretation": "Pilot evidence from the frozen dev slice only; no dialogue or agent claim.",
    }
    output = run / "lang-gate3-evaluation.json"
    encoded = json_bytes(report)
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("Existing Gate 3 evaluation differs; refusing overwrite")
    output.write_bytes(encoded)
    return output
