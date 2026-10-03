"""Versioned, reloadable artifacts for AsterDecision models.

The artifact stores inference state and the identities needed to interpret that state.
Optimizer state is intentionally excluded: reloadable inference is not training resume.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import cast

import torch

from aster.corpus.pipeline import digest, json_bytes
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.tokenizer.artifact import AsterTokenizer, load_tokenizer, save_tokenizer


SCHEMA_VERSION = "aster-decision-model-artifact-0"
_MODEL_FILE = "model.pt"
_TOKENIZER_DIR = "tokenizer"
_TOKENIZER_FILES = (
    "manifest.json",
    "vocab.json",
    "merges.json",
    "special_tokens.json",
)


def save_decision_artifact(
    output_dir: str | Path,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    *,
    model_id: str,
    candidate_builder_id: str,
    suite_id: str,
    suite_sha256: str,
    train_config: dict[str, object],
    temperature: float,
    autonomous_threshold: float,
    fallback_threshold: float,
    initialization: str = "scratch",
    source_git_sha: str | None = None,
) -> dict[str, object]:
    """Persist one inference-ready DecisionModel and its interpretation metadata."""

    if not model_id or not candidate_builder_id or not suite_id:
        raise ValueError("Decision artifact identities must not be empty")
    if len(suite_sha256) != 64:
        raise ValueError("suite_sha256 must be a SHA-256 hex digest")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not 0.0 <= fallback_threshold <= autonomous_threshold <= 1.0:
        raise ValueError("routing thresholds must satisfy 0 <= fallback <= autonomous <= 1")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    tokenizer_dir = output / _TOKENIZER_DIR
    save_tokenizer(tokenizer, tokenizer_dir)
    tokenizer_sha256 = _directory_digest(tokenizer_dir, _TOKENIZER_FILES)

    state_path = output / _MODEL_FILE
    temporary = output / f"{_MODEL_FILE}.tmp"
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, temporary)
    temporary.replace(state_path)
    model_sha256 = _sha256_file(state_path)

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_config": asdict(model.backbone.config),
        "decision_head": {
            "type": "linear_scalar_v0",
            "width": model.backbone.config.width,
        },
        "model_state_file": _MODEL_FILE,
        "model_state_sha256": model_sha256,
        "tokenizer_dir": _TOKENIZER_DIR,
        "tokenizer_sha256": tokenizer_sha256,
        "candidate_builder_id": candidate_builder_id,
        "suite_id": suite_id,
        "suite_sha256": suite_sha256,
        "train_config": train_config,
        "initialization": initialization,
        "source_git_sha": source_git_sha,
        "calibration": {
            "fit_split": "calibration",
            "temperature": float(temperature),
        },
        "routing": {
            "autonomous_threshold": float(autonomous_threshold),
            "fallback_threshold": float(fallback_threshold),
        },
        "resume_supported": False,
    }
    manifest["artifact_id"] = _artifact_id(manifest)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def load_decision_artifact(
    input_dir: str | Path,
) -> tuple[DecisionModel, AsterTokenizer, dict[str, object]]:
    """Load and verify one DecisionModel artifact without changing caller RNG state."""

    import json

    input_path = Path(input_dir)
    raw_manifest = json.loads((input_path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Decision artifact manifest must be a JSON object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported decision artifact schema")
    if manifest.get("artifact_id") != _artifact_id(manifest):
        raise ValueError("Decision artifact identity mismatch")

    model_file = _require_str(manifest, "model_state_file")
    if model_file != _MODEL_FILE:
        raise ValueError("Unexpected decision model state filename")
    state_path = input_path / model_file
    if _sha256_file(state_path) != _require_str(manifest, "model_state_sha256"):
        raise ValueError("Decision model state digest mismatch")

    tokenizer_dir_name = _require_str(manifest, "tokenizer_dir")
    if tokenizer_dir_name != _TOKENIZER_DIR:
        raise ValueError("Unexpected decision tokenizer directory")
    tokenizer_dir = input_path / tokenizer_dir_name
    if _directory_digest(tokenizer_dir, _TOKENIZER_FILES) != _require_str(
        manifest, "tokenizer_sha256"
    ):
        raise ValueError("Decision tokenizer digest mismatch")
    tokenizer = load_tokenizer(tokenizer_dir)

    raw_config = manifest.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Decision artifact model_config must be an object")
    config = ModelConfig(**raw_config)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Decision artifact tokenizer/model vocabulary mismatch")

    with torch.random.fork_rng(devices=[]):
        model = DecisionModel(TinyLM(config))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Decision model state must be a mapping")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, manifest


def _artifact_id(manifest: dict[str, object]) -> str:
    identity = {key: value for key, value in manifest.items() if key != "artifact_id"}
    return f"decision_model:{digest(json_bytes(identity))}"


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _directory_digest(path: Path, names: tuple[str, ...]) -> str:
    value = hashlib.sha256()
    for name in names:
        item = path / name
        if not item.is_file() or item.is_symlink():
            raise ValueError(f"Decision artifact tokenizer file is missing: {name}")
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Decision artifact field {key!r} must be a non-empty string")
    return value
