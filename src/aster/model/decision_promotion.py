"""Promote a verified Decision fit checkpoint into an inference-ready DecisionModel artifact."""

from __future__ import annotations

from pathlib import Path

from aster.model.decision_artifact import save_decision_artifact
from aster.training.decision_fit import load_decision_fit_checkpoint

CANDIDATE_BUILDER_ID = "calculate-and-store-v0"
SERIALIZER_ID = "aster-decision-input-0"
SUPPORTED_SUITE_ID = "calculate-and-store-v0"


def promote_decision_fit_checkpoint(
    source: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Convert one verified fit checkpoint without inventing calibration evidence."""
    model, tokenizer, manifest = load_decision_fit_checkpoint(source)

    suite_id = _require_str(manifest, "suite_id")
    if suite_id != SUPPORTED_SUITE_ID:
        raise ValueError("ACT v0 promotion only supports calculate-and-store-v0 checkpoints")
    suite_sha256 = _require_str(manifest, "suite_sha256")
    checkpoint_id = _require_str(manifest, "checkpoint_id")
    source_git_sha = manifest.get("source_git_sha")
    if source_git_sha is not None and not isinstance(source_git_sha, str):
        raise ValueError("Decision fit source_git_sha is invalid")
    study_config = manifest.get("study_config")
    arm = manifest.get("arm")
    if not isinstance(study_config, dict) or not isinstance(arm, dict):
        raise ValueError("Decision fit checkpoint lineage is incomplete")

    return save_decision_artifact(
        output_dir,
        model,
        tokenizer,
        model_id=checkpoint_id,
        candidate_builder_id=CANDIDATE_BUILDER_ID,
        suite_id=suite_id,
        suite_sha256=suite_sha256,
        train_config={
            "source": "decision_fit_checkpoint",
            "study_config": dict(study_config),
            "arm": dict(arm),
        },
        temperature=None,
        autonomous_threshold=None,
        fallback_threshold=None,
        initialization="promoted_fit_checkpoint",
        source_git_sha=source_git_sha,
        source_checkpoint_id=checkpoint_id,
        serializer_id=SERIALIZER_ID,
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Decision fit checkpoint field {key!r} must be a non-empty string")
    return value
