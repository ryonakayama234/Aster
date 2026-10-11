"""Audit OBS v0b frozen-model observation ablations (offline score-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from aster.model.decision_artifact import load_decision_artifact
from aster.records.runlog import RunLog
from aster.runtime.observation_ablation import (
    PINNED_ARTIFACT_ID, PROTOCOL_ID, build_probes, run_probe,
)
from aster.runtime.observation_recovery import SOURCE_BUILDER_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, help="Previously pinned DecisionModel, not a new winner")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "runs").is_dir():
        parser.error("Aster root must contain runs/ directory")
    if args.artifact is None:
        # Without a local model only structural input construction is checked.
        probes = build_probes()
        print(json.dumps({"protocol_id": PROTOCOL_ID, "G0": "PASS",
                          "model": "NOT_RUN_NO_ARTIFACT",
                          "conditions": list(probes)}, ensure_ascii=False, indent=2))
        return

    path = args.artifact if args.artifact.is_absolute() else root / args.artifact
    model, tokenizer, manifest = load_decision_artifact(path.resolve())
    if manifest.get("candidate_builder_id") != SOURCE_BUILDER_ID:
        raise ValueError("Unexpected source candidate builder")
    if manifest.get("artifact_id") != PINNED_ARTIFACT_ID:
        raise ValueError("OBS v0b refuses a different model (no checkpoint search)")

    diagnostic = run_probe(model, tokenizer)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    summary = {
        **diagnostic, "artifact_id": PINNED_ARTIFACT_ID,
        "source_git_sha": git_sha,
        "artifact_source_git_sha": manifest.get("source_git_sha"),
        "model_state_sha256": manifest.get("model_state_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "candidate_builder_id": "obs-v0b-offline-input-intervention",
    }
    if diagnostic["status"] == "BLOCKED_CONTEXT":
        # Preflight failure is not a model outcome or an executable episode.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    run = RunLog(
        root, "agent", {
            "protocol_id": PROTOCOL_ID, "mode": "offline_score_only",
            "artifact_id": PINNED_ARTIFACT_ID, "weights_update": False,
            "test": "sealed", "source_git_sha": git_sha,
        }, producer="runtime",
    )
    try:
        (run.path / "obs-v0b-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        run.finish("completed", protocol_file="obs-v0b-summary.json", outcome=diagnostic["status"])
    except BaseException as error:
        run.finish("failed", error_type=type(error).__name__, error=str(error))
        raise
    print(json.dumps({**summary, "run_path": str(run.path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
