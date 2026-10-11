"""Run OBS v0a G0/G1; optionally diagnose a pinned local DecisionModel (G2).

No training, no corpus read, and no change to existing ACT artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.evaluator.episode import write_episode_evaluation
from aster.model.decision_artifact import load_decision_artifact
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.runtime.observation_recovery import (
    BRANCHES, CANDIDATE_BUILDER_ID, PROTOCOL_ID, SOURCE_BUILDER_ID,
    run_frozen_model, run_reference_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, default=None,
                        help="Existing frozen DecisionModel artifact; omitted means G0/G1 only.")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "runs").is_dir():
        parser.error("Aster root must contain a runs/ directory")
    result = run_reference_checks()
    if args.artifact is None:
        print(json.dumps({**result, "G2": "NOT_RUN_NO_ARTIFACT"}, ensure_ascii=False, indent=2))
        return

    artifact_path = args.artifact.resolve()
    # Check that the artifact remains the ACT v0 lineage. The new builder is an
    # explicit experimental adapter, never silently substituted in production.
    model, tokenizer, manifest = load_decision_artifact(artifact_path)
    if manifest.get("candidate_builder_id") != SOURCE_BUILDER_ID:
        raise ValueError("OBS v0a requires an unchanged calculate-and-store-v0 artifact")
    artifact_id = manifest.get("artifact_id")
    model_id = manifest.get("model_id")
    if not isinstance(artifact_id, str) or not isinstance(model_id, str):
        raise ValueError("Verified DecisionModel artifact is missing identity")
    diagnostic = run_frozen_model(model, tokenizer, model_id=model_id)
    preflight = diagnostic["preflight"]
    if diagnostic["G2"] == "BLOCKED_CONTEXT":
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "G0": "BLOCKED_CONTEXT_FOR_FROZEN_MODEL",
            "G1_reference": result["G1_reference"],
            "G2": "BLOCKED_CONTEXT",
            "artifact_id": artifact_id,
            "preflight": preflight,
            "training_updates": 0,
        }, ensure_ascii=False, indent=2))
        return

    run = RunLog(
        root, "agent", {
            "protocol_id": PROTOCOL_ID,
            "model_artifact_id": artifact_id,
            "source_candidate_builder_id": SOURCE_BUILDER_ID,
            "experimental_candidate_builder_id": CANDIDATE_BUILDER_ID,
            "scripted_prefix": True, "model_only_continuation": True,
            "test": "sealed", "model_update": False,
            "fallback_enabled": False, "rule_intervention_enabled": False,
        }, producer="runtime",
    )
    try:
        branch_evidence = {}
        for branch in BRANCHES:
            evidence = diagnostic["branches"][branch]
            trajectory = evidence["trajectory"]
            recorder = TrajectoryRecorder()
            for transition in trajectory.transitions:
                recorder.record(transition)
            trajectory_file = run.path / f"trajectory-{branch}.jsonl"
            recorder.write_jsonl(trajectory_file)
            evaluation_file = run.path / f"evaluation-{branch}.json"
            persisted_eval = write_episode_evaluation(evaluation_file, trajectory_file)
            if persisted_eval != evidence["episode"]:
                raise AssertionError("Readback evaluation does not match live evidence")
            traces_file = run.path / f"decision-traces-{branch}.jsonl"
            traces = evidence["decision_traces"]
            if len(traces) != len(trajectory.transitions) - 1:
                raise AssertionError("Scripted prefix was counted as a model decision")
            with traces_file.open("w", encoding="utf-8") as stream:
                for trace in traces:
                    stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            branch_evidence[branch] = {
                "trajectory_file": trajectory_file.name,
                "evaluation_file": evaluation_file.name,
                "decision_traces_file": traces_file.name,
                "task_success": persisted_eval.task_success,
                "goal_verified": persisted_eval.goal_verified,
                "model_actions": [t.action.to_dict() for t in trajectory.transitions[1:]],
                "steps_including_scripted_prefix": persisted_eval.steps,
            }

        summary = {
            "protocol_id": PROTOCOL_ID, "G0": "PASS",
            "G1_reference": result["G1_reference"],
            "G2": diagnostic["G2"], "preflight": preflight,
            "branches": branch_evidence,
            "artifact_id": artifact_id, "model_weights_unchanged": diagnostic["weights_unchanged"],
            "training_updates": 0,
            "interpretation": "one scripted-prefix development pair; not independent generalization",
        }
        (run.path / "obs-v0a-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run.finish("completed", protocol_file="obs-v0a-summary.json", G2=diagnostic["G2"])
        print(json.dumps({**summary, "run_path": str(run.path)}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        run.finish("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
