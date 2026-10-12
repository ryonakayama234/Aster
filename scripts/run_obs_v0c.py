"""Run OBS v0c reference gates, optionally a PINNED local frozen-model trial.

Default mode runs G0/G1 only. --artifact explicitly opts into the local G2
diagnostic after protocol review. This script never trains or reads sealed tests.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from aster.evaluator.episode import write_episode_evaluation
from aster.model.decision_artifact import load_decision_artifact
from aster.records.decision import serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.runtime.observation_reconstruction import (
    CANDIDATE_BUILDER_ID, PINNED_ARTIFACT_ID, PROTOCOL_ID,
    SOURCE_BUILDER_ID, run_frozen_model, run_reference_checks,
)
from aster.runtime.state import RuntimeState


def _current_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    sha = process.stdout.strip()
    if len(sha) != 40:
        raise ValueError("Aster source Git SHA is missing")
    return sha


def _decision_input_digests(trajectory: Trajectory, traces) -> list[dict]:
    """Hash exact model-visible candidate inputs, not evaluator/teacher values."""
    result = []
    for trace in traces:
        transition = trajectory.transitions[trace.step]
        state_payload = transition.state_before
        state = RuntimeState(
            task=state_payload["task"],
            memory=state_payload["memory"],
            step=int(state_payload["step"]),
        )
        prior = Trajectory(trajectory.transitions[:trace.step])
        digests = [
            sha256(serialize_decision_input(state, prior, candidate).encode("utf-8")).hexdigest()
            for candidate in trace.candidates
        ]
        if trace.selected != transition.action:
            raise AssertionError("Trace/action mismatch during serialized-input audit")
        result.append({
            "step": trace.step, "candidate_input_sha256": digests,
            "selected_index": trace.selected_index,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--artifact", type=Path, default=None,
        help="Explicit G2 only: exact, previously pinned local DecisionModel directory.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "runs").is_dir():
        parser.error("Aster root must contain a runs/ directory")
    gates = run_reference_checks()
    if args.artifact is None:
        print(json.dumps({
            **gates, "G2": "NOT_RUN_NO_ARTIFACT", "training_updates": 0,
            "evidence_scope": "development",
        }, indent=2, ensure_ascii=False))
        return

    # Refuse a different checkpoint even if it has a compatible model config.
    model, tokenizer, manifest = load_decision_artifact(args.artifact.resolve())
    artifact_id = manifest.get("artifact_id")
    model_id = manifest.get("model_id")
    if artifact_id != PINNED_ARTIFACT_ID:
        raise ValueError("OBS v0c requires the previously pinned DecisionModel")
    if manifest.get("candidate_builder_id") != SOURCE_BUILDER_ID:
        raise ValueError("OBS v0c cannot relabel the source ACT builder")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Pinned DecisionModel model_id missing")
    source_sha = _current_commit(root)
    result = run_frozen_model(model, tokenizer, model_id=model_id)
    if result["G2"] == "BLOCKED_CONTEXT":
        print(json.dumps({
            **gates, "G2": "BLOCKED_CONTEXT", "G0_model_preflight": "BLOCKED_CONTEXT",
            "preflight": result["preflight"],
            "model_artifact_id": artifact_id, "source_git_sha": source_sha,
            "training_updates": 0,
        }, indent=2, ensure_ascii=False))
        return

    run = RunLog(
        root, "agent", {
            "protocol_id": PROTOCOL_ID,
            "experiment_mode": "trial",
            "evidence_scope": "development",
            "source_git_sha": source_sha,
            "model_artifact_id": artifact_id,
            "source_artifact_model_sha256": manifest["model_state_sha256"],
            "source_artifact_tokenizer_sha256": manifest["tokenizer_sha256"],
            "source_artifact_git_sha": manifest.get("source_git_sha"),
            "source_candidate_builder_id": SOURCE_BUILDER_ID,
            "experimental_candidate_builder_id": CANDIDATE_BUILDER_ID,
            "scripted_prefix": True,
            "model_only_continuation": True,
            "sealed_test_access": False,
            "fallback_enabled": False,
            "rule_intervention_enabled": False,
            "model_update": False,
        }, producer="runtime",
    )
    try:
        trajectory = result["trajectory"]
        recorder = TrajectoryRecorder()
        for transition in trajectory.transitions:
            recorder.record(transition)
        trajectory_path = run.path / "trajectory.jsonl"
        recorder.write_jsonl(trajectory_path)
        persisted = write_episode_evaluation(run.path / "evaluation.json", trajectory_path)
        if persisted != result["episode"]:
            raise AssertionError("Persisted trajectory evaluation differs from live result")

        trace_path = run.path / "decision-traces.jsonl"
        with trace_path.open("w", encoding="utf-8") as stream:
            for trace in result["decision_traces"]:
                stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        digests = _decision_input_digests(trajectory, result["decision_traces"])
        summary = {
            "protocol_id": PROTOCOL_ID,
            "experiment_mode": "trial", "evidence_scope": "development",
            "G0_structure": gates["G0_structure"],
            "G1_reference": gates["G1_reference"],
            "G2": result["G2"], "preflight": result["preflight"],
            "artifact_id": artifact_id, "source_git_sha": source_sha,
            "model_weights_unchanged": result["weights_unchanged"],
            "training_updates": 0, "model_actions": result["selected_recovery_actions"],
            "decision_input_digests": digests,
            "task_success": persisted.task_success,
            "goal_verified": persisted.goal_verified,
            "stop_reason": persisted.stop_reason,
            "scripted_prefix_steps": 1,
            "model_decision_steps": len(result["decision_traces"]),
            "trajectory_file": trajectory_path.name,
            "evaluation_file": "evaluation.json",
            "decision_traces_file": trace_path.name,
            "interpretation": "Single scripted-prefix development case, not generalization",
        }
        (run.path / "obs-v0c-summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        run.finish("completed", G2=result["G2"], summary_file="obs-v0c-summary.json")
        print(json.dumps({**summary, "run_path": str(run.path)}, indent=2, ensure_ascii=False))
    except BaseException as exc:
        run.finish("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
