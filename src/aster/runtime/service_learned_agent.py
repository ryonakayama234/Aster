"""Model-only ACT v0 recipe: saved DecisionModel -> real tools -> canonical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from aster.evaluator.episode import write_episode_evaluation
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_artifact import load_decision_artifact
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.service_agent import build_executor


RECIPE_ID = "agent-decision-model-v0"
TASK_ID = "act-calculate-store-dev-v0"
SERIALIZER_ID = "aster-decision-input-0"
CANDIDATE_BUILDER_ID = "calculate-and-store-v0"
POLICY_MODE = "model_only"
TASK: dict[str, JsonValue] = {
    "task_id": TASK_ID,
    "kind": "calculate_and_store",
    "operation": "add",
    "left": 23,
    "right": 19,
    "store_as": "answer",
}


def run_learned_agent_recipe(
    root: str | Path,
    artifact_dir: str | Path,
    *,
    max_steps: int = 8,
) -> Path:
    """Run one fixed ACT-v0 episode without training, fallback, or rule intervention."""
    root = Path(root)
    artifact_dir = Path(artifact_dir)
    model, tokenizer, manifest = load_decision_artifact(artifact_dir)
    artifact_id = _require_str(manifest, "artifact_id")
    model_id = _require_str(manifest, "model_id")
    if manifest.get("candidate_builder_id") != CANDIDATE_BUILDER_ID:
        raise ValueError("ACT v0 requires calculate-and-store-v0 candidate builder")
    suite_id = _require_str(manifest, "suite_id")
    suite_sha256 = _require_str(manifest, "suite_sha256")
    if len(suite_sha256) != 64:
        raise ValueError("Decision artifact suite digest is invalid")

    policy = ModelPolicy(model, tokenizer, model_id=model_id)
    context = RuntimeContext(task=TASK)
    recorder = TrajectoryRecorder()
    weights_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    policy_id = f"model-only:{artifact_id}"
    run = RunLog(
        root,
        "agent",
        {
            "recipe_id": RECIPE_ID,
            "policy_id": policy_id,
            "policy_mode": POLICY_MODE,
            "decision_model_artifact_id": artifact_id,
            "model_id": model_id,
            "candidate_builder_id": CANDIDATE_BUILDER_ID,
            "serializer_id": SERIALIZER_ID,
            "suite_id": suite_id,
            "suite_sha256": suite_sha256,
            "task_id": TASK_ID,
            "task": dict(TASK),
            "max_steps": max_steps,
            "model_update": False,
            "fallback_enabled": False,
            "rule_intervention_enabled": False,
            "test": "sealed",
        },
        producer="runtime",
    )

    try:
        trajectory = run_loop(
            policy=policy,
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=context,
            recorder=recorder,
            max_steps=max_steps,
        )
        weights_unchanged = all(
            torch.equal(weights_before[name], tensor.detach().cpu())
            for name, tensor in model.state_dict().items()
        )
        if not weights_unchanged:
            raise RuntimeError("ACT v0 changed DecisionModel weights")

        alignment = _verify_model_only_alignment(policy, trajectory)
        trajectory_path = run.path / "trajectory.jsonl"
        recorder.write_jsonl(trajectory_path)
        decision_path = run.path / "decision-traces.jsonl"
        _write_jsonl(decision_path, [trace.to_dict() for trace in policy.traces])
        evaluation_path = run.path / "evaluation.json"
        episode_evaluation = write_episode_evaluation(evaluation_path, trajectory_path)

        trajectory_payload = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decision_payload = [
            json.loads(line)
            for line in decision_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))

        wiring_gate = {
            "passed": True,
            "artifact_reload_verified": True,
            "weights_unchanged": weights_unchanged,
            "decision_transition_alignment": alignment,
            "model_only": True,
            "fallback_count": 0,
            "rule_intervention_count": 0,
            "serializer_id": SERIALIZER_ID,
            "candidate_builder_id": CANDIDATE_BUILDER_ID,
        }
        capability_gate = {
            "passed": bool(episode_evaluation.task_success and episode_evaluation.goal_verified),
            "task_id": TASK_ID,
            "task_success": episode_evaluation.task_success,
            "goal_verified": episode_evaluation.goal_verified,
            "scope": "predeclared fresh development probe; not sealed test or independent task-family holdout",
        }
        learned_decision = {
            "schema_version": "aster-learned-agent-decision-0",
            "policy_mode": POLICY_MODE,
            "decision_model_artifact_id": artifact_id,
            "model_id": model_id,
            "candidate_builder_id": CANDIDATE_BUILDER_ID,
            "serializer_id": SERIALIZER_ID,
            "suite_id": suite_id,
            "suite_sha256": suite_sha256,
            "model_update": False,
            "fallback_enabled": False,
            "rule_intervention_enabled": False,
            "traces": decision_payload,
            "wiring_gate": wiring_gate,
            "capability_gate": capability_gate,
        }
        agent_bundle = {
            "schema_version": "aster-agent-bundle-0",
            "recipe_id": RECIPE_ID,
            "policy_id": policy_id,
            "task": dict(TASK),
            "trajectory_file": "trajectory.jsonl",
            "decision_trace_file": "decision-traces.jsonl",
            "evaluation_file": "evaluation.json",
            "trajectory": trajectory_payload,
            "decision": learned_decision,
            "evaluation": evaluation_payload,
            "final_memory": dict(context.memory),
            "summary": episode_evaluation.to_dict(),
        }
        _write_json(run.path / "agent-bundle.json", agent_bundle)

        run.event(
            "learned_agent_evaluated",
            {
                "artifact_id": artifact_id,
                "model_id": model_id,
                "wiring_gate_passed": True,
                "capability_gate_passed": capability_gate["passed"],
                "task_success": episode_evaluation.task_success,
            },
        )
        run.finish(
            "completed",
            agent="agent-bundle.json",
            trajectory="trajectory.jsonl",
            decision_traces="decision-traces.jsonl",
            evaluation="evaluation.json",
            test_status="sealed",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _verify_model_only_alignment(policy: ModelPolicy, trajectory) -> bool:
    if len(policy.traces) != len(trajectory.transitions):
        raise RuntimeError("DecisionTrace count does not match Transition count")
    for decision, transition in zip(policy.traces, trajectory.transitions, strict=True):
        if decision.step != transition.step:
            raise RuntimeError("DecisionTrace step does not match Transition step")
        if decision.selected != transition.action:
            raise RuntimeError("Model-selected Action does not match executed Transition Action")
    return True


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Decision artifact field {key!r} must be a non-empty string")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    run_path = run_learned_agent_recipe(args.root, args.artifact)
    print(run_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
