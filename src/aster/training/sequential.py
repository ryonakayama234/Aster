"""Model-only sequential evaluation for LEARN-v0 candidate arms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch

from aster.agent.policy import Policy
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.protocol import StepEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_head import DecisionModel
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import AsterTokenizer
from aster.tools.executor import ToolExecutor


def run_logged_model_only_sequential_episode(
    root: str | Path,
    *,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    model_id: str,
    artifact_id: str | None,
    arm: str,
    task: dict[str, JsonValue],
    teacher: Policy,
    teacher_id: str,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    max_steps: int = 8,
    lineage: dict[str, object] | None = None,
) -> Path:
    """Run one arm model-only and preserve divergence/failure propagation evidence."""
    root = Path(root)
    run = RunLog(
        root,
        "learn_sequential_episode",
        {
            "arm": arm,
            "model_id": model_id,
            "artifact_id": artifact_id,
            "policy_mode": "model_only",
            "teacher_mode": "shadow_only",
            "teacher_id": teacher_id,
            "task": dict(task),
            "max_steps": max_steps,
            "lineage": lineage,
        },
        producer="evaluator",
    )
    policy = ModelPolicy(model, tokenizer, model_id=model_id)
    context = RuntimeContext(task=task)
    recorder = TrajectoryRecorder()
    weights_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    try:
        trajectory = run_loop(
            policy=policy,
            executor=executor,
            evaluator=evaluator,
            context=context,
            recorder=recorder,
            max_steps=max_steps,
        )
        weights_unchanged = all(
            torch.equal(weights_before[name], tensor.detach().cpu())
            for name, tensor in model.state_dict().items()
        )
        if not weights_unchanged:
            raise RuntimeError("Sequential evaluation changed DecisionModel weights")
        if len(policy.traces) != len(trajectory.transitions):
            raise RuntimeError("DecisionTrace count does not match sequential trajectory")

        teacher_rows: list[dict[str, object]] = []
        divergence_steps: list[int] = []
        for index, transition in enumerate(trajectory.transitions):
            state_data = transition.state_before
            state = RuntimeState(
                task=cast(dict[str, JsonValue], state_data["task"]),
                memory=cast(dict[str, JsonValue], state_data["memory"]),
                step=transition.step,
            )
            history = Trajectory(trajectory.transitions[:index])
            teacher_action = teacher.decide(
                state,
                history,
                transition.available_actions,
            )
            agrees = teacher_action == transition.action
            if not agrees:
                divergence_steps.append(transition.step)
            teacher_rows.append(
                {
                    "schema_version": "aster-sequential-teacher-comparison-0",
                    "step": transition.step,
                    "student_action": transition.action.to_dict(),
                    "teacher_action": teacher_action.to_dict(),
                    "agrees": agrees,
                    "execution_accepted": transition.observation.accepted,
                    "execution_ok": transition.observation.ok,
                    "goal_satisfied": transition.evaluation.goal_satisfied,
                }
            )

        first_divergence = None if not divergence_steps else divergence_steps[0]
        first_tool_failure = next(
            (
                transition.step
                for transition in trajectory.transitions
                if transition.action.kind == "tool" and not transition.observation.ok
            ),
            None,
        )
        episode = evaluate_episode(trajectory)
        summary = {
            "schema_version": "aster-learn-sequential-episode-0",
            "arm": arm,
            "model_id": model_id,
            "artifact_id": artifact_id,
            "policy_mode": "model_only",
            "teacher_mode": "shadow_only",
            "teacher_id": teacher_id,
            "task": dict(task),
            "evaluation": episode.to_dict(),
            "teacher_agreements": len(teacher_rows) - len(divergence_steps),
            "teacher_disagreements": len(divergence_steps),
            "first_teacher_divergence_step": first_divergence,
            "steps_after_first_teacher_divergence": (
                0
                if first_divergence is None
                else len(trajectory.transitions) - first_divergence - 1
            ),
            "first_tool_failure_step": first_tool_failure,
            "weights_unchanged": weights_unchanged,
            "final_memory": dict(context.memory),
        }

        recorder.write_jsonl(run.path / "trajectory.jsonl")
        _write_jsonl(
            run.path / "decision-traces.jsonl",
            [trace.to_dict() for trace in policy.traces],
        )
        _write_jsonl(run.path / "teacher-comparison.jsonl", teacher_rows)
        _write_json(run.path / "sequential.json", summary)
        run.event(
            "sequential_episode_evaluated",
            {
                "arm": arm,
                "task_success": episode.task_success,
                "goal_verified": episode.goal_verified,
                "steps": episode.steps,
                "first_teacher_divergence_step": first_divergence,
                "first_tool_failure_step": first_tool_failure,
            },
        )
        run.finish(
            "completed",
            sequential="sequential.json",
            trajectory="trajectory.jsonl",
            decision_traces="decision-traces.jsonl",
            teacher_comparison="teacher-comparison.jsonl",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
