"""Observable student rollout with shadow-teacher intervention collection."""

import json
from pathlib import Path

from aster.agent.intervention import InterventionPolicy
from aster.agent.policy import Policy
from aster.agent.selective import SelectivePolicy
from aster.evaluator.episode import write_episode_evaluation
from aster.evaluator.protocol import StepEvaluator
from aster.records.intervention import InterventionTrace, write_intervention_traces_jsonl
from aster.records.recorder import TrajectoryRecorder
from aster.records.routing import RoutingTrace, write_routing_traces_jsonl
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.reward.contract import RewardSpec, write_reward_result
from aster.runtime.context import RuntimeContext
from aster.tools.executor import ToolExecutor


def run_logged_intervention_agent(
    root: str | Path,
    *,
    policy: SelectivePolicy,
    teacher: Policy,
    teacher_id: str,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    context: RuntimeContext,
    reward_spec: RewardSpec | None = None,
    max_steps: int = 8,
) -> Path:
    """Run one student trajectory while labeling every visited state in shadow mode."""
    root = Path(root)
    run = RunLog(
        root,
        "intervention_agent",
        {
            "student_model_id": policy.model_id,
            "teacher_id": teacher_id,
            "routing": policy.config.to_dict(),
            "fallback_policy": policy.fallback_id,
            "reward_spec": None if reward_spec is None else reward_spec.to_dict(),
            "max_steps": max_steps,
        },
        producer="runtime",
    )
    recorder = TrajectoryRecorder()
    collector = InterventionPolicy(policy, teacher, teacher_id=teacher_id)
    first_routing_trace = len(policy.routing_traces)

    try:
        from aster.runtime.loop import run_loop

        trajectory = run_loop(
            policy=collector,
            executor=executor,
            evaluator=evaluator,
            context=context,
            recorder=recorder,
            max_steps=max_steps,
        )
        routing = tuple(policy.routing_traces[first_routing_trace:])
        interventions = tuple(collector.interventions)
        if len(routing) != len(trajectory.transitions):
            raise RuntimeError("Routing trace count must match trajectory length")
        if len(interventions) != len(trajectory.transitions):
            raise RuntimeError("Intervention trace count must match trajectory length")

        recorder.write_jsonl(run.path / "trajectory.jsonl")
        write_routing_traces_jsonl(run.path / "routing.jsonl", routing)
        write_intervention_traces_jsonl(run.path / "interventions.jsonl", interventions)
        episode_evaluation = write_episode_evaluation(
            run.path / "evaluation.json",
            trajectory,
        )
        reward_result = None
        if reward_spec is not None:
            reward_result = write_reward_result(
                run.path / "reward.json",
                episode_evaluation,
                reward_spec,
            )
        summary = summarize_intervention_rollout(trajectory, routing, interventions)
        _write_json(
            run.path / "learning.json",
            {
                "schema_version": "aster-intervention-rollout-0",
                "student_model_id": policy.model_id,
                "teacher_id": teacher_id,
                "trajectory_file": "trajectory.jsonl",
                "routing_file": "routing.jsonl",
                "interventions_file": "interventions.jsonl",
                "evaluation_file": "evaluation.json",
                "reward_file": None if reward_result is None else "reward.json",
                "summary": summary,
            },
        )
        run.event("episode_evaluated", episode_evaluation.to_dict())
        if reward_result is not None:
            run.event("reward_computed", reward_result.to_dict())
        run.event("interventions_collected", summary)
        artifacts = {
            "learning": "learning.json",
            "trajectory": "trajectory.jsonl",
            "routing": "routing.jsonl",
            "interventions": "interventions.jsonl",
            "evaluation": "evaluation.json",
        }
        if reward_result is not None:
            artifacts["reward"] = "reward.json"
        run.finish("completed", **artifacts)
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def summarize_intervention_rollout(
    trajectory: Trajectory,
    routing: tuple[RoutingTrace, ...],
    interventions: tuple[InterventionTrace, ...],
) -> dict:
    if len(trajectory.transitions) != len(routing) or len(routing) != len(interventions):
        raise ValueError("Trajectory, routing, and intervention counts must match")

    steps = len(interventions)
    disagreements = sum(not trace.agrees for trace in interventions)
    trainable = sum(trace.trainable for trace in interventions)
    high_confidence_disagreements = sum(
        trace.route == "model" and not trace.agrees for trace in interventions
    )
    route_counts = {
        route: sum(trace.route == route for trace in interventions)
        for route in ("model", "fallback", "abstain")
    }
    return {
        "steps": steps,
        "task_success": trajectory.successful,
        "route_counts": route_counts,
        "teacher_agreements": steps - disagreements,
        "teacher_disagreements": disagreements,
        "teacher_agreement_rate": None if steps == 0 else (steps - disagreements) / steps,
        "trainable_examples": trainable,
        "teacher_action_missing_from_candidates": steps - trainable,
        "high_confidence_disagreements": high_confidence_disagreements,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
