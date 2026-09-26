"""Explicit forced-action branch evaluation over saved agent decision states."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aster.agent.policy import Policy
from aster.evaluator.episode import EpisodeEvaluation, write_episode_evaluation
from aster.evaluator.protocol import StepEvaluator
from aster.records.counterfactual import CounterfactualTrace, write_counterfactual_traces_jsonl
from aster.records.recorder import TrajectoryRecorder
from aster.records.routing import RoutingTrace
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Transition
from aster.reward.contract import RewardSpec, write_reward_result
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import execute_step
from aster.runtime.state import RuntimeState
from aster.tools.executor import ToolExecutor


@dataclass(frozen=True, slots=True)
class _BranchOutcome:
    status: str
    trajectory: Trajectory | None = None
    note: str | None = None


def run_logged_counterfactual_analysis(
    root: str | Path,
    *,
    source_run_id: str,
    trajectory: Trajectory,
    routing: tuple[RoutingTrace, ...],
    continuation_policy: Policy,
    continuation_policy_id: str,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    reward_spec: RewardSpec,
    max_steps: int = 8,
) -> Path:
    """Evaluate saved model candidates in isolated forks under one fixed continuation policy."""
    if not source_run_id:
        raise ValueError("source_run_id must be non-empty")
    if not continuation_policy_id:
        raise ValueError("continuation_policy_id must be non-empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    source_steps = [trace.step for trace in routing]
    if len(source_steps) != len(set(source_steps)):
        raise ValueError("Counterfactual analysis requires at most one routing trace per source step")
    for trace in routing:
        if trace.step >= len(trajectory.transitions):
            raise ValueError("Routing trace step is outside the source trajectory")
        if trace.step >= max_steps:
            raise ValueError("Routing trace step must be below max_steps")
        _require_probability_distribution(trace.calibrated_probabilities)

    root = Path(root)
    run = RunLog(
        root,
        "counterfactual_analysis",
        {
            "source_run_id": source_run_id,
            "source_steps": source_steps,
            "continuation_policy": continuation_policy_id,
            "reward_spec": reward_spec.to_dict(),
            "max_steps": max_steps,
        },
        producer="runtime",
    )

    try:
        traces: list[CounterfactualTrace] = []
        for routing_trace in routing:
            traces.extend(
                _analyze_step(
                    run.path,
                    trajectory=trajectory,
                    routing_trace=routing_trace,
                    continuation_policy=continuation_policy,
                    continuation_policy_id=continuation_policy_id,
                    executor=executor,
                    evaluator=evaluator,
                    reward_spec=reward_spec,
                    max_steps=max_steps,
                )
            )

        write_counterfactual_traces_jsonl(run.path / "counterfactuals.jsonl", traces)
        completed = sum(trace.status == "completed" for trace in traces)
        summary = {
            "source_steps": len(routing),
            "candidate_branches": len(traces),
            "completed_branches": completed,
            "unsupported_branches": len(traces) - completed,
            "steps_with_advantage": len(
                {
                    trace.source_step
                    for trace in traces
                    if trace.advantage_estimate is not None
                }
            ),
        }
        _write_json(
            run.path / "counterfactual.json",
            {
                "schema_version": "aster-counterfactual-analysis-0",
                "source_run_id": source_run_id,
                "continuation_policy": continuation_policy_id,
                "reward_spec": reward_spec.to_dict(),
                "traces_file": "counterfactuals.jsonl",
                "branches_dir": "branches",
                "summary": summary,
            },
        )
        run.event("counterfactuals_evaluated", summary)
        run.finish(
            "completed",
            counterfactual="counterfactual.json",
            traces="counterfactuals.jsonl",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _analyze_step(
    run_path: Path,
    *,
    trajectory: Trajectory,
    routing_trace: RoutingTrace,
    continuation_policy: Policy,
    continuation_policy_id: str,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    reward_spec: RewardSpec,
    max_steps: int,
) -> tuple[CounterfactualTrace, ...]:
    outcomes: list[_BranchOutcome] = []
    persisted: list[tuple[str, float, EpisodeEvaluation] | None] = []

    for candidate_index, candidate in enumerate(routing_trace.candidates):
        outcome = _run_branch(
            trajectory=trajectory,
            source_step=routing_trace.step,
            forced_action=candidate,
            continuation_policy=continuation_policy,
            executor=executor,
            evaluator=evaluator,
            max_steps=max_steps,
        )
        outcomes.append(outcome)
        if outcome.trajectory is None:
            persisted.append(None)
            continue

        relative_dir = f"branches/step-{routing_trace.step:03d}-candidate-{candidate_index:03d}"
        branch_dir = run_path / relative_dir
        recorder = TrajectoryRecorder()
        for transition in outcome.trajectory.transitions:
            recorder.record(transition)
        trajectory_path = branch_dir / "trajectory.jsonl"
        recorder.write_jsonl(trajectory_path)
        evaluation = write_episode_evaluation(branch_dir / "evaluation.json", trajectory_path)
        reward = write_reward_result(
            branch_dir / "reward.json",
            evaluation,
            reward_spec,
        )
        persisted.append((relative_dir, reward.total, evaluation))

    all_completed = all(item is not None for item in persisted)
    baseline = None
    if all_completed:
        baseline = sum(
            probability * cast(tuple[str, float, EpisodeEvaluation], item)[1]
            for probability, item in zip(
                routing_trace.calibrated_probabilities,
                persisted,
                strict=True,
            )
        )

    traces: list[CounterfactualTrace] = []
    for candidate_index, (candidate, probability, outcome, saved) in enumerate(
        zip(
            routing_trace.candidates,
            routing_trace.calibrated_probabilities,
            outcomes,
            persisted,
            strict=True,
        )
    ):
        if saved is None:
            traces.append(
                CounterfactualTrace(
                    source_step=routing_trace.step,
                    candidate_index=candidate_index,
                    model_id=routing_trace.model_id,
                    source_route=routing_trace.route,
                    action=candidate,
                    model_probability=probability,
                    model_selected=candidate_index == routing_trace.selected_index,
                    source_executed=candidate == routing_trace.final_action,
                    continuation_policy=continuation_policy_id,
                    status=outcome.status,
                    note=outcome.note,
                )
            )
            continue

        relative_dir, reward_total, evaluation = saved
        advantage = None if baseline is None else reward_total - baseline
        traces.append(
            CounterfactualTrace(
                source_step=routing_trace.step,
                candidate_index=candidate_index,
                model_id=routing_trace.model_id,
                source_route=routing_trace.route,
                action=candidate,
                model_probability=probability,
                model_selected=candidate_index == routing_trace.selected_index,
                source_executed=candidate == routing_trace.final_action,
                continuation_policy=continuation_policy_id,
                status="completed",
                branch_dir=relative_dir,
                branch_reward_total=reward_total,
                branch_steps=evaluation.steps,
                task_success=evaluation.task_success,
                terminal_reached=evaluation.terminal_reached,
                baseline_reward=baseline,
                advantage_estimate=advantage,
            )
        )
    return tuple(traces)


def _run_branch(
    *,
    trajectory: Trajectory,
    source_step: int,
    forced_action: Action,
    continuation_policy: Policy,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    max_steps: int,
) -> _BranchOutcome:
    source_transition = trajectory.transitions[source_step]
    if source_transition.step != source_step:
        raise ValueError("Source trajectory steps must be contiguous and index-aligned")

    state_before = _runtime_state(source_transition)
    context = RuntimeContext(task=state_before.task, memory=state_before.memory)
    available_actions = executor.registry.names() + ("stop",)
    prefix = list(trajectory.transitions[:source_step])
    history = Trajectory(tuple(prefix))

    unsafe = _unsafe_reason(forced_action, executor)
    if unsafe is not None:
        return _BranchOutcome(
            status="unsupported_counterfactual_tool",
            note=unsafe,
        )

    forced_transition = execute_step(
        step=source_step,
        action=forced_action,
        history=history,
        available_actions=available_actions,
        executor=executor,
        evaluator=evaluator,
        context=context,
        state_before=state_before,
    )
    branch = prefix + [forced_transition]
    if forced_action.kind == "stop":
        return _BranchOutcome(status="completed", trajectory=Trajectory(tuple(branch)))

    for step in range(source_step + 1, max_steps):
        history = Trajectory(tuple(branch))
        state = context.snapshot(step)
        action = continuation_policy.decide(state, history, available_actions)
        unsafe = _unsafe_reason(action, executor)
        if unsafe is not None:
            return _BranchOutcome(
                status="unsupported_counterfactual_tool",
                note=f"continuation policy selected {unsafe}",
            )
        transition = execute_step(
            step=step,
            action=action,
            history=history,
            available_actions=available_actions,
            executor=executor,
            evaluator=evaluator,
            context=context,
            state_before=state,
        )
        branch.append(transition)
        if action.kind == "stop":
            break

    return _BranchOutcome(status="completed", trajectory=Trajectory(tuple(branch)))


def _unsafe_reason(action: Action, executor: ToolExecutor) -> str | None:
    if action.kind == "stop":
        return None
    if action.name is None:
        return "tool action has no name"
    registered = executor.registry.get(action.name)
    if registered is None:
        return f"tool is not registered: {action.name}"
    if not registered.spec.counterfactual_safe:
        return f"tool is not marked counterfactual_safe: {action.name}"
    return None


def _runtime_state(transition: Transition) -> RuntimeState:
    data = transition.state_before
    task = data.get("task")
    memory = data.get("memory")
    step = data.get("step")
    if not isinstance(task, dict) or not isinstance(memory, dict):
        raise ValueError("Persisted RuntimeState task and memory must be objects")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError("Persisted RuntimeState step must be an integer")
    return RuntimeState(
        task=cast(dict[str, JsonValue], task),
        memory=cast(dict[str, JsonValue], memory),
        step=step,
    )


def _require_probability_distribution(probabilities: tuple[float, ...]) -> None:
    if not probabilities:
        raise ValueError("Counterfactual analysis requires candidate probabilities")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Calibrated candidate probabilities must sum to one")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
