"""Observable confidence-gated Agent rollout for Sites and later training loops."""

import json
from pathlib import Path

from aster.agent.selective import SelectivePolicy
from aster.evaluator.verifier import TaskEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.routing import RoutingTrace, write_routing_traces_jsonl
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.runtime.context import RuntimeContext
from aster.tools.executor import ToolExecutor


def run_logged_selective_agent(
    root: str | Path,
    *,
    policy: SelectivePolicy,
    executor: ToolExecutor,
    evaluator: TaskEvaluator,
    context: RuntimeContext,
    max_steps: int = 8,
) -> Path:
    """Run one selective Agent trajectory and emit replayable runtime artifacts."""
    root = Path(root)
    run = RunLog(
        root,
        "selective_agent",
        {
            "model_id": policy.model_id,
            "routing": policy.config.to_dict(),
            "fallback_policy": policy.fallback_id,
            "max_steps": max_steps,
        },
        producer="runtime",
    )
    recorder = TrajectoryRecorder()
    first_routing_trace = len(policy.routing_traces)

    try:
        from aster.runtime.loop import run_loop

        trajectory = run_loop(
            policy=policy,
            executor=executor,
            evaluator=evaluator,
            context=context,
            recorder=recorder,
            max_steps=max_steps,
        )
        traces = tuple(policy.routing_traces[first_routing_trace:])
        if len(traces) != len(trajectory.transitions):
            raise RuntimeError("Selective routing trace count must match trajectory length")

        recorder.write_jsonl(run.path / "trajectory.jsonl")
        write_routing_traces_jsonl(run.path / "routing.jsonl", traces)
        summary = summarize_selective_rollout(trajectory, traces)
        _write_json(
            run.path / "selective.json",
            {
                "schema_version": "aster-selective-rollout-0",
                "model_id": policy.model_id,
                "routing": policy.config.to_dict(),
                "fallback_policy": policy.fallback_id,
                "trajectory_file": "trajectory.jsonl",
                "routing_file": "routing.jsonl",
                "summary": summary,
            },
        )
        run.event("rolled_out", summary)
        run.finish(
            "completed",
            selective="selective.json",
            trajectory="trajectory.jsonl",
            routing="routing.jsonl",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def summarize_selective_rollout(
    trajectory: Trajectory,
    traces: tuple[RoutingTrace, ...],
) -> dict:
    """Summarize routing behavior without pretending it is a capability benchmark."""
    if len(traces) != len(trajectory.transitions):
        raise ValueError("Routing traces and transitions must have the same length")

    count = len(traces)
    route_counts = {route: sum(trace.route == route for trace in traces) for route in (
        "model",
        "fallback",
        "abstain",
    )}
    model_steps = [
        transition
        for transition, trace in zip(trajectory.transitions, traces, strict=True)
        if trace.route == "model"
    ]
    return {
        "steps": count,
        "task_success": trajectory.successful,
        "route_counts": route_counts,
        "autonomous_coverage": 0.0 if count == 0 else route_counts["model"] / count,
        "fallback_rate": 0.0 if count == 0 else route_counts["fallback"] / count,
        "abstain_rate": 0.0 if count == 0 else route_counts["abstain"] / count,
        "mean_calibrated_confidence": (
            None if count == 0 else sum(trace.confidence for trace in traces) / count
        ),
        "autonomous_execution_failures": sum(
            not transition.observation.ok for transition in model_steps
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
