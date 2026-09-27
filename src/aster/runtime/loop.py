"""Small policy -> execution -> record -> evaluation loop for Agent Kernel v0."""

from aster.agent.policy import Policy
from aster.evaluator.protocol import StepEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.runtime.state import RuntimeState
from aster.tools.executor import ToolExecutor


def execute_step(
    *,
    step: int,
    action: Action,
    history: Trajectory,
    available_actions: tuple[str, ...],
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    context: RuntimeContext,
    state_before: RuntimeState | None = None,
) -> Transition:
    """Execute one already-selected Action without choosing a policy action."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if len(history.transitions) != step:
        raise ValueError("history length must match step")

    before = context.snapshot(step) if state_before is None else state_before
    if before.step != step:
        raise ValueError("state_before step must match step")

    if action.kind == "stop":
        observation = Observation(
            accepted=True,
            ok=True,
            output={"stopped": True, "reason": action.arguments.get("reason", "policy_stop")},
        )
    else:
        observation = executor.execute(action, context)

    state_after = context.snapshot(step + 1)
    evaluation = evaluator.evaluate(
        context.task,
        state_after,
        action,
        observation,
        history,
    )
    return Transition(
        step=step,
        state_before=before.to_dict(),
        available_actions=available_actions,
        action=action,
        observation=observation,
        state_after=state_after.to_dict(),
        evaluation=evaluation,
    )


def run_loop(
    *,
    policy: Policy,
    executor: ToolExecutor,
    evaluator: StepEvaluator,
    context: RuntimeContext,
    recorder: TrajectoryRecorder | None = None,
    max_steps: int = 8,
) -> Trajectory:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    recorder = recorder or TrajectoryRecorder()
    available_actions = executor.registry.names() + ("stop",)

    for step in range(max_steps):
        history = recorder.trajectory()
        state_before = context.snapshot(step)
        action = policy.decide(state_before, history, available_actions)
        recorder.record(
            execute_step(
                step=step,
                action=action,
                history=history,
                available_actions=available_actions,
                executor=executor,
                evaluator=evaluator,
                context=context,
                state_before=state_before,
            )
        )
        if action.kind == "stop":
            break

    return recorder.trajectory()
