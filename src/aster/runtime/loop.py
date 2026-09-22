"""Small policy -> execution -> record -> evaluation loop for Agent Kernel v0."""

from aster.agent.policy import Policy
from aster.evaluator.verifier import TaskEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.tools.executor import ToolExecutor


def run_loop(
    *,
    policy: Policy,
    executor: ToolExecutor,
    evaluator: TaskEvaluator,
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
        recorder.record(
            Transition(
                step=step,
                state_before=state_before.to_dict(),
                available_actions=available_actions,
                action=action,
                observation=observation,
                state_after=state_after.to_dict(),
                evaluation=evaluation,
            )
        )
        if action.kind == "stop":
            break

    return recorder.trajectory()
