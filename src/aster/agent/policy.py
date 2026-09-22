"""Policy boundary: choose an Action without executing it."""

from typing import Protocol

from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue
from aster.runtime.state import RuntimeState


class Policy(Protocol):
    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action: ...


class RuleBasedPolicy:
    """Deterministic v0 policy for a structured calculate-and-store task."""

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        task = state.task
        if task.get("kind") != "calculate_and_store":
            return Action.stop("unsupported_task")

        if trajectory.last is not None and not trajectory.last.observation.ok:
            return Action.stop("tool_failure")

        calculation = _successful_output(trajectory, "calculator")
        if calculation is _MISSING:
            return Action.tool(
                "calculator",
                operation=task["operation"],
                left=task["left"],
                right=task["right"],
            )

        key = task["store_as"]
        if not isinstance(key, str):
            return Action.stop("invalid_task")

        if not _successful_put(trajectory, key, calculation):
            return Action.tool("memory.put", key=key, value=calculation)

        if not _successful_get(trajectory, key, calculation):
            return Action.tool("memory.get", key=key)

        return Action.stop("goal_verified")


class _Missing:
    pass


_MISSING = _Missing()


def _successful_output(trajectory: Trajectory, tool_name: str) -> JsonValue | _Missing:
    for transition in reversed(trajectory.transitions):
        if transition.action.name == tool_name and transition.observation.ok:
            return transition.observation.output
    return _MISSING


def _successful_put(trajectory: Trajectory, key: str, value: JsonValue) -> bool:
    for transition in trajectory.transitions:
        if (
            transition.action.name == "memory.put"
            and transition.observation.ok
            and transition.action.arguments.get("key") == key
            and transition.action.arguments.get("value") == value
        ):
            return True
    return False


def _successful_get(trajectory: Trajectory, key: str, value: JsonValue) -> bool:
    for transition in trajectory.transitions:
        output = transition.observation.output
        if (
            transition.action.name == "memory.get"
            and transition.observation.ok
            and isinstance(output, dict)
            and output.get("key") == key
            and output.get("value") == value
        ):
            return True
    return False
