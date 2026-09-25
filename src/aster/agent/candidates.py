"""Build structured Action candidates without executing them."""

from typing import Protocol

from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue
from aster.runtime.state import RuntimeState


class CandidateBuilder(Protocol):
    def build(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> tuple[Action, ...]: ...


class CalculateAndStoreCandidates:
    """Candidate generator for the Agent Kernel v0 calculate-and-store task."""

    def build(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> tuple[Action, ...]:
        task = state.task
        if task.get("kind") != "calculate_and_store":
            return (Action.stop("unsupported_task"),)

        if trajectory.last is not None and not trajectory.last.observation.ok:
            return (Action.stop("tool_failure"),)

        required = ("operation", "left", "right", "store_as")
        if any(name not in task for name in required):
            return (Action.stop("invalid_task"),)
        key = task["store_as"]
        if not isinstance(key, str):
            return (Action.stop("invalid_task"),)

        candidates: list[Action] = []
        if "calculator" in available_actions:
            candidates.append(
                Action.tool(
                    "calculator",
                    operation=task["operation"],
                    left=task["left"],
                    right=task["right"],
                )
            )

        calculation = _successful_output(trajectory, "calculator")
        if not isinstance(calculation, _Missing) and "memory.put" in available_actions:
            candidates.append(Action.tool("memory.put", key=key, value=calculation))

        if "memory.get" in available_actions:
            candidates.append(Action.tool("memory.get", key=key))

        if "stop" in available_actions:
            candidates.append(Action.stop("policy_stop"))
            if not isinstance(calculation, _Missing) and _successful_get(trajectory, key, calculation):
                candidates.append(Action.stop("goal_verified"))

        if not candidates:
            return (Action.stop("no_available_action"),)
        return tuple(candidates)


class _Missing:
    pass


_MISSING = _Missing()


def _successful_output(trajectory: Trajectory, tool_name: str) -> JsonValue | _Missing:
    for transition in reversed(trajectory.transitions):
        if transition.action.name == tool_name and transition.observation.ok:
            return transition.observation.output
    return _MISSING


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
