"""Policy boundary: choose an Action without executing it."""

from typing import Protocol

from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue
from aster.runtime.state import RuntimeState
from aster.evaluator.goal_contract import EVER_GOAL_V0, CURRENT_GOAL_V1, validate_goal_contract


class Policy(Protocol):
    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action: ...


class RuleBasedPolicy:
    """Deterministic v0 policy for a structured calculate-and-store task."""

    def __init__(self, *, goal_contract: str = EVER_GOAL_V0) -> None:
        validate_goal_contract(goal_contract)
        self.goal_contract = goal_contract

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

        calculation = (current_calculation(state, trajectory)
                       if self.goal_contract == CURRENT_GOAL_V1
                       else _successful_output(trajectory, "calculator"))
        if isinstance(calculation, MissingCalculation):
            return Action.tool(
                "calculator",
                operation=task["operation"],
                left=task["left"],
                right=task["right"],
            )

        key = task["store_as"]
        if not isinstance(key, str):
            return Action.stop("invalid_task")

        if self.goal_contract == CURRENT_GOAL_V1:
            if (key not in state.memory or isinstance(state.memory[key], bool)
                    or state.memory[key] != calculation):
                return Action.tool("memory.put", key=key, value=calculation)
            if not has_current_confirmation(trajectory, key, calculation):
                return Action.tool("memory.get", key=key)
            return Action.stop("goal_verified")

        if not _successful_put(trajectory, key, calculation):
            return Action.tool("memory.put", key=key, value=calculation)

        if not _successful_get(trajectory, key, calculation):
            return Action.tool("memory.get", key=key)

        return Action.stop("goal_verified")


class MissingCalculation:
    pass


_MISSING = MissingCalculation()


def current_calculation(state: RuntimeState, trajectory: Trajectory) -> JsonValue | MissingCalculation:
    expected_arguments = {name: state.task.get(name) for name in ("operation", "left", "right")}
    for transition in reversed(trajectory.transitions):
        if (transition.action.name == "calculator" and transition.observation.ok
                and transition.action.arguments == expected_arguments):
            return transition.observation.output
    return _MISSING


def has_current_confirmation(trajectory: Trajectory, key: str, value: JsonValue) -> bool:
    for transition in reversed(trajectory.transitions):
        if not transition.observation.ok or transition.action.arguments.get("key") != key:
            continue
        if transition.action.name == "memory.put":
            return False
        if transition.action.name == "memory.get":
            output = transition.observation.output
            return (isinstance(output, dict) and output.get("key") == key
                    and not isinstance(output.get("value"), bool)
                    and output.get("value") == value)
    return False


def _successful_output(trajectory: Trajectory, tool_name: str) -> JsonValue | MissingCalculation:
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
