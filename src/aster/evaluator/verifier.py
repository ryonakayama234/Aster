"""Independent verifier for Agent Kernel v0 tasks."""

from aster.evaluator.result import EvaluationResult
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Observation
from aster.runtime.state import RuntimeState
from aster.evaluator.goal_contract import EVER_GOAL_V0, CURRENT_GOAL_V1, validate_goal_contract


class TaskEvaluator:
    def __init__(self, *, goal_contract: str = EVER_GOAL_V0) -> None:
        validate_goal_contract(goal_contract)
        self.goal_contract = goal_contract

    def evaluate(
        self,
        task: dict[str, JsonValue],
        state_after: RuntimeState,
        action: Action,
        observation: Observation,
        history: Trajectory,
    ) -> EvaluationResult:
        if history.transitions and history.goal_contract != self.goal_contract:
            raise ValueError("Evaluator and history goal contracts disagree")
        notes: list[str] = []
        action_valid = observation.accepted
        execution_success = observation.ok

        if not action_valid:
            notes.append("action_rejected")
        elif not execution_success:
            notes.append("execution_failed")

        goal_satisfied = (_history_has_verified_goal(history)
                          if self.goal_contract == EVER_GOAL_V0 else False)
        if task.get("kind") == "calculate_and_store":
            expected = _expected_value(task)
            key = task.get("store_as")
            verified_now = (
                action.name == "memory.get"
                and observation.ok
                and isinstance(key, str)
                and isinstance(observation.output, dict)
                and observation.output.get("key") == key
                and observation.output.get("value") == expected
                and state_after.memory.get(key) == expected
            )
            if self.goal_contract == CURRENT_GOAL_V1:
                goal_satisfied = _current_goal_verified(
                    state_after, key, expected, action, observation, history
                )
            else:
                goal_satisfied = goal_satisfied or verified_now
        else:
            notes.append("unsupported_task")

        if goal_satisfied:
            notes.append("goal_verified")

        return EvaluationResult(
            action_valid=action_valid,
            execution_success=execution_success,
            goal_satisfied=goal_satisfied,
            terminal=action.kind == "stop",
            notes=tuple(notes),
            goal_contract=self.goal_contract,
        )


def _current_goal_verified(state, key, expected, action, observation, history) -> bool:
    if (not isinstance(key, str) or expected is None or key not in state.memory
            or isinstance(state.memory[key], bool) or state.memory[key] != expected):
        return False
    # Inspect raw execution evidence, not policy decisions or past goal flags.
    events = [(t.action, t.observation) for t in history.transitions]
    events.append((action, observation))
    for event_action, event_observation in reversed(events):
        if not event_observation.ok or event_action.arguments.get("key") != key:
            continue
        if event_action.name == "memory.put":
            return False
        if event_action.name == "memory.get":
            output = event_observation.output
            return (isinstance(output, dict) and output.get("key") == key
                    and not isinstance(output.get("value"), bool)
                    and output.get("value") == expected)
    return False


def _history_has_verified_goal(history: Trajectory) -> bool:
    return any(transition.evaluation.goal_satisfied for transition in history.transitions)


def _expected_value(task: dict[str, JsonValue]) -> JsonValue:
    operation = task.get("operation")
    left = task.get("left")
    right = task.get("right")
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return None
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return None
    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if operation == "divide" and right != 0:
        return left / right
    return None
