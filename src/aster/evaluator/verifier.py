"""Independent verifier for Agent Kernel v0 tasks."""

from aster.evaluator.result import EvaluationResult
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Observation
from aster.runtime.state import RuntimeState


class TaskEvaluator:
    def evaluate(
        self,
        task: dict[str, JsonValue],
        state_after: RuntimeState,
        action: Action,
        observation: Observation,
        history: Trajectory,
    ) -> EvaluationResult:
        notes: list[str] = []
        action_valid = observation.accepted
        execution_success = observation.ok

        if not action_valid:
            notes.append("action_rejected")
        elif not execution_success:
            notes.append("execution_failed")

        goal_satisfied = _history_has_verified_goal(history)
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
        )


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
