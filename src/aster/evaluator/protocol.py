"""Runtime-facing protocol for step-level evaluation."""

from typing import Protocol

from aster.evaluator.result import EvaluationResult
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Observation
from aster.runtime.state import RuntimeState


class StepEvaluator(Protocol):
    """Evaluate one observed transition without owning runtime execution.

    Implementations may decide task-specific goal satisfaction and notes, but
    runtime facts have fixed meanings: action_valid mirrors
    observation.accepted, execution_success mirrors observation.ok, and
    terminal mirrors whether the action is stop. goal_satisfied is cumulative
    within an episode: once verified, subsequent results must remain verified.
    Transition and Trajectory enforce these invariants at runtime.
    """

    def evaluate(
        self,
        task: dict[str, JsonValue],
        state_after: RuntimeState,
        action: Action,
        observation: Observation,
        history: Trajectory,
    ) -> EvaluationResult:
        ...
