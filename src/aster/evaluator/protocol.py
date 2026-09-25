"""Runtime-facing protocol for step-level evaluation."""

from typing import Protocol

from aster.evaluator.result import EvaluationResult
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Observation
from aster.runtime.state import RuntimeState


class StepEvaluator(Protocol):
    """Evaluate one observed transition without owning runtime execution."""

    def evaluate(
        self,
        task: dict[str, JsonValue],
        state_after: RuntimeState,
        action: Action,
        observation: Observation,
        history: Trajectory,
    ) -> EvaluationResult:
        ...
