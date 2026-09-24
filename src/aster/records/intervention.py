"""Shadow-teacher supervision kept separate from semantic execution truth."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Transition
from aster.runtime.state import RuntimeState

_ROUTE_NAMES = {"model", "fallback", "abstain"}


@dataclass(frozen=True, slots=True)
class InterventionTrace:
    """One student-visited state paired with a shadow-teacher Action."""

    step: int
    student_model_id: str
    state: RuntimeState
    trajectory: Trajectory
    candidates: tuple[Action, ...]
    scores: tuple[float, ...]
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...]
    selected_index: int
    route: str
    executed_action: Action
    teacher_id: str
    teacher_action: Action
    teacher_target_index: int | None

    def __post_init__(self):
        size = len(self.candidates)
        if self.step < 0 or self.state.step != self.step:
            raise ValueError("Intervention step must match RuntimeState step")
        if not self.student_model_id or not self.teacher_id:
            raise ValueError("student_model_id and teacher_id are required")
        if self.route not in _ROUTE_NAMES:
            raise ValueError(f"Unknown routing route: {self.route}")
        if not size:
            raise ValueError("Intervention trace requires at least one candidate")
        if len(self.scores) != size:
            raise ValueError("Intervention scores must match candidate count")
        if len(self.raw_probabilities) != size or len(self.calibrated_probabilities) != size:
            raise ValueError("Intervention probabilities must match candidate count")
        if not 0 <= self.selected_index < size:
            raise ValueError("selected_index must reference one candidate")
        for probabilities in (self.raw_probabilities, self.calibrated_probabilities):
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in probabilities
            ):
                raise ValueError("Intervention probabilities must be finite values in [0, 1]")
        if self.teacher_target_index is not None:
            if not 0 <= self.teacher_target_index < size:
                raise ValueError("teacher_target_index must reference one candidate")
            if self.candidates[self.teacher_target_index] != self.teacher_action:
                raise ValueError("teacher_target_index must reference teacher_action")

    @property
    def student_action(self) -> Action:
        return self.candidates[self.selected_index]

    @property
    def student_confidence(self) -> float:
        return max(self.calibrated_probabilities)

    @property
    def agrees(self) -> bool:
        return self.student_action == self.teacher_action

    @property
    def trainable(self) -> bool:
        return self.teacher_target_index is not None

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-intervention-trace-0",
            "step": self.step,
            "student_model_id": self.student_model_id,
            "state": self.state.to_dict(),
            "trajectory": [
                transition.to_dict() for transition in self.trajectory.transitions
            ],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scores": list(self.scores),
            "raw_probabilities": list(self.raw_probabilities),
            "calibrated_probabilities": list(self.calibrated_probabilities),
            "selected_index": self.selected_index,
            "student_action": self.student_action.to_dict(),
            "student_confidence": self.student_confidence,
            "route": self.route,
            "executed_action": self.executed_action.to_dict(),
            "teacher_id": self.teacher_id,
            "teacher_action": self.teacher_action.to_dict(),
            "teacher_target_index": self.teacher_target_index,
            "agrees": self.agrees,
            "trainable": self.trainable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterventionTrace":
        if data.get("schema_version") != "aster-intervention-trace-0":
            raise ValueError("Unsupported intervention trace schema")
        state_data = data["state"]
        return cls(
            step=int(data["step"]),
            student_model_id=str(data["student_model_id"]),
            state=RuntimeState(
                task=state_data["task"],
                memory=state_data["memory"],
                step=int(state_data["step"]),
            ),
            trajectory=Trajectory(
                tuple(Transition.from_dict(item) for item in data.get("trajectory", []))
            ),
            candidates=tuple(Action.from_dict(item) for item in data["candidates"]),
            scores=tuple(float(value) for value in data["scores"]),
            raw_probabilities=tuple(float(value) for value in data["raw_probabilities"]),
            calibrated_probabilities=tuple(
                float(value) for value in data["calibrated_probabilities"]
            ),
            selected_index=int(data["selected_index"]),
            route=str(data["route"]),
            executed_action=Action.from_dict(data["executed_action"]),
            teacher_id=str(data["teacher_id"]),
            teacher_action=Action.from_dict(data["teacher_action"]),
            teacher_target_index=(
                None
                if data.get("teacher_target_index") is None
                else int(data["teacher_target_index"])
            ),
        )


def write_intervention_traces_jsonl(
    path: str | Path, traces: Sequence[InterventionTrace]
) -> None:
    """Write deterministic, Sites-readable shadow-teacher evidence."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for trace in traces:
            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_intervention_traces_jsonl(path: str | Path) -> tuple[InterventionTrace, ...]:
    """Reload an intervention artifact for a later training/evaluation run."""
    input_path = Path(path)
    traces: list[InterventionTrace] = []
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                traces.append(InterventionTrace.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid intervention trace at {input_path}:{line_number}"
                ) from error
    return tuple(traces)
