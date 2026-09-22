"""Records for supervised candidate decisions and model-side decision traces."""

from dataclasses import dataclass
from json import dumps

from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Transition
from aster.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class DecisionExample:
    """One supervised state/history/candidates -> chosen candidate example."""

    state: RuntimeState
    trajectory: Trajectory
    candidates: tuple[Action, ...]
    target_index: int
    teacher: str = "rule-v0"

    def __post_init__(self):
        if not self.candidates:
            raise ValueError("Decision examples require at least one candidate")
        if not 0 <= self.target_index < len(self.candidates):
            raise ValueError("target_index must reference one candidate")

    @property
    def target(self) -> Action:
        return self.candidates[self.target_index]

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-decision-example-0",
            "state": self.state.to_dict(),
            "trajectory": [transition.to_dict() for transition in self.trajectory.transitions],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "target_index": self.target_index,
            "teacher": self.teacher,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionExample":
        if data.get("schema_version") != "aster-decision-example-0":
            raise ValueError("Unsupported decision example schema")
        state_data = data["state"]
        return cls(
            state=RuntimeState(
                task=state_data["task"],
                memory=state_data["memory"],
                step=int(state_data["step"]),
            ),
            trajectory=Trajectory(
                tuple(Transition.from_dict(item) for item in data.get("trajectory", []))
            ),
            candidates=tuple(Action.from_dict(item) for item in data["candidates"]),
            target_index=int(data["target_index"]),
            teacher=str(data.get("teacher", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Diagnostic evidence from one ModelPolicy decision."""

    step: int
    model_id: str
    candidates: tuple[Action, ...]
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    selected_index: int

    def __post_init__(self):
        size = len(self.candidates)
        if self.step < 0:
            raise ValueError("Decision trace step must be non-negative")
        if not size or len(self.scores) != size or len(self.probabilities) != size:
            raise ValueError("Decision trace fields must have matching non-zero lengths")
        if not 0 <= self.selected_index < size:
            raise ValueError("selected_index must reference one candidate")

    @property
    def selected(self) -> Action:
        return self.candidates[self.selected_index]

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-decision-trace-0",
            "step": self.step,
            "model_id": self.model_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scores": list(self.scores),
            "probabilities": list(self.probabilities),
            "selected_index": self.selected_index,
        }


def serialize_decision_input(
    state: RuntimeState,
    trajectory: Trajectory,
    candidate: Action,
) -> str:
    """Deterministically serialize exactly the information a decision scorer receives."""
    payload = {
        "schema": "aster-decision-input-0",
        "state": state.to_dict(),
        "history": [
            {
                "action": transition.action.to_dict(),
                "observation": transition.observation.to_dict(),
            }
            for transition in trajectory.transitions
        ],
        "candidate": candidate.to_dict(),
    }
    return dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
