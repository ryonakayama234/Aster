"""Model-side routing evidence kept separate from semantic Transition truth."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from aster.records.transition import Action

_ROUTE_NAMES = {"model", "fallback", "abstain"}


@dataclass(frozen=True, slots=True)
class RoutingTrace:
    """One confidence-gated routing decision made around a model Action."""

    step: int
    model_id: str
    route: str
    temperature: float
    autonomous_threshold: float
    fallback_threshold: float
    candidates: tuple[Action, ...]
    scores: tuple[float, ...]
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...]
    selected_index: int
    final_action: Action
    fallback_policy: str | None = None

    def __post_init__(self):
        size = len(self.candidates)
        if self.step < 0:
            raise ValueError("Routing trace step must be non-negative")
        if self.route not in _ROUTE_NAMES:
            raise ValueError(f"Unknown routing route: {self.route}")
        if not size:
            raise ValueError("Routing trace requires at least one candidate")
        if len(self.scores) != size:
            raise ValueError("Routing trace scores must match candidate count")
        if len(self.raw_probabilities) != size or len(self.calibrated_probabilities) != size:
            raise ValueError("Routing trace probabilities must match candidate count")
        if not 0 <= self.selected_index < size:
            raise ValueError("selected_index must reference one candidate")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        for threshold in (self.fallback_threshold, self.autonomous_threshold):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("routing thresholds must be finite values in [0, 1]")
        if self.fallback_threshold > self.autonomous_threshold:
            raise ValueError("fallback_threshold cannot exceed autonomous_threshold")
        for probabilities in (self.raw_probabilities, self.calibrated_probabilities):
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError("Routing probabilities must be finite values in [0, 1]")
        if self.route == "model" and self.final_action != self.model_action:
            raise ValueError("model route must execute the model-selected Action")
        if self.route == "abstain" and self.final_action.kind != "stop":
            raise ValueError("abstain route must return a stop Action")

    @property
    def model_action(self) -> Action:
        return self.candidates[self.selected_index]

    @property
    def confidence(self) -> float:
        return max(self.calibrated_probabilities)

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-routing-trace-0",
            "step": self.step,
            "model_id": self.model_id,
            "route": self.route,
            "temperature": self.temperature,
            "autonomous_threshold": self.autonomous_threshold,
            "fallback_threshold": self.fallback_threshold,
            "confidence": self.confidence,
            "fallback_policy": self.fallback_policy,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scores": list(self.scores),
            "raw_probabilities": list(self.raw_probabilities),
            "calibrated_probabilities": list(self.calibrated_probabilities),
            "selected_index": self.selected_index,
            "model_action": self.model_action.to_dict(),
            "final_action": self.final_action.to_dict(),
        }


def write_routing_traces_jsonl(path: str | Path, traces: Sequence[RoutingTrace]) -> None:
    """Write routing evidence as deterministic, Sites-readable JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for trace in traces:
            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
