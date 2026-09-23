"""Confidence-gated routing around a model-backed decision policy."""

from dataclasses import dataclass
import math
from typing import Protocol

from aster.agent.policy import Policy
from aster.records.decision import DecisionTrace
from aster.records.routing import RoutingTrace, write_routing_traces_jsonl
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.state import RuntimeState


class TracedDecisionPolicy(Protocol):
    """Small boundary needed by SelectivePolicy from a model-backed policy."""

    model_id: str
    traces: list[DecisionTrace]

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action: ...


@dataclass(frozen=True, slots=True)
class SelectivePolicyConfig:
    """Operational thresholds for calibrated model confidence."""

    temperature: float = 1.0
    autonomous_threshold: float = 0.8
    fallback_threshold: float = 0.6

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        for threshold in (self.fallback_threshold, self.autonomous_threshold):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("routing thresholds must be finite values in [0, 1]")
        if self.fallback_threshold > self.autonomous_threshold:
            raise ValueError("fallback_threshold cannot exceed autonomous_threshold")

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "autonomous_threshold": self.autonomous_threshold,
            "fallback_threshold": self.fallback_threshold,
        }


class SelectivePolicy:
    """Use calibrated confidence to choose model, fallback, or abstention.

    The wrapped model policy remains responsible for producing DecisionTrace evidence.
    This adapter does not execute tools and does not modify Transition semantics.
    """

    def __init__(
        self,
        primary: TracedDecisionPolicy,
        *,
        fallback: Policy | None = None,
        config: SelectivePolicyConfig = SelectivePolicyConfig(),
        fallback_id: str | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.config = config
        self.fallback_id = (
            fallback_id
            if fallback_id is not None
            else (None if fallback is None else type(fallback).__name__)
        )
        self.routing_traces: list[RoutingTrace] = []

    @property
    def model_id(self) -> str:
        return self.primary.model_id

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        before = len(self.primary.traces)
        model_action = self.primary.decide(state, trajectory, available_actions)
        if len(self.primary.traces) != before + 1:
            raise RuntimeError("Primary decision policy must append exactly one DecisionTrace")

        decision = self.primary.traces[-1]
        if decision.step != state.step:
            raise RuntimeError("DecisionTrace step does not match RuntimeState step")
        if decision.selected != model_action:
            raise RuntimeError("DecisionTrace selected Action does not match primary Policy result")

        calibrated = _probabilities_from_scores(decision.scores, self.config.temperature)
        confidence = max(calibrated)

        if confidence >= self.config.autonomous_threshold:
            route = "model"
            final_action = model_action
        elif confidence >= self.config.fallback_threshold and self.fallback is not None:
            route = "fallback"
            final_action = self.fallback.decide(state, trajectory, available_actions)
        else:
            route = "abstain"
            final_action = Action.stop("low_confidence")

        self.routing_traces.append(
            RoutingTrace(
                step=state.step,
                model_id=decision.model_id,
                route=route,
                temperature=self.config.temperature,
                autonomous_threshold=self.config.autonomous_threshold,
                fallback_threshold=self.config.fallback_threshold,
                candidates=decision.candidates,
                scores=decision.scores,
                raw_probabilities=decision.probabilities,
                calibrated_probabilities=calibrated,
                selected_index=decision.selected_index,
                final_action=final_action,
                fallback_policy=self.fallback_id,
            )
        )
        return final_action

    def clear_routing_traces(self) -> None:
        self.routing_traces.clear()

    def write_routing_jsonl(self, path) -> None:
        write_routing_traces_jsonl(path, self.routing_traces)


def _probabilities_from_scores(scores: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    if not scores:
        raise ValueError("At least one score is required")
    scaled = [float(score) / temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
