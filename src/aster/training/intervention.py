"""Dataset aggregation and supervised v+1 updates from intervention traces."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.intervention import InterventionTrace
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.decision import (
    DecisionTrainConfig,
    evaluate_decisions,
    train_decision,
)


def decision_example_from_intervention(trace: InterventionTrace) -> DecisionExample:
    """Convert one trainable shadow-teacher label into the existing DecisionExample."""
    if trace.teacher_target_index is None:
        raise ValueError("Teacher Action is absent from the student candidate set")
    return DecisionExample(
        state=trace.state,
        trajectory=trace.trajectory,
        candidates=trace.candidates,
        target_index=trace.teacher_target_index,
        teacher=trace.teacher_id,
    )


def examples_from_interventions(
    traces: Sequence[InterventionTrace],
    *,
    disagreements_only: bool = False,
) -> tuple[DecisionExample, ...]:
    """Build DAgger-style labels on every trainable student-visited state by default.

    Agreements are intentionally retained: DAgger learns the expert label on the learner's
    state distribution, not only a filtered list of mistakes.
    """
    examples: list[DecisionExample] = []
    for trace in traces:
        if not trace.trainable:
            continue
        if disagreements_only and trace.agrees:
            continue
        examples.append(decision_example_from_intervention(trace))
    return tuple(examples)


def aggregate_decision_examples(
    base_examples: Sequence[DecisionExample],
    intervention_examples: Sequence[DecisionExample],
) -> tuple[DecisionExample, ...]:
    """Append new supervision without silently replacing or deduplicating old data."""
    return tuple(base_examples) + tuple(intervention_examples)


@dataclass(slots=True)
class DecisionCandidateUpdate:
    """An in-memory candidate model plus auditable before/after measurements."""

    model: DecisionModel
    training_examples: tuple[DecisionExample, ...]
    intervention_examples: tuple[DecisionExample, ...]
    losses: tuple[float, ...]
    aggregate_before: dict[str, float | int]
    aggregate_after: dict[str, float | int]
    intervention_before: dict[str, float | int]
    intervention_after: dict[str, float | int]

    def summary(self) -> dict:
        return {
            "schema_version": "aster-intervention-update-0",
            "training_examples": len(self.training_examples),
            "intervention_examples": len(self.intervention_examples),
            "aggregate_before": self.aggregate_before,
            "aggregate_after": self.aggregate_after,
            "intervention_before": self.intervention_before,
            "intervention_after": self.intervention_after,
            "steps": len(self.losses),
            "first_loss": self.losses[0],
            "last_loss": self.losses[-1],
        }


def train_intervention_candidate(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    base_examples: Sequence[DecisionExample],
    traces: Sequence[InterventionTrace],
    *,
    config: DecisionTrainConfig = DecisionTrainConfig(),
) -> DecisionCandidateUpdate:
    """Train a deep-copied candidate so the parent model remains unchanged."""
    intervention_examples = examples_from_interventions(traces)
    if not intervention_examples:
        raise ValueError("At least one trainable intervention is required")
    training_examples = aggregate_decision_examples(base_examples, intervention_examples)
    if not training_examples:
        raise ValueError("At least one training example is required")

    candidate = deepcopy(model)
    aggregate_before = evaluate_decisions(candidate, tokenizer, training_examples)
    intervention_before = evaluate_decisions(candidate, tokenizer, intervention_examples)
    losses = tuple(train_decision(candidate, tokenizer, training_examples, config))
    aggregate_after = evaluate_decisions(candidate, tokenizer, training_examples)
    intervention_after = evaluate_decisions(candidate, tokenizer, intervention_examples)

    return DecisionCandidateUpdate(
        model=candidate,
        training_examples=training_examples,
        intervention_examples=intervention_examples,
        losses=losses,
        aggregate_before=aggregate_before,
        aggregate_after=aggregate_after,
        intervention_before=intervention_before,
        intervention_after=intervention_after,
    )
