"""Supervised imitation utilities for AsterDecision-v0."""

from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch
from torch.nn import functional as F

from aster.agent.candidates import CandidateBuilder, CalculateAndStoreCandidates
from aster.inference.decide import score_candidates
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.trajectory import Trajectory
from aster.records.transition import JsonValue
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import AsterTokenizer


@dataclass(frozen=True, slots=True)
class DecisionTrainConfig:
    steps: int = 100
    learning_rate: float = 3e-3
    train_backbone: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def examples_from_teacher_trajectory(
    trajectory: Trajectory,
    *,
    candidate_builder: CandidateBuilder | None = None,
    teacher: str = "rule-v0",
) -> list[DecisionExample]:
    """Convert execution truth into a separate supervised decision view."""
    builder = candidate_builder or CalculateAndStoreCandidates()
    examples: list[DecisionExample] = []

    for index, transition in enumerate(trajectory.transitions):
        state_data = transition.state_before
        state = RuntimeState(
            task=_require_mapping(state_data["task"], "task"),
            memory=_require_mapping(state_data["memory"], "memory"),
            step=_require_step(state_data["step"]),
        )
        history = Trajectory(trajectory.transitions[:index])
        candidates = builder.build(state, history, transition.available_actions)
        try:
            target_index = candidates.index(transition.action)
        except ValueError as error:
            raise ValueError(
                f"Teacher action at step {transition.step} is absent from candidate set"
            ) from error

        examples.append(
            DecisionExample(
                state=state,
                trajectory=history,
                candidates=candidates,
                target_index=target_index,
                teacher=teacher,
            )
        )

    return examples


def _require_mapping(value: JsonValue, field: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"Runtime state field {field!r} must be an object")
    return value


def _require_step(value: JsonValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Runtime state step must be an integer")
    return value


def decision_loss(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    example: DecisionExample,
) -> torch.Tensor:
    scores = score_candidates(
        model,
        tokenizer,
        example.state,
        example.trajectory,
        example.candidates,
    )
    target = torch.tensor([example.target_index], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores.unsqueeze(0), target)


def evaluate_decisions(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("At least one decision example is required")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for example in examples:
            scores = score_candidates(
                model,
                tokenizer,
                example.state,
                example.trajectory,
                example.candidates,
            )
            target = torch.tensor([example.target_index], dtype=torch.long, device=scores.device)
            total_loss += float(F.cross_entropy(scores.unsqueeze(0), target).item())
            correct += int(torch.argmax(scores).item() == example.target_index)
    if was_training:
        model.train()

    return {
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "nll": total_loss / len(examples),
    }


def evaluate_decision_diagnostics(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
) -> dict[str, float | int | None]:
    """Evaluate accuracy/NLL plus target-vs-best-wrong score margin.

    DecisionExample permits a single candidate. Such states remain valid for
    accuracy/NLL but have no "best wrong" candidate, so their margin is
    intentionally undefined rather than fabricated.
    """
    if not examples:
        raise ValueError("At least one decision example is required")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    margins: list[float] = []
    singleton_candidate_examples = 0
    with torch.no_grad():
        for example in examples:
            scores = score_candidates(
                model,
                tokenizer,
                example.state,
                example.trajectory,
                example.candidates,
            )
            target = torch.tensor(
                [example.target_index], dtype=torch.long, device=scores.device
            )
            total_loss += float(F.cross_entropy(scores.unsqueeze(0), target).item())
            correct += int(torch.argmax(scores).item() == example.target_index)

            if scores.numel() == 1:
                singleton_candidate_examples += 1
                continue

            target_score = scores[example.target_index]
            wrong_mask = torch.ones_like(scores, dtype=torch.bool)
            wrong_mask[example.target_index] = False
            best_wrong = torch.max(scores[wrong_mask])
            margins.append(float((target_score - best_wrong).item()))
    if was_training:
        model.train()

    return {
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "nll": total_loss / len(examples),
        "margin_examples": len(margins),
        "singleton_candidate_examples": singleton_candidate_examples,
        "margin_mean": None if not margins else sum(margins) / len(margins),
        "margin_min": None if not margins else min(margins),
    }


def train_decision_with_checkpoints(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
    *,
    checkpoints: Sequence[int],
    config: DecisionTrainConfig = DecisionTrainConfig(),
    trainable_parameter_names: Collection[str] | None = None,
    loss_fn: Callable[[DecisionModel, AsterTokenizer, DecisionExample], torch.Tensor] | None = None,
) -> tuple[list[float], dict[int, DecisionModel], dict[int, float]]:
    """Train once while snapshotting model weights at fixed optimizer steps.

    Checkpoints are repeated observations from one training trajectory. They are
    not independent samples. The optimizer is created once and preserved across
    all requested checkpoints.
    """
    if not examples:
        raise ValueError("At least one decision example is required")
    points = tuple(checkpoints)
    if not points:
        raise ValueError("At least one checkpoint is required")
    if any(type(step) is not int for step in points):
        raise ValueError("Checkpoints must be integers")
    if tuple(sorted(set(points))) != points:
        raise ValueError("Checkpoints must be unique and sorted")
    if points[0] < 0 or points[-1] > config.steps:
        raise ValueError("Checkpoints must fall within 0..config.steps")

    torch.manual_seed(config.seed)
    _configure_trainable_parameters(
        model,
        config=config,
        trainable_parameter_names=trainable_parameter_names,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    model.train()
    losses: list[float] = []
    snapshots: dict[int, DecisionModel] = {}
    elapsed_wall_seconds: dict[int, float] = {}
    requested = set(points)
    started = perf_counter()

    if 0 in requested:
        snapshots[0] = deepcopy(model)
        elapsed_wall_seconds[0] = 0.0

    for step in range(config.steps):
        example = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        loss = (decision_loss if loss_fn is None else loss_fn)(model, tokenizer, example)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

        completed_steps = step + 1
        if completed_steps in requested:
            snapshots[completed_steps] = deepcopy(model)
            elapsed_wall_seconds[completed_steps] = perf_counter() - started

    if set(snapshots) != requested:
        missing = sorted(requested.difference(snapshots))
        raise RuntimeError(f"Decision training checkpoints were not captured: {missing}")
    if set(elapsed_wall_seconds) != requested:
        missing = sorted(requested.difference(elapsed_wall_seconds))
        raise RuntimeError(f"Decision checkpoint timings were not captured: {missing}")
    return losses, snapshots, elapsed_wall_seconds


def train_decision(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
    config: DecisionTrainConfig = DecisionTrainConfig(),
    *,
    trainable_parameter_names: Collection[str] | None = None,
) -> list[float]:
    """Small deterministic behavior-cloning loop for the v0 candidate scorer."""
    if not examples:
        raise ValueError("At least one decision example is required")

    torch.manual_seed(config.seed)
    _configure_trainable_parameters(
        model,
        config=config,
        trainable_parameter_names=trainable_parameter_names,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    model.train()
    losses: list[float] = []

    for step in range(config.steps):
        example = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        loss = decision_loss(model, tokenizer, example)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return losses


def _configure_trainable_parameters(
    model: DecisionModel,
    *,
    config: DecisionTrainConfig,
    trainable_parameter_names: Collection[str] | None,
) -> None:
    """Apply legacy backbone/head semantics or an explicit parameter mask.

    The explicit mask is an opt-in override used by causal diagnostics that need
    finer control than the historical `train_backbone` boolean. Leaving the
    mask unset preserves all existing Decision training behavior.
    """
    if trainable_parameter_names is None:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(config.train_backbone)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        return

    requested = frozenset(trainable_parameter_names)
    if not requested:
        raise ValueError("Explicit trainable parameter mask must not be empty")

    named_parameters = dict(model.named_parameters())
    unknown = requested.difference(named_parameters)
    if unknown:
        raise ValueError(
            "Unknown trainable parameter names: " + ", ".join(sorted(unknown))
        )

    for name, parameter in named_parameters.items():
        parameter.requires_grad_(name in requested)
