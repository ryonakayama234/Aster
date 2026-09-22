"""Supervised imitation utilities for AsterDecision-v0."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from aster.agent.candidates import CandidateBuilder, CalculateAndStoreCandidates
from aster.inference.decide import score_candidates
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.trajectory import Trajectory
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
            task=state_data["task"],
            memory=state_data["memory"],
            step=int(state_data["step"]),
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


def train_decision(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
    config: DecisionTrainConfig = DecisionTrainConfig(),
) -> list[float]:
    """Small deterministic behavior-cloning loop for the v0 candidate scorer."""
    if not examples:
        raise ValueError("At least one decision example is required")

    torch.manual_seed(config.seed)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(config.train_backbone)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

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
