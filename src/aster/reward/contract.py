"""Explicit mapping from episode evaluation to inspectable scalar reward."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from aster.evaluator.episode import EpisodeEvaluation


SCHEMA_VERSION = "aster-episode-reward-0"


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Versionable value judgment applied to one EpisodeEvaluation."""

    spec_id: str
    task_success: float = 1.0
    task_failure: float = 0.0
    rejected_action: float = 0.0
    failed_execution: float = 0.0
    step_cost: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.spec_id, str) or not self.spec_id.strip():
            raise ValueError("spec_id must be a non-empty string")
        for name in (
            "task_success",
            "task_failure",
            "rejected_action",
            "failed_execution",
            "step_cost",
        ):
            _require_finite_number(name, getattr(self, name))

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "task_success": float(self.task_success),
            "task_failure": float(self.task_failure),
            "rejected_action": float(self.rejected_action),
            "failed_execution": float(self.failed_execution),
            "step_cost": float(self.step_cost),
        }


@dataclass(frozen=True, slots=True)
class RewardComponent:
    """One auditable contribution to a scalar episode reward."""

    name: str
    weight: float
    quantity: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Reward component name must be a non-empty string")
        _require_finite_number("weight", self.weight)
        if type(self.quantity) is not int or self.quantity < 0:
            raise ValueError("Reward component quantity must be a non-negative integer")

    @property
    def amount(self) -> float:
        return float(self.weight) * self.quantity

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "quantity": self.quantity,
            "amount": self.amount,
        }


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Component-wise reward result without any policy-update semantics."""

    spec_id: str
    components: tuple[RewardComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec_id, str) or not self.spec_id.strip():
            raise ValueError("spec_id must be a non-empty string")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Reward component names must be unique")

    @property
    def total(self) -> float:
        return sum(component.amount for component in self.components)

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "components": [component.to_dict() for component in self.components],
            "total": self.total,
        }


def compute_reward(
    evaluation: EpisodeEvaluation,
    spec: RewardSpec,
) -> RewardResult:
    """Apply one explicit RewardSpec to one immutable episode evaluation."""

    components = (
        RewardComponent("task_success", spec.task_success, int(evaluation.task_success)),
        RewardComponent("task_failure", spec.task_failure, int(not evaluation.task_success)),
        RewardComponent("rejected_action", spec.rejected_action, evaluation.rejected_actions),
        RewardComponent("failed_execution", spec.failed_execution, evaluation.failed_executions),
        RewardComponent("step_cost", spec.step_cost, evaluation.steps),
    )
    return RewardResult(spec_id=spec.spec_id, components=components)


def write_reward_result(
    path: str | Path,
    evaluation: EpisodeEvaluation,
    spec: RewardSpec,
    *,
    trajectory_file: str = "trajectory.jsonl",
    evaluation_file: str = "evaluation.json",
) -> RewardResult:
    """Write a Sites-readable reward view while keeping evaluation as its input truth."""

    result = compute_reward(evaluation, spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_file": trajectory_file,
        "evaluation_file": evaluation_file,
        "spec": spec.to_dict(),
        "result": result.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
