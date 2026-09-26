"""Auditable credit assignment from episode reward to directly evidenced steps."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from aster.records.trajectory import Trajectory
from aster.reward.contract import RewardResult


SCHEMA_VERSION = "aster-credit-0"
METHOD_ID = "direct-evidence-v0"
_REQUIRED_COMPONENTS = {
    "task_success",
    "task_failure",
    "rejected_action",
    "failed_execution",
    "step_cost",
}


@dataclass(frozen=True, slots=True)
class CreditComponent:
    """One reward contribution assigned to one observed transition."""

    source_reward_component: str
    amount: float

    def __post_init__(self) -> None:
        if not self.source_reward_component:
            raise ValueError("source_reward_component must be non-empty")
        _require_finite("amount", self.amount)

    def to_dict(self) -> dict:
        return {
            "source_reward_component": self.source_reward_component,
            "amount": float(self.amount),
        }


@dataclass(frozen=True, slots=True)
class StepCredit:
    """Directly evidenced reward contributions for one executed step."""

    step: int
    components: tuple[CreditComponent, ...]

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")

    @property
    def total(self) -> float:
        return sum(component.amount for component in self.components)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "components": [component.to_dict() for component in self.components],
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class UnassignedCredit:
    """Reward intentionally left without step attribution by this method."""

    source_reward_component: str
    amount: float
    reason: str

    def __post_init__(self) -> None:
        if not self.source_reward_component:
            raise ValueError("source_reward_component must be non-empty")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        _require_finite("amount", self.amount)

    def to_dict(self) -> dict:
        return {
            "source_reward_component": self.source_reward_component,
            "amount": float(self.amount),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CreditResult:
    """Conservative partition of reward into assigned and explicitly unassigned credit."""

    method_id: str
    reward_spec_id: str
    reward_total: float
    steps: tuple[StepCredit, ...]
    unassigned: tuple[UnassignedCredit, ...]

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must be non-empty")
        if not self.reward_spec_id:
            raise ValueError("reward_spec_id must be non-empty")
        _require_finite("reward_total", self.reward_total)
        step_ids = [step.step for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("CreditResult step IDs must be unique")
        if not math.isclose(
            self.assigned_total + self.unassigned_total,
            self.reward_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Assigned plus unassigned credit must conserve reward total")

    @property
    def assigned_total(self) -> float:
        return sum(step.total for step in self.steps)

    @property
    def unassigned_total(self) -> float:
        return sum(item.amount for item in self.unassigned)

    def to_dict(self) -> dict:
        return {
            "method_id": self.method_id,
            "reward_spec_id": self.reward_spec_id,
            "reward_total": float(self.reward_total),
            "steps": [step.to_dict() for step in self.steps],
            "unassigned": [item.to_dict() for item in self.unassigned],
            "assigned_total": self.assigned_total,
            "unassigned_total": self.unassigned_total,
        }


def assign_direct_credit(
    trajectory: Trajectory,
    reward: RewardResult,
) -> CreditResult:
    """Assign only reward whose responsible transition is directly observable.

    Step cost, rejected-action cost, and failed-execution cost are mapped back to
    the transitions counted by EpisodeEvaluation. Episode-level task success or
    failure remains explicitly unassigned; this method makes no causal claim.
    """

    components = {component.name: component for component in reward.components}
    if set(components) != _REQUIRED_COMPONENTS:
        missing = sorted(_REQUIRED_COMPONENTS - set(components))
        extra = sorted(set(components) - _REQUIRED_COMPONENTS)
        raise ValueError(
            f"direct-evidence-v0 requires Reward Contract v0 components; "
            f"missing={missing}, extra={extra}"
        )

    expected_quantities = {
        "task_success": int(trajectory.successful),
        "task_failure": int(not trajectory.successful),
        "rejected_action": sum(
            not transition.observation.accepted for transition in trajectory.transitions
        ),
        "failed_execution": sum(
            transition.observation.accepted and not transition.observation.ok
            for transition in trajectory.transitions
        ),
        "step_cost": len(trajectory.transitions),
    }
    for name, expected in expected_quantities.items():
        actual = components[name].quantity
        if actual != expected:
            raise ValueError(
                f"Reward component {name!r} quantity {actual} does not match "
                f"trajectory-derived quantity {expected}"
            )

    step_credits: list[StepCredit] = []
    for transition in trajectory.transitions:
        direct = [
            CreditComponent(
                source_reward_component="step_cost",
                amount=components["step_cost"].weight,
            )
        ]
        if not transition.observation.accepted:
            direct.append(
                CreditComponent(
                    source_reward_component="rejected_action",
                    amount=components["rejected_action"].weight,
                )
            )
        if transition.observation.accepted and not transition.observation.ok:
            direct.append(
                CreditComponent(
                    source_reward_component="failed_execution",
                    amount=components["failed_execution"].weight,
                )
            )
        step_credits.append(StepCredit(step=transition.step, components=tuple(direct)))

    unassigned = tuple(
        UnassignedCredit(
            source_reward_component=name,
            amount=components[name].amount,
            reason="episode-level outcome has no step attribution in direct-evidence-v0",
        )
        for name in ("task_success", "task_failure")
        if components[name].quantity > 0
    )

    return CreditResult(
        method_id=METHOD_ID,
        reward_spec_id=reward.spec_id,
        reward_total=reward.total,
        steps=tuple(step_credits),
        unassigned=unassigned,
    )


def write_credit_result(
    path: str | Path,
    trajectory: Trajectory,
    reward: RewardResult,
    *,
    trajectory_file: str = "trajectory.jsonl",
    reward_file: str = "reward.json",
) -> CreditResult:
    """Write a Sites-readable credit view that references its evidence inputs."""

    result = assign_direct_credit(trajectory, reward)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_file": trajectory_file,
        "reward_file": reward_file,
        "result": result.to_dict(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _require_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
