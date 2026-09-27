"""Reward mapping kept separate from evaluation and policy learning."""

from aster.reward.contract import (
    RewardComponent,
    RewardResult,
    RewardSpec,
    compute_reward,
    write_reward_result,
)

__all__ = [
    "RewardComponent",
    "RewardResult",
    "RewardSpec",
    "compute_reward",
    "write_reward_result",
]
