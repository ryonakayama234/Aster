"""Version identifiers for achievement and current-state completion."""

EVER_GOAL_V0 = "calculate-and-store-ever-v0"
CURRENT_GOAL_V1 = "calculate-and-store-current-v1"


def validate_goal_contract(goal_contract: str) -> None:
    if goal_contract not in (EVER_GOAL_V0, CURRENT_GOAL_V1):
        raise ValueError(f"Unsupported goal contract: {goal_contract}")
