"""A complete ordered sequence of semantic agent transitions."""

from dataclasses import dataclass

from aster.evaluator.goal_contract import EVER_GOAL_V0

from aster.records.transition import Transition


@dataclass(frozen=True, slots=True)
class Trajectory:
    transitions: tuple[Transition, ...] = ()

    def __post_init__(self) -> None:
        contracts = {t.evaluation.goal_contract for t in self.transitions}
        if len(contracts) > 1:
            raise ValueError("A trajectory cannot mix goal contracts")
        if self.goal_contract != EVER_GOAL_V0:
            return
        goal_verified = False
        for transition in self.transitions:
            if goal_verified and not transition.evaluation.goal_satisfied:
                raise ValueError(
                    "goal_satisfied must remain true after a goal has been verified"
                )
            goal_verified = goal_verified or transition.evaluation.goal_satisfied

    @property
    def goal_contract(self) -> str:
        return (EVER_GOAL_V0 if not self.transitions
                else self.transitions[0].evaluation.goal_contract)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def last(self) -> Transition | None:
        return None if not self.transitions else self.transitions[-1]

    @property
    def stopped(self) -> bool:
        return self.last is not None and self.last.action.kind == "stop"

    @property
    def successful(self) -> bool:
        return self.last is not None and self.last.evaluation.successful_terminal
