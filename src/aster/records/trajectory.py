"""A complete ordered sequence of semantic agent transitions."""

from dataclasses import dataclass

from aster.records.transition import Transition


@dataclass(frozen=True, slots=True)
class Trajectory:
    transitions: tuple[Transition, ...] = ()

    def __post_init__(self) -> None:
        goal_verified = False
        for transition in self.transitions:
            if goal_verified and not transition.evaluation.goal_satisfied:
                raise ValueError(
                    "goal_satisfied must remain true after a goal has been verified"
                )
            goal_verified = goal_verified or transition.evaluation.goal_satisfied

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
