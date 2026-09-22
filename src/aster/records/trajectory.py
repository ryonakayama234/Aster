"""A complete ordered sequence of semantic agent transitions."""

from dataclasses import dataclass

from aster.records.transition import Transition


@dataclass(frozen=True, slots=True)
class Trajectory:
    transitions: tuple[Transition, ...] = ()

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
