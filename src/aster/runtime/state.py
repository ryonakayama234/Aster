"""Immutable snapshots exposed to policies and stored in transitions."""

from copy import deepcopy
from dataclasses import dataclass

from aster.records.transition import JsonValue


@dataclass(frozen=True, slots=True)
class RuntimeState:
    task: dict[str, JsonValue]
    memory: dict[str, JsonValue]
    step: int

    def __post_init__(self):
        if self.step < 0:
            raise ValueError("Runtime step must be non-negative")
        object.__setattr__(self, "task", deepcopy(self.task))
        object.__setattr__(self, "memory", deepcopy(self.memory))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"task": deepcopy(self.task), "memory": deepcopy(self.memory), "step": self.step}
