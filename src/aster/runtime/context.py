"""Mutable environment state owned by the runtime, not by the policy."""

from copy import deepcopy
from dataclasses import dataclass, field

from aster.records.transition import JsonValue
from aster.runtime.state import RuntimeState


@dataclass(slots=True)
class RuntimeContext:
    task: dict[str, JsonValue]
    memory: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self):
        self.task = deepcopy(self.task)
        self.memory = deepcopy(self.memory)

    def snapshot(self, step: int) -> RuntimeState:
        return RuntimeState(task=self.task, memory=self.memory, step=step)
