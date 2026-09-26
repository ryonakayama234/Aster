"""Tool descriptions, argument validation, and execution errors."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from aster.records.transition import JsonValue

if TYPE_CHECKING:
    from aster.runtime.context import RuntimeContext

_ALLOWED_KINDS = {"any", "string", "number", "integer", "boolean", "object", "array"}


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ArgumentSpec:
    name: str
    kind: str = "any"
    required: bool = True
    choices: tuple[JsonValue, ...] = ()

    def __post_init__(self):
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported argument kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    arguments: tuple[ArgumentSpec, ...] = ()
    counterfactual_safe: bool = False

    def __post_init__(self):
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("Tool argument names must be unique")

    def validate_arguments(self, values: dict[str, JsonValue]) -> None:
        expected = {argument.name: argument for argument in self.arguments}
        unknown = sorted(set(values) - set(expected))
        if unknown:
            raise ValueError(f"Unknown arguments: {', '.join(unknown)}")
        for argument in self.arguments:
            if argument.required and argument.name not in values:
                raise ValueError(f"Missing required argument: {argument.name}")
            if argument.name not in values:
                continue
            value = values[argument.name]
            if not _matches_kind(value, argument.kind):
                raise ValueError(f"Argument {argument.name} must be {argument.kind}")
            if argument.choices and value not in argument.choices:
                raise ValueError(f"Argument {argument.name} must be one of {argument.choices}")


def _matches_kind(value: JsonValue, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    return False


ToolHandler = Callable[["RuntimeContext", dict[str, JsonValue]], JsonValue]
