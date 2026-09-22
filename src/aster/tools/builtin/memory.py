"""Explicit runtime memory tools. Policies cannot mutate memory directly."""

from copy import deepcopy

from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.tools.schema import ArgumentSpec, ToolExecutionError, ToolSpec

MEMORY_PUT_SPEC = ToolSpec(
    name="memory.put",
    description="Store one JSON-compatible value under a key.",
    arguments=(ArgumentSpec("key", "string"), ArgumentSpec("value", "any")),
)

MEMORY_GET_SPEC = ToolSpec(
    name="memory.get",
    description="Read one previously stored value by key.",
    arguments=(ArgumentSpec("key", "string"),),
)


def memory_put(context: RuntimeContext, arguments: dict[str, JsonValue]) -> JsonValue:
    key = arguments["key"]
    if not isinstance(key, str):
        raise ToolExecutionError("invalid_key", "memory key must be a string")
    context.memory[key] = deepcopy(arguments["value"])
    return {"key": key, "stored": True}


def memory_get(context: RuntimeContext, arguments: dict[str, JsonValue]) -> JsonValue:
    key = arguments["key"]
    if not isinstance(key, str):
        raise ToolExecutionError("invalid_key", "memory key must be a string")
    if key not in context.memory:
        raise ToolExecutionError("memory_key_not_found", f"No value stored for key: {key}")
    return {"key": key, "value": deepcopy(context.memory[key])}
