"""Small structured calculator used by Agent Kernel v0."""

from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.tools.schema import ArgumentSpec, ToolExecutionError, ToolSpec

CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description="Apply one basic arithmetic operation to two numbers.",
    arguments=(
        ArgumentSpec("operation", "string", choices=("add", "subtract", "multiply", "divide")),
        ArgumentSpec("left", "number"),
        ArgumentSpec("right", "number"),
    ),
)


def calculator(_context: RuntimeContext, arguments: dict[str, JsonValue]) -> JsonValue:
    operation = arguments["operation"]
    left = arguments["left"]
    right = arguments["right"]

    if not isinstance(left, (int, float)) or isinstance(left, bool):
        raise ToolExecutionError("invalid_number", "left must be a number")
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        raise ToolExecutionError("invalid_number", "right must be a number")

    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if operation == "divide":
        if right == 0:
            raise ToolExecutionError("division_by_zero", "Cannot divide by zero")
        return left / right
    raise ToolExecutionError("unsupported_operation", f"Unsupported operation: {operation}")
