"""Validate tool actions, execute registered handlers, and return observations."""

from copy import deepcopy

from aster.records.transition import Action, ErrorInfo, Observation
from aster.runtime.context import RuntimeContext
from aster.tools.registry import ToolRegistry
from aster.tools.schema import ToolExecutionError


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, action: Action, context: RuntimeContext) -> Observation:
        if action.kind != "tool" or action.name is None:
            return Observation(
                accepted=False,
                ok=False,
                error=ErrorInfo("invalid_action", "ToolExecutor only accepts tool actions"),
            )

        registered = self.registry.get(action.name)
        if registered is None:
            return Observation(
                accepted=False,
                ok=False,
                error=ErrorInfo("tool_not_found", f"Unknown tool: {action.name}"),
            )

        try:
            registered.spec.validate_arguments(action.arguments)
        except ValueError as exc:
            return Observation(
                accepted=False,
                ok=False,
                error=ErrorInfo("invalid_arguments", str(exc)),
            )

        try:
            output = registered.handler(context, deepcopy(action.arguments))
        except ToolExecutionError as exc:
            return Observation(
                accepted=True,
                ok=False,
                error=ErrorInfo(exc.code, exc.message),
            )
        except Exception as exc:  # Keep unexpected tool failures observable to the agent runtime.
            return Observation(
                accepted=True,
                ok=False,
                error=ErrorInfo("tool_execution_error", f"{type(exc).__name__}: {exc}"),
            )
        return Observation(accepted=True, ok=True, output=output)
