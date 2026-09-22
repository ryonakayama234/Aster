"""Semantic records for state -> action -> observation -> next state."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

from aster.evaluator.result import EvaluationResult

JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str
    message: str
    details: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "details", deepcopy(self.details))

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": deepcopy(self.details)}

    @classmethod
    def from_dict(cls, data: dict) -> "ErrorInfo":
        return cls(code=str(data["code"]), message=str(data["message"]), details=data.get("details", {}))


@dataclass(frozen=True, slots=True)
class Action:
    """A request chosen by a policy. It does not execute anything by itself."""

    kind: str
    name: str | None = None
    arguments: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"tool", "stop"}:
            raise ValueError("Action kind must be 'tool' or 'stop'")
        if self.kind == "tool" and not self.name:
            raise ValueError("Tool actions require a name")
        if self.kind == "stop" and self.name is not None:
            raise ValueError("Stop actions do not have a tool name")
        object.__setattr__(self, "arguments", deepcopy(self.arguments))

    @classmethod
    def tool(cls, name: str, **arguments: JsonValue) -> "Action":
        return cls(kind="tool", name=name, arguments=arguments)

    @classmethod
    def stop(cls, reason: str = "policy_stop") -> "Action":
        return cls(kind="stop", arguments={"reason": reason})

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "arguments": deepcopy(self.arguments)}

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        return cls(kind=str(data["kind"]), name=data.get("name"), arguments=data.get("arguments", {}))


@dataclass(frozen=True, slots=True)
class Observation:
    """What the runtime observed after attempting an action."""

    accepted: bool
    ok: bool
    output: JsonValue = None
    error: ErrorInfo | None = None

    def __post_init__(self):
        if self.ok and not self.accepted:
            raise ValueError("A successful observation must have been accepted")
        if self.ok and self.error is not None:
            raise ValueError("Successful observations cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("Failed observations require structured error information")
        object.__setattr__(self, "output", deepcopy(self.output))

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "ok": self.ok,
            "output": deepcopy(self.output),
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        error = data.get("error")
        return cls(
            accepted=bool(data["accepted"]),
            ok=bool(data["ok"]),
            output=data.get("output"),
            error=None if error is None else ErrorInfo.from_dict(error),
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """One immutable unit of agent experience."""

    step: int
    state_before: dict[str, JsonValue]
    available_actions: tuple[str, ...]
    action: Action
    observation: Observation
    state_after: dict[str, JsonValue]
    evaluation: EvaluationResult

    def __post_init__(self):
        if self.step < 0:
            raise ValueError("Transition step must be non-negative")
        object.__setattr__(self, "state_before", deepcopy(self.state_before))
        object.__setattr__(self, "state_after", deepcopy(self.state_after))
        object.__setattr__(self, "available_actions", tuple(self.available_actions))

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-transition-0",
            "step": self.step,
            "state_before": deepcopy(self.state_before),
            "available_actions": list(self.available_actions),
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "state_after": deepcopy(self.state_after),
            "evaluation": self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        if data.get("schema_version") != "aster-transition-0":
            raise ValueError("Unsupported transition schema")
        return cls(
            step=int(data["step"]),
            state_before=data["state_before"],
            available_actions=tuple(data["available_actions"]),
            action=Action.from_dict(data["action"]),
            observation=Observation.from_dict(data["observation"]),
            state_after=data["state_after"],
            evaluation=EvaluationResult.from_dict(data["evaluation"]),
        )
