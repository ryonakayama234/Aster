"""Structured evaluator output kept separate from scalar rewards."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Independent evaluation axes for one agent transition."""

    action_valid: bool
    execution_success: bool
    goal_satisfied: bool
    terminal: bool = False
    notes: tuple[str, ...] = ()

    @property
    def successful_terminal(self) -> bool:
        return (
            self.terminal
            and self.action_valid
            and self.execution_success
            and self.goal_satisfied
        )

    def to_dict(self) -> dict:
        return {
            "action_valid": self.action_valid,
            "execution_success": self.execution_success,
            "goal_satisfied": self.goal_satisfied,
            "terminal": self.terminal,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(
            action_valid=bool(data["action_valid"]),
            execution_success=bool(data["execution_success"]),
            goal_satisfied=bool(data["goal_satisfied"]),
            terminal=bool(data.get("terminal", False)),
            notes=tuple(str(note) for note in data.get("notes", ())),
        )
