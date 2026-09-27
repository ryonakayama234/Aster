"""Records for explicit forced-action counterfactual branch evaluation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aster.records.transition import Action


_STATUSES = {"completed", "unsupported_counterfactual_tool"}


@dataclass(frozen=True, slots=True)
class CounterfactualTrace:
    """One candidate branch under a fixed continuation policy."""

    source_step: int
    candidate_index: int
    model_id: str
    source_route: str
    action: Action
    model_probability: float
    model_selected: bool
    source_executed: bool
    continuation_policy: str
    status: str
    branch_dir: str | None = None
    branch_reward_total: float | None = None
    branch_steps: int | None = None
    task_success: bool | None = None
    terminal_reached: bool | None = None
    baseline_reward: float | None = None
    advantage_estimate: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.source_step < 0 or self.candidate_index < 0:
            raise ValueError("Counterfactual trace indices must be non-negative")
        if not self.model_id or not self.continuation_policy:
            raise ValueError("Counterfactual trace IDs must be non-empty")
        if self.status not in _STATUSES:
            raise ValueError(f"Unknown counterfactual status: {self.status}")
        if not math.isfinite(self.model_probability) or not 0.0 <= self.model_probability <= 1.0:
            raise ValueError("model_probability must be finite and in [0, 1]")
        for name, value in (
            ("branch_reward_total", self.branch_reward_total),
            ("baseline_reward", self.baseline_reward),
            ("advantage_estimate", self.advantage_estimate),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        if self.status == "completed":
            if (
                self.branch_dir is None
                or self.branch_reward_total is None
                or self.branch_steps is None
                or self.task_success is None
                or self.terminal_reached is None
            ):
                raise ValueError("Completed counterfactual traces require branch outcome evidence")
        elif any(
            value is not None
            for value in (
                self.branch_dir,
                self.branch_reward_total,
                self.branch_steps,
                self.task_success,
                self.terminal_reached,
            )
        ):
            raise ValueError("Unsupported counterfactual traces cannot claim branch outcome evidence")
        if self.branch_steps is not None and self.branch_steps < 0:
            raise ValueError("branch_steps must be non-negative")
        if (self.baseline_reward is None) != (self.advantage_estimate is None):
            raise ValueError("baseline_reward and advantage_estimate must be present together")

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-counterfactual-trace-0",
            "source_step": self.source_step,
            "candidate_index": self.candidate_index,
            "model_id": self.model_id,
            "source_route": self.source_route,
            "action": self.action.to_dict(),
            "model_probability": self.model_probability,
            "model_selected": self.model_selected,
            "source_executed": self.source_executed,
            "continuation_policy": self.continuation_policy,
            "status": self.status,
            "branch_dir": self.branch_dir,
            "branch_reward_total": self.branch_reward_total,
            "branch_steps": self.branch_steps,
            "task_success": self.task_success,
            "terminal_reached": self.terminal_reached,
            "baseline_reward": self.baseline_reward,
            "advantage_estimate": self.advantage_estimate,
            "value_scope": "full-episode reward under forced action and fixed continuation policy",
            "note": self.note,
        }


def write_counterfactual_traces_jsonl(
    path: str | Path,
    traces: Sequence[CounterfactualTrace],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for trace in traces:
            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
