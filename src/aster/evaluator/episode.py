"""Deterministic run-level evaluation derived from recorded trajectories."""

import json
from dataclasses import dataclass
from pathlib import Path

from aster.records.trajectory import Trajectory


SCHEMA_VERSION = "aster-episode-evaluation-0"


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """Run-level evaluation derived only from immutable transition evidence."""

    task_success: bool
    terminal_reached: bool
    steps: int
    accepted_actions: int
    rejected_actions: int
    successful_executions: int
    failed_executions: int
    goal_verified: bool
    first_goal_verified_step: int | None
    stop_reason: str | None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            self.steps,
            self.accepted_actions,
            self.rejected_actions,
            self.successful_executions,
            self.failed_executions,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Episode evaluation counts must be non-negative")
        if self.accepted_actions + self.rejected_actions != self.steps:
            raise ValueError("Accepted and rejected actions must account for every step")
        if self.successful_executions + self.failed_executions != self.accepted_actions:
            raise ValueError("Execution counts must account for every accepted action")
        if self.first_goal_verified_step is not None:
            if not self.goal_verified:
                raise ValueError("A first verified-goal step requires goal_verified")
            if not 0 <= self.first_goal_verified_step < self.steps:
                raise ValueError("First verified-goal step must refer to this episode")

    def to_dict(self) -> dict:
        return {
            "task_success": self.task_success,
            "terminal_reached": self.terminal_reached,
            "steps": self.steps,
            "accepted_actions": self.accepted_actions,
            "rejected_actions": self.rejected_actions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "goal_verified": self.goal_verified,
            "first_goal_verified_step": self.first_goal_verified_step,
            "stop_reason": self.stop_reason,
            "notes": list(self.notes),
        }


def evaluate_episode(trajectory: Trajectory) -> EpisodeEvaluation:
    """Summarize one trajectory without introducing a scalar reward."""

    transitions = trajectory.transitions
    accepted_actions = sum(transition.observation.accepted for transition in transitions)
    rejected_actions = len(transitions) - accepted_actions
    successful_executions = sum(
        transition.observation.accepted and transition.observation.ok
        for transition in transitions
    )
    failed_executions = sum(
        transition.observation.accepted and not transition.observation.ok
        for transition in transitions
    )
    verified_steps = [
        transition.step
        for transition in transitions
        if transition.evaluation.goal_satisfied
    ]
    notes = tuple(
        dict.fromkeys(
            note
            for transition in transitions
            for note in transition.evaluation.notes
        )
    )

    stop_reason = None
    if transitions and transitions[-1].action.kind == "stop":
        reason = transitions[-1].action.arguments.get("reason")
        stop_reason = reason if isinstance(reason, str) else None

    return EpisodeEvaluation(
        task_success=trajectory.successful,
        terminal_reached=trajectory.stopped,
        steps=len(transitions),
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        successful_executions=successful_executions,
        failed_executions=failed_executions,
        goal_verified=bool(verified_steps),
        first_goal_verified_step=None if not verified_steps else verified_steps[0],
        stop_reason=stop_reason,
        notes=notes,
    )


def write_episode_evaluation(
    path: str | Path,
    trajectory: Trajectory,
    *,
    trajectory_file: str = "trajectory.jsonl",
) -> EpisodeEvaluation:
    """Write a Sites-readable episode summary while keeping trajectory as source of truth."""

    evaluation = evaluate_episode(trajectory)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_file": trajectory_file,
        "summary": evaluation.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation
