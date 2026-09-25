import hashlib
import json

import pytest

from aster.evaluator.episode import evaluate_episode, write_episode_evaluation
from aster.evaluator.result import EvaluationResult
from aster.records.recorder import TrajectoryRecorder
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, ErrorInfo, Observation, Transition


SUMMARY_FIELDS = {
    "task_success",
    "terminal_reached",
    "steps",
    "accepted_actions",
    "rejected_actions",
    "successful_executions",
    "failed_executions",
    "goal_verified",
    "first_goal_verified_step",
    "stop_reason",
    "notes",
}


def test_episode_evaluation_separates_rejection_execution_failure_and_goal_verification():
    trajectory = _example_trajectory()

    evaluation = evaluate_episode(trajectory)

    assert evaluation.task_success is True
    assert evaluation.terminal_reached is True
    assert evaluation.steps == 4
    assert evaluation.accepted_actions == 3
    assert evaluation.rejected_actions == 1
    assert evaluation.successful_executions == 2
    assert evaluation.failed_executions == 1
    assert evaluation.goal_verified is True
    assert evaluation.first_goal_verified_step == 2
    assert evaluation.stop_reason == "goal_complete"
    assert evaluation.notes == (
        "action_rejected",
        "execution_failed",
        "goal_verified",
    )


def test_episode_evaluation_artifact_is_derived_from_persisted_trajectory(tmp_path):
    trajectory = _example_trajectory()
    trajectory_path = tmp_path / "trajectory.jsonl"
    _write_trajectory(trajectory_path, trajectory)

    written = write_episode_evaluation(tmp_path / "evaluation.json", trajectory_path)
    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    restored = TrajectoryRecorder.read_jsonl(trajectory_path)

    assert set(payload) == {
        "schema_version",
        "trajectory_file",
        "trajectory_sha256",
        "summary",
    }
    assert payload["schema_version"] == "aster-episode-evaluation-0"
    assert payload["trajectory_file"] == "trajectory.jsonl"
    assert payload["trajectory_sha256"] == hashlib.sha256(
        trajectory_path.read_bytes()
    ).hexdigest()
    assert set(payload["summary"]) == SUMMARY_FIELDS
    assert payload["summary"] == written.to_dict()
    assert written == evaluate_episode(restored)
    assert payload["summary"]["first_goal_verified_step"] == 2


def test_transition_rejects_action_valid_mismatch():
    with pytest.raises(ValueError, match="action_valid"):
        _transition(
            0,
            Action.tool("memory.get", key="total"),
            Observation(accepted=True, ok=True, output={"key": "total", "value": 42}),
            EvaluationResult(
                action_valid=False,
                execution_success=True,
                goal_satisfied=False,
            ),
        )


def test_transition_rejects_execution_success_mismatch():
    with pytest.raises(ValueError, match="execution_success"):
        _transition(
            0,
            Action.tool("memory.get", key="total"),
            Observation(accepted=True, ok=True, output={"key": "total", "value": 42}),
            EvaluationResult(
                action_valid=True,
                execution_success=False,
                goal_satisfied=False,
            ),
        )


def test_transition_rejects_terminal_mismatch():
    with pytest.raises(ValueError, match="terminal"):
        _transition(
            0,
            Action.tool("memory.get", key="total"),
            Observation(accepted=True, ok=True, output={"key": "total", "value": 42}),
            EvaluationResult(
                action_valid=True,
                execution_success=True,
                goal_satisfied=False,
                terminal=True,
            ),
        )


def test_trajectory_rejects_goal_satisfaction_regression():
    verified = _transition(
        0,
        Action.tool("memory.get", key="total"),
        Observation(accepted=True, ok=True, output={"key": "total", "value": 42}),
        EvaluationResult(
            action_valid=True,
            execution_success=True,
            goal_satisfied=True,
        ),
    )
    regressed = _transition(
        1,
        Action.tool("memory.get", key="total"),
        Observation(accepted=True, ok=True, output={"key": "total", "value": 42}),
        EvaluationResult(
            action_valid=True,
            execution_success=True,
            goal_satisfied=False,
        ),
    )

    with pytest.raises(ValueError, match="goal_satisfied"):
        Trajectory((verified, regressed))


def _example_trajectory() -> Trajectory:
    return Trajectory(
        (
            _transition(
                0,
                Action.tool("missing"),
                Observation(
                    accepted=False,
                    ok=False,
                    error=ErrorInfo(code="tool_not_found", message="missing"),
                ),
                EvaluationResult(
                    action_valid=False,
                    execution_success=False,
                    goal_satisfied=False,
                    notes=("action_rejected",),
                ),
            ),
            _transition(
                1,
                Action.tool("calculator", operation="divide", left=1, right=0),
                Observation(
                    accepted=True,
                    ok=False,
                    error=ErrorInfo(code="division_by_zero", message="cannot divide by zero"),
                ),
                EvaluationResult(
                    action_valid=True,
                    execution_success=False,
                    goal_satisfied=False,
                    notes=("execution_failed",),
                ),
            ),
            _transition(
                2,
                Action.tool("memory.get", key="total"),
                Observation(
                    accepted=True,
                    ok=True,
                    output={"key": "total", "value": 42},
                ),
                EvaluationResult(
                    action_valid=True,
                    execution_success=True,
                    goal_satisfied=True,
                    notes=("goal_verified",),
                ),
            ),
            _transition(
                3,
                Action.stop("goal_complete"),
                Observation(
                    accepted=True,
                    ok=True,
                    output={"stopped": True, "reason": "goal_complete"},
                ),
                EvaluationResult(
                    action_valid=True,
                    execution_success=True,
                    goal_satisfied=True,
                    terminal=True,
                    notes=("goal_verified",),
                ),
            ),
        )
    )


def _write_trajectory(path, trajectory: Trajectory) -> None:
    recorder = TrajectoryRecorder()
    for transition in trajectory.transitions:
        recorder.record(transition)
    recorder.write_jsonl(path)


def _transition(step, action, observation, evaluation) -> Transition:
    return Transition(
        step=step,
        state_before={"step": step},
        available_actions=("calculator", "memory.get", "stop"),
        action=action,
        observation=observation,
        state_after={"step": step + 1},
        evaluation=evaluation,
    )
