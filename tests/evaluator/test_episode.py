import json

from aster.evaluator.episode import evaluate_episode, write_episode_evaluation
from aster.evaluator.result import EvaluationResult
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, ErrorInfo, Observation, Transition


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


def test_episode_evaluation_artifact_points_back_to_trajectory(tmp_path):
    trajectory = _example_trajectory()

    written = write_episode_evaluation(tmp_path / "evaluation.json", trajectory)
    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "aster-episode-evaluation-0"
    assert payload["trajectory_file"] == "trajectory.jsonl"
    assert payload["summary"] == written.to_dict()
    assert payload["summary"]["first_goal_verified_step"] == 2
    assert "reward" not in payload


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
