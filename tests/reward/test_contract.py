import json

import pytest

from aster.evaluator.episode import EpisodeEvaluation
from aster.reward.contract import RewardSpec, compute_reward, write_reward_result


def successful_evaluation() -> EpisodeEvaluation:
    return EpisodeEvaluation(
        task_success=True,
        terminal_reached=True,
        steps=4,
        accepted_actions=4,
        rejected_actions=0,
        successful_executions=3,
        failed_executions=1,
        goal_verified=True,
        first_goal_verified_step=2,
        stop_reason="goal_complete",
        notes=("goal_verified",),
    )


def test_reward_components_explain_scalar_total():
    evaluation = successful_evaluation()
    spec = RewardSpec(
        spec_id="balanced-v0",
        task_success=1.0,
        failed_execution=-0.25,
        step_cost=-0.01,
    )

    result = compute_reward(evaluation, spec)

    assert result.total == pytest.approx(0.71)
    assert [component.name for component in result.components] == [
        "task_success",
        "task_failure",
        "rejected_action",
        "failed_execution",
        "step_cost",
    ]
    assert sum(component.amount for component in result.components) == pytest.approx(result.total)
    assert result.components[3].quantity == 1
    assert result.components[3].amount == pytest.approx(-0.25)
    assert result.components[4].quantity == 4
    assert result.components[4].amount == pytest.approx(-0.04)


def test_same_evaluation_can_be_reinterpreted_by_different_specs():
    evaluation = successful_evaluation()
    before = evaluation.to_dict()

    success_only = compute_reward(
        evaluation,
        RewardSpec(spec_id="success-only-v0"),
    )
    efficiency = compute_reward(
        evaluation,
        RewardSpec(
            spec_id="efficiency-v0",
            failed_execution=-0.25,
            step_cost=-0.01,
        ),
    )

    assert success_only.total == pytest.approx(1.0)
    assert efficiency.total == pytest.approx(0.71)
    assert evaluation.to_dict() == before


def test_reward_spec_rejects_non_finite_weights():
    with pytest.raises(ValueError, match="finite"):
        RewardSpec(spec_id="bad", step_cost=float("inf"))


def test_reward_artifact_references_evaluation_without_copying_it(tmp_path):
    evaluation = successful_evaluation()
    spec = RewardSpec(
        spec_id="artifact-v0",
        task_success=2.0,
        failed_execution=-0.5,
    )

    result = write_reward_result(tmp_path / "reward.json", evaluation, spec)
    payload = json.loads((tmp_path / "reward.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "aster-episode-reward-0"
    assert payload["trajectory_file"] == "trajectory.jsonl"
    assert payload["evaluation_file"] == "evaluation.json"
    assert payload["spec"] == spec.to_dict()
    assert payload["result"] == result.to_dict()
    assert payload["result"]["total"] == pytest.approx(1.5)
    assert "evaluation" not in payload
