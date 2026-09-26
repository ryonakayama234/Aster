import json

import pytest

from aster.agent.policy import RuleBasedPolicy
from aster.credit.contract import assign_direct_credit, write_credit_result
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.verifier import TaskEvaluator
from aster.records.transition import Action
from aster.reward.contract import RewardComponent, RewardResult, RewardSpec, compute_reward
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


class SequencePolicy:
    def __init__(self, actions):
        self.actions = tuple(actions)

    def decide(self, state, trajectory, available_actions):
        return self.actions[state.step]


def _executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def _task():
    return {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }


def test_success_reward_stays_unassigned_while_step_cost_maps_to_each_step():
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
    )
    reward = compute_reward(
        evaluate_episode(trajectory),
        RewardSpec(spec_id="success-v0", task_success=1.0, step_cost=-0.01),
    )

    credit = assign_direct_credit(trajectory, reward)

    assert credit.reward_total == pytest.approx(0.96)
    assert credit.assigned_total == pytest.approx(-0.04)
    assert credit.unassigned_total == pytest.approx(1.0)
    assert [step.total for step in credit.steps] == pytest.approx([-0.01] * 4)
    assert [item.source_reward_component for item in credit.unassigned] == ["task_success"]
    assert credit.unassigned[0].reason == (
        "episode-level outcome has no step attribution in direct-evidence-v0"
    )


def test_rejected_and_failed_execution_penalties_map_only_to_observed_steps():
    trajectory = run_loop(
        policy=SequencePolicy(
            (
                Action.tool("missing-tool"),
                Action.tool("calculator", operation="divide", left=1, right=0),
                Action.stop("done"),
            )
        ),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
        max_steps=3,
    )
    reward = compute_reward(
        evaluate_episode(trajectory),
        RewardSpec(
            spec_id="failure-v0",
            task_success=0.0,
            task_failure=-1.0,
            rejected_action=-0.1,
            failed_execution=-0.2,
            step_cost=-0.01,
        ),
    )

    credit = assign_direct_credit(trajectory, reward)

    assert reward.total == pytest.approx(-1.33)
    assert [step.total for step in credit.steps] == pytest.approx([-0.11, -0.21, -0.01])
    assert credit.assigned_total == pytest.approx(-0.33)
    assert credit.unassigned_total == pytest.approx(-1.0)
    assert credit.assigned_total + credit.unassigned_total == pytest.approx(reward.total)


def test_credit_rejects_reward_quantities_from_a_different_trajectory():
    trajectory = run_loop(
        policy=SequencePolicy((Action.tool("missing-tool"), Action.stop("done"))),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
        max_steps=2,
    )
    evaluation = evaluate_episode(trajectory)
    reward = compute_reward(
        evaluation,
        RewardSpec(spec_id="mismatch-v0", rejected_action=-0.1),
    )
    components = tuple(
        RewardComponent(component.name, component.weight, 0)
        if component.name == "rejected_action"
        else component
        for component in reward.components
    )
    mismatched = RewardResult(spec_id=reward.spec_id, components=components)

    with pytest.raises(ValueError, match="rejected_action.*does not match"):
        assign_direct_credit(trajectory, mismatched)


def test_credit_artifact_references_inputs_without_copying_them(tmp_path):
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
    )
    reward = compute_reward(
        evaluate_episode(trajectory),
        RewardSpec(spec_id="artifact-v0", task_success=1.0, step_cost=-0.01),
    )

    result = write_credit_result(tmp_path / "credit.json", trajectory, reward)
    payload = json.loads((tmp_path / "credit.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "aster-credit-0"
    assert payload["trajectory_file"] == "trajectory.jsonl"
    assert payload["reward_file"] == "reward.json"
    assert payload["result"] == result.to_dict()
    assert "trajectory" not in payload
    assert "reward" not in payload
