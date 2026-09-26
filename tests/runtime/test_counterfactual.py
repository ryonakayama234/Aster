import json

import pytest

from aster.agent.policy import RuleBasedPolicy
from aster.evaluator.verifier import TaskEvaluator
from aster.records.routing import RoutingTrace
from aster.records.transition import Action
from aster.reward.contract import RewardSpec
from aster.runtime.context import RuntimeContext
from aster.runtime.counterfactual import run_logged_counterfactual_analysis
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.tools.schema import ToolSpec


def _task():
    return {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }


def _executor(*, unsafe_handler=None) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    if unsafe_handler is not None:
        registry.register(
            ToolSpec(name="unsafe", description="test-only side effect"),
            unsafe_handler,
        )
    return ToolExecutor(registry)


def _source_trajectory(executor: ToolExecutor):
    return run_loop(
        policy=RuleBasedPolicy(),
        executor=executor,
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
    )


def test_counterfactual_candidates_produce_branch_rewards_and_advantages(tmp_path):
    executor = _executor()
    source = _source_trajectory(executor)
    candidates = (
        Action.tool("calculator", operation="add", left=40, right=2),
        Action.tool("memory.get", key="total"),
        Action.stop("policy_stop"),
    )
    routing = RoutingTrace(
        step=0,
        model_id="decision-v0",
        route="model",
        temperature=1.0,
        autonomous_threshold=0.8,
        fallback_threshold=0.5,
        candidates=candidates,
        scores=(2.0, 1.0, 0.0),
        raw_probabilities=(0.6, 0.3, 0.1),
        calibrated_probabilities=(0.6, 0.3, 0.1),
        selected_index=0,
        final_action=candidates[0],
    )

    run_path = run_logged_counterfactual_analysis(
        tmp_path,
        source_run_id="source-v0",
        trajectory=source,
        routing=(routing,),
        continuation_policy=RuleBasedPolicy(),
        continuation_policy_id="rule-v0",
        executor=executor,
        evaluator=TaskEvaluator(),
        reward_spec=RewardSpec(
            spec_id="counterfactual-test-v0",
            task_success=1.0,
            failed_execution=-0.2,
            step_cost=-0.01,
        ),
    )

    traces = [
        json.loads(line)
        for line in (run_path / "counterfactuals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((run_path / "counterfactual.json").read_text(encoding="utf-8"))

    assert [trace["status"] for trace in traces] == ["completed"] * 3
    assert [trace["branch_reward_total"] for trace in traces] == pytest.approx(
        [0.96, -0.22, -0.01]
    )
    assert [trace["baseline_reward"] for trace in traces] == pytest.approx([0.509] * 3)
    assert [trace["advantage_estimate"] for trace in traces] == pytest.approx(
        [0.451, -0.729, -0.519]
    )
    assert traces[0]["model_selected"] is True
    assert traces[0]["source_executed"] is True
    assert traces[1]["source_executed"] is False
    assert summary["summary"] == {
        "source_steps": 1,
        "candidate_branches": 3,
        "completed_branches": 3,
        "unsupported_branches": 0,
        "steps_with_advantage": 1,
    }
    for trace in traces:
        branch_dir = run_path / trace["branch_dir"]
        assert (branch_dir / "trajectory.jsonl").exists()
        assert (branch_dir / "evaluation.json").exists()
        assert (branch_dir / "reward.json").exists()


def test_unsafe_tool_is_never_executed_and_disables_step_baseline(tmp_path):
    calls = {"count": 0}

    def unsafe_handler(context, arguments):
        calls["count"] += 1
        return {"executed": True}

    executor = _executor(unsafe_handler=unsafe_handler)
    source = _source_trajectory(executor)
    candidates = (
        Action.tool("unsafe"),
        Action.tool("calculator", operation="add", left=40, right=2),
    )
    routing = RoutingTrace(
        step=0,
        model_id="decision-v0",
        route="model",
        temperature=1.0,
        autonomous_threshold=0.8,
        fallback_threshold=0.5,
        candidates=candidates,
        scores=(0.0, 0.0),
        raw_probabilities=(0.5, 0.5),
        calibrated_probabilities=(0.5, 0.5),
        selected_index=1,
        final_action=candidates[1],
    )

    run_path = run_logged_counterfactual_analysis(
        tmp_path,
        source_run_id="source-unsafe-v0",
        trajectory=source,
        routing=(routing,),
        continuation_policy=RuleBasedPolicy(),
        continuation_policy_id="rule-v0",
        executor=executor,
        evaluator=TaskEvaluator(),
        reward_spec=RewardSpec(spec_id="counterfactual-test-v0", task_success=1.0),
    )

    traces = [
        json.loads(line)
        for line in (run_path / "counterfactuals.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert calls["count"] == 0
    assert traces[0]["status"] == "unsupported_counterfactual_tool"
    assert "not marked counterfactual_safe" in traces[0]["note"]
    assert traces[0]["branch_dir"] is None
    assert traces[0]["advantage_estimate"] is None
    assert traces[1]["status"] == "completed"
    assert traces[1]["advantage_estimate"] is None
