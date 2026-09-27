import json
import math

import pytest

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import DecisionTrace
from aster.reward.contract import RewardSpec
from aster.runtime.context import RuntimeContext
from aster.runtime.selective import run_logged_selective_agent
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


class FirstCandidateDecisionPolicy:
    model_id = "credit-test-student-v0"

    def __init__(self):
        self.builder = CalculateAndStoreCandidates()
        self.traces: list[DecisionTrace] = []

    def decide(self, state, trajectory, available_actions):
        candidates = self.builder.build(state, trajectory, available_actions)
        scores = tuple(2.0 if index == 0 else 0.0 for index in range(len(candidates)))
        probabilities = _softmax(scores)
        self.traces.append(
            DecisionTrace(
                step=state.step,
                model_id=self.model_id,
                candidates=candidates,
                scores=scores,
                probabilities=probabilities,
                selected_index=0,
            )
        )
        return candidates[0]


def _policy() -> SelectivePolicy:
    return SelectivePolicy(
        FirstCandidateDecisionPolicy(),
        fallback=RuleBasedPolicy(),
        fallback_id="rule-v0",
        config=SelectivePolicyConfig(
            autonomous_threshold=0.99,
            fallback_threshold=0.5,
        ),
    )


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


def test_logged_reward_run_also_emits_conservative_credit(tmp_path):
    run_path = run_logged_selective_agent(
        tmp_path,
        policy=_policy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
        reward_spec=RewardSpec(
            spec_id="credit-runtime-v0",
            task_success=1.0,
            step_cost=-0.01,
        ),
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "selective.json").read_text(encoding="utf-8"))
    credit = json.loads((run_path / "credit.json").read_text(encoding="utf-8"))

    assert run["credit"] == "credit.json"
    assert summary["credit_file"] == "credit.json"
    assert credit["trajectory_file"] == "trajectory.jsonl"
    assert credit["reward_file"] == "reward.json"
    assert credit["result"]["assigned_total"] == pytest.approx(-0.04)
    assert credit["result"]["unassigned_total"] == pytest.approx(1.0)
    assert credit["result"]["reward_total"] == pytest.approx(0.96)


def test_reward_free_run_stays_credit_free(tmp_path):
    run_path = run_logged_selective_agent(
        tmp_path,
        policy=_policy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "selective.json").read_text(encoding="utf-8"))

    assert not (run_path / "credit.json").exists()
    assert "credit" not in run
    assert summary["credit_file"] is None


def _softmax(scores):
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
