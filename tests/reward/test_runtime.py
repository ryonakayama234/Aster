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
from aster.runtime.learning import run_logged_intervention_agent
from aster.runtime.selective import run_logged_selective_agent
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


class FirstCandidateDecisionPolicy:
    model_id = "reward-test-student-v0"

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


def _spec() -> RewardSpec:
    return RewardSpec(
        spec_id="runtime-test-v0",
        task_success=1.0,
        failed_execution=-0.25,
        step_cost=-0.01,
    )


def test_selective_runtime_emits_reward_only_when_requested(tmp_path):
    run_path = run_logged_selective_agent(
        tmp_path,
        policy=_policy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
        reward_spec=_spec(),
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "selective.json").read_text(encoding="utf-8"))
    reward = json.loads((run_path / "reward.json").read_text(encoding="utf-8"))

    assert run["reward"] == "reward.json"
    assert summary["reward_file"] == "reward.json"
    assert reward["evaluation_file"] == "evaluation.json"
    assert reward["result"]["total"] == pytest.approx(0.96)

    no_reward_path = run_logged_selective_agent(
        tmp_path,
        policy=_policy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
    )
    no_reward_run = json.loads((no_reward_path / "run.json").read_text(encoding="utf-8"))
    no_reward_summary = json.loads(
        (no_reward_path / "selective.json").read_text(encoding="utf-8")
    )

    assert not (no_reward_path / "reward.json").exists()
    assert "reward" not in no_reward_run
    assert no_reward_summary["reward_file"] is None


def test_intervention_runtime_uses_same_reward_contract(tmp_path):
    run_path = run_logged_intervention_agent(
        tmp_path,
        policy=_policy(),
        teacher=RuleBasedPolicy(),
        teacher_id="rule-v0:teacher",
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=_task()),
        reward_spec=_spec(),
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    learning = json.loads((run_path / "learning.json").read_text(encoding="utf-8"))
    reward = json.loads((run_path / "reward.json").read_text(encoding="utf-8"))

    assert run["reward"] == "reward.json"
    assert learning["reward_file"] == "reward.json"
    assert reward["result"]["total"] == pytest.approx(0.96)


def _softmax(scores):
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
