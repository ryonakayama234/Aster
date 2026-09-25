import json
import math

import pytest

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import DecisionTrace
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.context import RuntimeContext
from aster.runtime.selective import run_logged_selective_agent
from aster.runtime.state import RuntimeState
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


class FakeDecisionPolicy:
    model_id = "fake-decision-v0"

    def __init__(self, scores=(1.0, 0.0)):
        self.scores = tuple(float(value) for value in scores)
        self.traces: list[DecisionTrace] = []

    def decide(self, state, trajectory, available_actions):
        candidates = (
            Action.stop("model_choice"),
            Action.stop("alternative"),
        )
        probabilities = _softmax(self.scores)
        selected_index = max(range(len(self.scores)), key=self.scores.__getitem__)
        self.traces.append(
            DecisionTrace(
                step=state.step,
                model_id=self.model_id,
                candidates=candidates,
                scores=self.scores,
                probabilities=probabilities,
                selected_index=selected_index,
            )
        )
        return candidates[selected_index]


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def decide(self, state, trajectory, available_actions):
        self.calls += 1
        return self.action


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def test_selective_policy_routes_high_medium_and_low_confidence():
    state = RuntimeState(task={}, memory={}, step=0)
    history = Trajectory()
    actions = ("calculator", "stop")

    high_fallback = FixedPolicy(Action.stop("fallback"))
    high = SelectivePolicy(
        FakeDecisionPolicy(scores=(3.0, 0.0)),
        fallback=high_fallback,
        config=SelectivePolicyConfig(autonomous_threshold=0.8, fallback_threshold=0.6),
    )
    assert high.decide(state, history, actions) == Action.stop("model_choice")
    assert high.routing_traces[-1].route == "model"
    assert high_fallback.calls == 0

    medium_fallback = FixedPolicy(Action.stop("fallback"))
    medium = SelectivePolicy(
        FakeDecisionPolicy(scores=(1.0, 0.0)),
        fallback=medium_fallback,
        config=SelectivePolicyConfig(autonomous_threshold=0.8, fallback_threshold=0.6),
    )
    assert medium.decide(state, history, actions) == Action.stop("fallback")
    assert medium.routing_traces[-1].route == "fallback"
    assert medium_fallback.calls == 1

    low_fallback = FixedPolicy(Action.stop("fallback"))
    low = SelectivePolicy(
        FakeDecisionPolicy(scores=(0.0, 0.0)),
        fallback=low_fallback,
        config=SelectivePolicyConfig(autonomous_threshold=0.8, fallback_threshold=0.6),
    )
    assert low.decide(state, history, actions) == Action.stop("low_confidence")
    assert low.routing_traces[-1].route == "abstain"
    assert low_fallback.calls == 0


def test_temperature_scaling_changes_control_route_without_changing_argmax():
    state = RuntimeState(task={}, memory={}, step=0)
    fallback = FixedPolicy(Action.stop("fallback"))
    policy = SelectivePolicy(
        FakeDecisionPolicy(scores=(1.0, 0.0)),
        fallback=fallback,
        config=SelectivePolicyConfig(
            temperature=0.5,
            autonomous_threshold=0.8,
            fallback_threshold=0.6,
        ),
    )

    action = policy.decide(state, Trajectory(), ("stop",))

    trace = policy.routing_traces[-1]
    assert action == Action.stop("model_choice")
    assert trace.route == "model"
    assert trace.selected_index == 0
    assert trace.raw_probabilities[0] == pytest.approx(0.7310586, rel=1e-6)
    assert trace.calibrated_probabilities[0] == pytest.approx(0.8807971, rel=1e-6)
    assert fallback.calls == 0


def test_selective_policy_config_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="fallback_threshold"):
        SelectivePolicyConfig(autonomous_threshold=0.5, fallback_threshold=0.6)


def test_logged_selective_runtime_falls_back_and_emits_sites_artifacts(tmp_path):
    task = {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }
    policy = SelectivePolicy(
        FakeDecisionPolicy(scores=(1.0, 0.0)),
        fallback=RuleBasedPolicy(),
        config=SelectivePolicyConfig(autonomous_threshold=0.8, fallback_threshold=0.6),
        fallback_id="rule-v0",
    )

    run_path = run_logged_selective_agent(
        tmp_path,
        policy=policy,
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )

    summary = json.loads((run_path / "selective.json").read_text(encoding="utf-8"))
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_path / "evaluation.json").read_text(encoding="utf-8"))
    trajectory_lines = (run_path / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    routing_lines = (run_path / "routing.jsonl").read_text(encoding="utf-8").splitlines()

    assert run["status"] == "completed"
    assert run["evaluation"] == "evaluation.json"
    assert summary["schema_version"] == "aster-selective-rollout-0"
    assert summary["evaluation_file"] == "evaluation.json"
    assert summary["summary"]["task_success"] is True
    assert summary["summary"]["route_counts"] == {
        "model": 0,
        "fallback": 4,
        "abstain": 0,
    }
    assert summary["summary"]["autonomous_coverage"] == 0.0
    assert summary["summary"]["fallback_rate"] == 1.0
    assert evaluation["schema_version"] == "aster-episode-evaluation-0"
    assert evaluation["trajectory_file"] == "trajectory.jsonl"
    assert evaluation["summary"]["task_success"] is True
    assert evaluation["summary"]["steps"] == 4
    assert evaluation["summary"]["goal_verified"] is True
    assert evaluation["summary"]["first_goal_verified_step"] == 2
    assert len(trajectory_lines) == 4
    assert len(routing_lines) == 4

    first_routing = json.loads(routing_lines[0])
    assert first_routing["route"] == "fallback"
    assert first_routing["model_action"] == Action.stop("model_choice").to_dict()
    assert first_routing["final_action"]["name"] == "calculator"


def _softmax(scores):
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
