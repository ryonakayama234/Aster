"""Intervention-learning wiring: shadow teacher, artifacts, and v+1 candidate update."""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.evaluator.verifier import TaskEvaluator
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import DecisionTrace, serialize_decision_input
from aster.records.intervention import InterventionTrace
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.reward.contract import RewardSpec
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.runtime.loop import run_loop
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import train_aster_tokenizer
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.training.decision import DecisionTrainConfig, examples_from_teacher_trajectory
from aster.training.intervention import (
    decision_example_from_intervention,
    train_intervention_candidate,
)


class FirstCandidateDecisionPolicy:
    model_id = "student-v0"

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


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def task():
    return {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }


def test_intervention_rollout_labels_every_student_visited_state_and_writes_artifacts(tmp_path):
    policy = SelectivePolicy(
        FirstCandidateDecisionPolicy(),
        fallback=RuleBasedPolicy(),
        fallback_id="rule-v0:fallback",
        config=SelectivePolicyConfig(
            autonomous_threshold=0.9,
            fallback_threshold=0.5,
        ),
    )
    reward_spec = RewardSpec(
        spec_id="intervention-test-v0",
        task_success=1.0,
        step_cost=-0.01,
    )
    run_path = run_logged_intervention_agent(
        tmp_path,
        policy=policy,
        teacher=RuleBasedPolicy(),
        teacher_id="rule-v0:teacher",
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task()),
        reward_spec=reward_spec,
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    learning = json.loads((run_path / "learning.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_path / "evaluation.json").read_text(encoding="utf-8"))
    reward = json.loads((run_path / "reward.json").read_text(encoding="utf-8"))
    interventions = [
        InterventionTrace.from_dict(json.loads(line))
        for line in (run_path / "interventions.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert run["status"] == "completed"
    assert run["evaluation"] == "evaluation.json"
    assert run["reward"] == "reward.json"
    assert learning["schema_version"] == "aster-intervention-rollout-0"
    assert learning["evaluation_file"] == "evaluation.json"
    assert learning["reward_file"] == "reward.json"
    assert learning["summary"]["task_success"] is True
    assert learning["summary"]["steps"] == 4
    assert learning["summary"]["trainable_examples"] == 4
    assert learning["summary"]["teacher_disagreements"] == 3
    assert learning["summary"]["route_counts"] == {
        "model": 0,
        "fallback": 4,
        "abstain": 0,
    }
    assert evaluation["schema_version"] == "aster-episode-evaluation-0"
    assert evaluation["trajectory_file"] == "trajectory.jsonl"
    assert evaluation["summary"]["task_success"] is True
    assert evaluation["summary"]["steps"] == 4
    assert evaluation["summary"]["goal_verified"] is True
    assert evaluation["summary"]["first_goal_verified_step"] == 2
    assert reward["schema_version"] == "aster-episode-reward-0"
    assert reward["evaluation_file"] == "evaluation.json"
    assert reward["spec"] == reward_spec.to_dict()
    assert reward["result"]["total"] == pytest.approx(0.96)
    assert [trace.executed_action.name or trace.executed_action.kind for trace in interventions] == [
        "calculator",
        "memory.put",
        "memory.get",
        "stop",
    ]
    assert interventions[0].agrees
    assert all(trace.trainable for trace in interventions)


def test_intervention_without_teacher_candidate_is_observable_but_not_trainable():
    state = RuntimeState(task={}, memory={}, step=0)
    trace = InterventionTrace(
        step=0,
        student_model_id="student",
        state=state,
        trajectory=Trajectory(),
        candidates=(Action.stop("student"),),
        scores=(0.0,),
        raw_probabilities=(1.0,),
        calibrated_probabilities=(1.0,),
        selected_index=0,
        route="model",
        executed_action=Action.stop("student"),
        teacher_id="teacher",
        teacher_action=Action.stop("different"),
        teacher_target_index=None,
    )
    assert not trace.trainable
    with pytest.raises(ValueError, match="absent"):
        decision_example_from_intervention(trace)


def test_intervention_training_creates_candidate_without_mutating_parent():
    teacher_trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task()),
    )
    examples = examples_from_teacher_trajectory(teacher_trajectory, teacher="rule-v0:base")

    texts = [
        serialize_decision_input(example.state, example.trajectory, candidate)
        for example in examples
        for candidate in example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=512,
        min_pair_frequency=1,
        name="InterventionTestTokenizer",
        version="0",
    )
    context_length = max(
        len(tokenizer.encode(text, add_bos=True, add_eos=True)) for text in texts
    )
    torch.manual_seed(11)
    model = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=context_length,
                width=16,
                heads=2,
                layers=1,
            )
        )
    )
    parent_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    traces = tuple(_trace_from_example(index, example) for index, example in enumerate(examples))
    update = train_intervention_candidate(
        model,
        tokenizer,
        (),
        traces,
        config=DecisionTrainConfig(
            steps=180,
            learning_rate=1e-2,
            train_backbone=True,
            seed=11,
        ),
    )

    assert update.intervention_after["nll"] < update.intervention_before["nll"]
    assert update.aggregate_after["accuracy"] == 1.0
    assert update.losses[-1] < update.losses[0]
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, parent_state[name])
    assert any(
        not torch.equal(update.model.state_dict()[name], parent_state[name])
        for name in parent_state
    )


def _trace_from_example(step, example):
    selected_index = (example.target_index + 1) % len(example.candidates)
    scores = tuple(1.0 if index == selected_index else 0.0 for index in range(len(example.candidates)))
    probabilities = _softmax(scores)
    return InterventionTrace(
        step=step,
        student_model_id="student-v0",
        state=RuntimeState(
            task=example.state.task,
            memory=example.state.memory,
            step=step,
        ),
        trajectory=example.trajectory,
        candidates=example.candidates,
        scores=scores,
        raw_probabilities=probabilities,
        calibrated_probabilities=probabilities,
        selected_index=selected_index,
        route="model",
        executed_action=example.candidates[selected_index],
        teacher_id="rule-v0:teacher",
        teacher_action=example.target,
        teacher_target_index=example.target_index,
    )


def _softmax(scores):
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)
