"""AsterDecision-v0 wiring tests: records, learning, and Agent Kernel integration."""

import pytest

torch = pytest.importorskip("torch")

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy, score_candidates
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tokenizer.artifact import train_aster_tokenizer
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.training.decision import (
    DecisionTrainConfig,
    evaluate_decisions,
    examples_from_teacher_trajectory,
    train_decision,
)


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def build_teacher_trajectory():
    task = {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )
    return task, trajectory


def build_teacher_examples():
    task, trajectory = build_teacher_trajectory()
    examples = examples_from_teacher_trajectory(trajectory)
    return task, trajectory, examples


def build_decision_model(examples):
    texts = [
        serialize_decision_input(example.state, example.trajectory, candidate)
        for example in examples
        for candidate in example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=512,
        min_pair_frequency=1,
        name="AsterDecisionTestTokenizer",
        version="0",
    )
    max_length = max(
        len(tokenizer.encode(text, add_bos=True, add_eos=True))
        for text in texts
    )
    model = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=max_length,
                width=16,
                heads=2,
                layers=1,
            )
        )
    )
    return model, tokenizer


def test_tinylm_encode_is_the_shared_representation():
    torch.manual_seed(7)
    model = TinyLM(ModelConfig(vocab_size=258, context_length=16, width=16, heads=2, layers=1))
    token_ids = torch.tensor([[256, 65, 66, 257]])
    with torch.no_grad():
        hidden = model.encode(token_ids)
        torch.testing.assert_close(model(token_ids), model.lm_head(hidden))
    assert hidden.shape == (1, 4, 16)


def test_teacher_trajectory_becomes_separate_decision_examples():
    _, trajectory, examples = build_teacher_examples()
    assert [example.target.name or example.target.kind for example in examples] == [
        "calculator",
        "memory.put",
        "memory.get",
        "stop",
    ]
    assert len(examples) == len(trajectory.transitions) == 4
    assert examples[-1].target.arguments == {"reason": "goal_verified"}
    assert any(
        candidate.kind == "stop" and candidate.arguments == {"reason": "policy_stop"}
        for candidate in examples[-1].candidates
    )
    restored = DecisionExample.from_dict(examples[2].to_dict())
    assert restored.to_dict() == examples[2].to_dict()


@pytest.mark.parametrize("bad_step", ["0", 0.0, False])
def test_teacher_examples_reject_coerced_runtime_steps(bad_step):
    _, trajectory = build_teacher_trajectory()
    trajectory.transitions[0].state_before["step"] = bad_step

    with pytest.raises(ValueError, match="Runtime state step must be an integer"):
        examples_from_teacher_trajectory(trajectory)


def test_candidate_scoring_is_equivariant_to_candidate_order():
    _, _, examples = build_teacher_examples()
    model, tokenizer = build_decision_model(examples)
    example = examples[1]
    candidates = example.candidates
    reversed_candidates = tuple(reversed(candidates))
    model.eval()
    with torch.no_grad():
        forward = score_candidates(
            model, tokenizer, example.state, example.trajectory, candidates
        )
        backward = score_candidates(
            model, tokenizer, example.state, example.trajectory, reversed_candidates
        )
    torch.testing.assert_close(forward, torch.flip(backward, dims=(0,)))


def test_decision_training_overfits_teacher_and_runs_same_kernel():
    torch.manual_seed(3)
    torch.set_num_threads(2)
    task, _, examples = build_teacher_examples()
    model, tokenizer = build_decision_model(examples)
    before = evaluate_decisions(model, tokenizer, examples)
    losses = train_decision(
        model,
        tokenizer,
        examples,
        DecisionTrainConfig(steps=240, learning_rate=1e-2, train_backbone=True, seed=3),
    )
    after = evaluate_decisions(model, tokenizer, examples)

    assert losses[-1] < losses[0]
    assert after["nll"] < before["nll"]
    assert after["accuracy"] == 1.0

    policy = ModelPolicy(
        model,
        tokenizer,
        candidate_builder=CalculateAndStoreCandidates(),
        model_id="decision-test",
    )
    trajectory = run_loop(
        policy=policy,
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )

    assert trajectory.successful
    assert [transition.action.name or transition.action.kind for transition in trajectory.transitions] == [
        "calculator",
        "memory.put",
        "memory.get",
        "stop",
    ]
    assert len(policy.traces) == 4
    for trace, transition in zip(policy.traces, trajectory.transitions):
        assert trace.selected == transition.action
        assert sum(trace.probabilities) == pytest.approx(1.0, abs=1e-6)
