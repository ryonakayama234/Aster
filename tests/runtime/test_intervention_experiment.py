"""End-to-end intervention experiment: rollout, update, recalibration, held-out comparison."""

import json

import pytest

torch = pytest.importorskip("torch")

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tokenizer.artifact import train_aster_tokenizer
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.training.decision import DecisionTrainConfig, examples_from_teacher_trajectory
from aster.training.experiment import run_logged_intervention_learning_experiment


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


def test_logged_intervention_experiment_closes_v0_to_v1_comparison(tmp_path):
    torch.manual_seed(23)
    torch.set_num_threads(2)

    teacher_trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task()),
    )
    base_examples = examples_from_teacher_trajectory(
        teacher_trajectory,
        teacher="rule-v0:base",
    )

    full_suite = build_calculate_and_store_suite()
    cases = full_suite.cases_for("calibration")[:1] + full_suite.cases_for("test")[:1]
    suite = BenchmarkSuite("intervention-experiment-test", cases)

    all_examples = tuple(base_examples) + tuple(case.example for case in suite.cases)
    texts = [
        serialize_decision_input(example.state, example.trajectory, candidate)
        for example in all_examples
        for candidate in example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=512,
        min_pair_frequency=1,
        name="InterventionExperimentTestTokenizer",
        version="0",
    )
    context_length = max(
        len(tokenizer.encode(text, add_bos=True, add_eos=True)) for text in texts
    )
    model = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=context_length,
                width=8,
                heads=2,
                layers=1,
            )
        )
    )
    parent_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    primary = ModelPolicy(
        model,
        tokenizer,
        candidate_builder=CalculateAndStoreCandidates(),
        model_id="decision-v0-test",
    )
    policy = SelectivePolicy(
        primary,
        fallback=RuleBasedPolicy(),
        fallback_id="rule-v0:fallback",
        config=SelectivePolicyConfig(
            autonomous_threshold=1.0,
            fallback_threshold=0.0,
        ),
    )

    run_path = run_logged_intervention_learning_experiment(
        tmp_path,
        policy=policy,
        teacher=RuleBasedPolicy(),
        teacher_id="rule-v0:teacher",
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task()),
        parent_model=model,
        tokenizer=tokenizer,
        base_examples=base_examples,
        benchmark_suite=suite,
        train_config=DecisionTrainConfig(
            steps=120,
            learning_rate=1e-2,
            train_backbone=True,
            seed=23,
        ),
        candidate_model_id="decision-v1-test",
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    experiment = json.loads((run_path / "experiment.json").read_text(encoding="utf-8"))
    training = json.loads((run_path / "training.json").read_text(encoding="utf-8"))
    parent_benchmark = json.loads(
        (run_path / "benchmark-parent" / "benchmark.json").read_text(encoding="utf-8")
    )
    candidate_benchmark = json.loads(
        (run_path / "benchmark-candidate" / "benchmark.json").read_text(encoding="utf-8")
    )
    parent_calibration = json.loads(
        (run_path / "benchmark-parent" / "calibration.json").read_text(encoding="utf-8")
    )
    candidate_calibration = json.loads(
        (run_path / "benchmark-candidate" / "calibration.json").read_text(encoding="utf-8")
    )

    assert run["status"] == "completed"
    assert experiment["schema_version"] == "aster-intervention-learning-experiment-0"
    assert experiment["promotion"] == "candidate_only"
    assert experiment["parent_model_id"] == "decision-v0-test"
    assert experiment["candidate_model_id"] == "decision-v1-test"
    assert training["rollout_run_id"] == experiment["rollout_run_id"]
    assert training["update"]["intervention_examples"] == 4
    assert training["update"]["training_examples"] == 8
    assert training["update"]["intervention_after"]["nll"] < training["update"]["intervention_before"]["nll"]
    assert len((run_path / "training-examples.jsonl").read_text(encoding="utf-8").splitlines()) == 8

    rollout_run = json.loads(
        (tmp_path / "runs" / experiment["rollout_run_id"] / "run.json").read_text(encoding="utf-8")
    )
    assert rollout_run["status"] == "completed"
    assert rollout_run["kind"] == "intervention_agent"

    assert parent_benchmark["model_id"] == "decision-v0-test"
    assert candidate_benchmark["model_id"] == "decision-v1-test"
    assert parent_calibration["model_id"] == "decision-v0-test"
    assert candidate_calibration["model_id"] == "decision-v1-test"
    assert parent_calibration["temperature"] > 0
    assert candidate_calibration["temperature"] > 0

    raw_comparison = experiment["comparison"]["test"]["raw"]
    assert raw_comparison["delta"]["accuracy"] == pytest.approx(
        raw_comparison["candidate"]["accuracy"] - raw_comparison["parent"]["accuracy"]
    )
    assert experiment["comparison"]["delta_definition"] == "candidate_minus_parent"

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, parent_state[name])
