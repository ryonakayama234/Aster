"""Decision benchmark wiring: split integrity, calibration, and observable artifacts."""

from dataclasses import replace
import json
import math

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.metrics import (
    DecisionPrediction,
    apply_temperature,
    fit_temperature,
    summarize_predictions,
)
from aster.benchmark.runner import run_logged_decision_benchmark
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.tokenizer.artifact import train_aster_tokenizer


def test_suite_is_deterministic_and_rejects_cross_split_leakage():
    first = build_calculate_and_store_suite()
    second = build_calculate_and_store_suite()
    assert first.to_dict() == second.to_dict()
    assert first.training_examples()
    assert first.cases_for("calibration")
    assert first.cases_for("dev")
    assert first.cases_for("test")

    permutation = [case for case in first.cases if case.slice_name == "candidate_permutation"]
    assert permutation
    for case in permutation:
        assert case.example.target == case.example.candidates[case.example.target_index]

    train_case = first.cases_for("train")[0]
    test_case = first.cases_for("test")[0]
    leaked = replace(test_case, leakage_group=train_case.leakage_group)
    with pytest.raises(ValueError, match="must not cross benchmark splits"):
        BenchmarkSuite("leaked", (train_case, leaked))


def test_probability_metrics_and_temperature_scaling_are_auditable():
    predictions = (
        DecisionPrediction(
            case_id="a",
            split="calibration",
            slice_name="toy",
            target_index=0,
            predicted_index=0,
            logits=(math.log(4.0), 0.0),
            raw_probabilities=(0.8, 0.2),
        ),
        DecisionPrediction(
            case_id="b",
            split="calibration",
            slice_name="toy",
            target_index=1,
            predicted_index=0,
            logits=(math.log(1.5), 0.0),
            raw_probabilities=(0.6, 0.4),
        ),
    )
    raw = summarize_predictions(predictions, calibrated=False, ece_bins=5)
    assert raw["accuracy"] == 0.5
    assert raw["brier"] == pytest.approx(0.4)
    assert raw["nll"] == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2)
    assert sum(row["count"] for row in raw["reliability"]) == 2

    temperature = fit_temperature(predictions)
    calibrated = apply_temperature(predictions, temperature)
    measured = summarize_predictions(calibrated, calibrated=True, ece_bins=5)
    assert temperature > 0
    assert measured["nll"] <= raw["nll"] + 1e-12
    assert measured["accuracy"] == raw["accuracy"]


def test_logged_benchmark_writes_sites_readable_artifacts(tmp_path):
    torch.manual_seed(9)
    torch.set_num_threads(2)
    full = build_calculate_and_store_suite()
    cases = full.cases_for("calibration")[:2] + full.cases_for("test")[:2]
    suite = BenchmarkSuite("benchmark-test", cases)

    texts = [
        serialize_decision_input(
            case.example.state,
            case.example.trajectory,
            candidate,
        )
        for case in suite.cases
        for candidate in case.example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=512,
        min_pair_frequency=1,
        name="AsterDecisionBenchmarkTestTokenizer",
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

    run_path = run_logged_decision_benchmark(
        tmp_path,
        model,
        tokenizer,
        suite,
        model_id="decision-benchmark-test",
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run_path / "benchmark.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_path / "calibration.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (run_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [
        json.loads(line)
        for line in (run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert run["status"] == "completed"
    assert benchmark["schema_version"] == "aster-decision-benchmark-0"
    assert benchmark["temperature_scaling"]["fit_split"] == "calibration"
    assert set(benchmark["test"]) == {"raw", "calibrated"}
    assert "brier" in benchmark["test"]["raw"]
    assert "risk_coverage" in benchmark["test"]["calibrated"]
    assert calibration["schema_version"] == "aster-decision-calibration-0"
    assert calibration["fit_split"] == "calibration"
    assert len(predictions) == len(cases)
    assert all(event["producer"] == "evaluator" for event in events)
    assert model.training
