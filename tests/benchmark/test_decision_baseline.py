"""CPU baseline keeps test sealed until a saved artifact is evaluated explicitly."""

import json
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.inference.decide import score_candidates
from aster.model.decision_artifact import load_decision_artifact
from aster.training.decision_baseline import (
    DecisionBaselineConfig,
    run_logged_decision_baseline,
    run_logged_decision_final_test,
)


def test_baseline_seals_test_then_final_test_uses_saved_artifact(tmp_path):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    run_path = run_logged_decision_baseline(
        tmp_path,
        suite,
        config=DecisionBaselineConfig(
            steps=12,
            learning_rate=1e-2,
            seed=31,
            target_vocab_size=320,
            context_length=2048,
            width=8,
            heads=2,
            layers=1,
        ),
        model_id="decision-baseline-test",
        source_git_sha="test-sha",
    )

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    baseline = json.loads((run_path / "baseline.json").read_text(encoding="utf-8"))
    training = json.loads((run_path / "training.json").read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (run_path / "split-manifest.json").read_text(encoding="utf-8")
    )
    development_predictions = [
        json.loads(line)
        for line in (run_path / "development-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert run["status"] == "completed"
    assert run["test_status"] == "sealed"
    assert baseline["schema_version"] == "aster-decision-baseline-0"
    assert baseline["test"]["status"] == "sealed"
    assert baseline["data_policy"] == {
        "model_update": "train",
        "tokenizer_training": "train",
        "temperature_fit": "calibration",
        "design_observation": "dev",
        "test": "sealed",
    }
    assert training["training_split"] == "train"
    assert training["tokenizer_training_split"] == "train"
    assert {row["split"] for row in development_predictions} == {"calibration", "dev"}
    assert split_manifest["splits"]["test"]["cases"] == len(suite.cases_for("test"))
    assert baseline["reload_check"]["max_abs_score_diff"] == 0.0
    assert set(baseline["dev"]["episodes"]["summary"]) == {
        "rule",
        "model",
        "model+fallback",
    }
    assert baseline["dev"]["episodes"]["candidate_builder"]["coverage"] == 1.0
    assert baseline["dev"]["episodes"]["candidate_builder"]["missing"] == []
    assert baseline["resources"]["torch_num_threads"] == 2
    assert baseline["resources"]["train_wall_seconds"] >= 0

    artifact_dir = run_path / "model-artifact"
    final_path = run_logged_decision_final_test(tmp_path, artifact_dir, suite)
    final_run = json.loads((final_path / "run.json").read_text(encoding="utf-8"))
    final_test = json.loads((final_path / "final-test.json").read_text(encoding="utf-8"))
    test_predictions = [
        json.loads(line)
        for line in (final_path / "test-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert final_run["status"] == "completed"
    assert final_test["schema_version"] == "aster-decision-baseline-final-test-0"
    assert final_test["split"] == "test"
    assert final_test["artifact_id"] == baseline["artifact_id"]
    assert {row["split"] for row in test_predictions} == {"test"}
    assert len(test_predictions) == len(suite.cases_for("test"))
    assert set(final_test["episodes"]["summary"]) == {
        "rule",
        "model",
        "model+fallback",
    }
    assert final_test["episodes"]["candidate_builder"]["coverage"] == 1.0


def test_saved_artifact_reproduces_scores_in_separate_python_process(tmp_path):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    run_path = run_logged_decision_baseline(
        tmp_path,
        suite,
        config=DecisionBaselineConfig(
            steps=4,
            learning_rate=1e-2,
            seed=7,
            target_vocab_size=320,
            context_length=2048,
            width=8,
            heads=2,
            layers=1,
        ),
        model_id="decision-separate-process-test",
    )
    artifact_dir = run_path / "model-artifact"
    model, tokenizer, manifest = load_decision_artifact(artifact_dir)
    case = suite.cases_for("dev")[0]
    with torch.no_grad():
        scores = score_candidates(
            model,
            tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu().tolist()
    selected = max(range(len(scores)), key=scores.__getitem__)

    code = f"""
import json
import torch
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.inference.decide import score_candidates
from aster.model.decision_artifact import load_decision_artifact

torch.set_num_threads(2)
model, tokenizer, manifest = load_decision_artifact({str(artifact_dir)!r})
case = build_calculate_and_store_suite().cases_for('dev')[0]
with torch.no_grad():
    scores = score_candidates(model, tokenizer, case.example.state, case.example.trajectory, case.example.candidates).detach().cpu().tolist()
print(json.dumps({{'artifact_id': manifest['artifact_id'], 'scores': scores, 'selected': max(range(len(scores)), key=scores.__getitem__)}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["artifact_id"] == manifest["artifact_id"]
    assert observed["selected"] == selected
    assert observed["scores"] == pytest.approx(scores, rel=0, abs=1e-7)
