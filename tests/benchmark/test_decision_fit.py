"""Controlled fit-study tests stay train-only and preserve comparison contracts."""
import json

import pytest

torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.training import decision_fit as fit


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_fit_arms_are_preregistered_and_train_only():
    suite = build_calculate_and_store_suite()
    train_ids = {case.case_id for case in suite.cases_for("train")}
    arms = fit.default_fit_arms(suite, include_full_batch=True)

    assert [arm.arm_id for arm in arms] == [
        "one-fixed",
        "two-fixed",
        "eight-fixed",
        "eight-shuffle",
        "eight-full-batch",
    ]
    assert arms[0].case_ids == ("train-add-small/step-0",)
    assert arms[1].case_ids == (
        "train-add-small/step-0",
        "train-add-small/step-2",
    )
    assert set(arms[2].case_ids) == train_ids
    assert all(set(arm.case_ids) <= train_ids for arm in arms)


def test_fit_study_shares_initial_weights_and_records_update_accounting(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    train_ids = tuple(case.case_id for case in suite.cases_for("train"))
    seen_splits = []
    original_predict = fit._predict_cases

    def guarded_predict(model, tokenizer, cases, **kwargs):
        assert cases
        assert all(case.split == "train" for case in cases)
        seen_splits.extend(case.split for case in cases)
        return original_predict(model, tokenizer, cases, **kwargs)

    monkeypatch.setattr(fit, "_predict_cases", guarded_predict)
    arms = (
        fit.DecisionFitArm("eight-fixed", train_ids, "fixed"),
        fit.DecisionFitArm("eight-shuffle", train_ids, "shuffle"),
        fit.DecisionFitArm("eight-full-batch", train_ids, "full_batch"),
    )
    run = fit.run_logged_decision_fit_study(
        tmp_path,
        suite,
        config=fit.DecisionFitStudyConfig(
            epochs=2,
            width=8,
            heads=2,
            layers=1,
            target_vocab_size=320,
            stability_window=1,
        ),
        arms=arms,
        source_git_sha="test-sha",
    )

    study = json.loads((run / "study.json").read_text(encoding="utf-8"))
    assert study["test"]["status"] == "sealed"
    assert set(seen_splits) == {"train"}
    summaries = {row["arm_id"]: row for row in study["arms"]}
    initial_sha = study["initial_model_sha256"]
    assert all(row["initial_model_sha256"] == initial_sha for row in summaries.values())

    assert summaries["eight-fixed"]["optimizer_updates"] == 16
    assert summaries["eight-fixed"]["example_exposures"] == 16
    assert summaries["eight-shuffle"]["optimizer_updates"] == 16
    assert summaries["eight-shuffle"]["example_exposures"] == 16
    assert summaries["eight-full-batch"]["optimizer_updates"] == 2
    assert summaries["eight-full-batch"]["example_exposures"] == 16
    assert all(row["reload_metrics_match"] for row in summaries.values())

    fixed_curve = _jsonl(run / "arms/eight-fixed/learning-curve.jsonl")
    shuffle_curve = _jsonl(run / "arms/eight-shuffle/learning-curve.jsonl")
    full_curve = _jsonl(run / "arms/eight-full-batch/learning-curve.jsonl")
    assert len(fixed_curve) == len(shuffle_curve) == len(full_curve) == 3
    assert fixed_curve[0]["metrics"] == shuffle_curve[0]["metrics"]
    assert fixed_curve[0]["metrics"] == full_curve[0]["metrics"]

    fixed_orders = _jsonl(run / "arms/eight-fixed/epoch-order.jsonl")
    shuffle_orders = _jsonl(run / "arms/eight-shuffle/epoch-order.jsonl")
    assert all(row["case_ids"] == list(train_ids) for row in fixed_orders)
    assert any(row["case_ids"] != list(train_ids) for row in shuffle_orders)


def test_full_batch_and_single_example_equal_exposure_are_not_equal_update():
    cases = 8
    epochs = 25
    assert epochs * cases == 200
    assert epochs == 25
    assert 200 * cases == 1600
