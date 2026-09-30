"""Budget matching, disjoint reserved families, and shared train-only tokenizer."""
from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.state_diagnostics import measure_state_diagnostics
from aster.benchmark.suite import _executor
from aster.corpus.pipeline import digest, json_bytes
from aster.runtime.context import RuntimeContext
from aster.tokenizer.artifact import load_tokenizer
from aster.training import decision_data as data
from aster.training import decision_fit as fit


def test_equal_slots_different_unique_data_and_reachable_targets():
    suite, arms = data.build_training_intervention()
    assert len(suite.cases) == 128 and not suite.cases_for("dev") and not suite.cases_for("test")
    for arm, expected in zip(arms, (8, 16, 16, 32), strict=True):
        cases = [c for c in suite.cases if c.case_id in arm.case_ids]
        assert len(cases) == 32
        assert len({data.example_signature(c.example) for c in cases}) == expected
        assert len({c.leakage_group for c in cases}) == 2
        for case in cases:
            example = case.example
            context = RuntimeContext(task=example.state.task)
            executor = _executor()
            for transition in example.trajectory.transitions:
                assert context.snapshot(transition.step).to_dict() == transition.state_before
                assert executor.execute(transition.action, context) == transition.observation
                assert context.snapshot(transition.step+1).to_dict() == transition.state_after
            assert context.snapshot(len(example.trajectory.transitions)) == example.state
            assert example.target in example.candidates


def test_reserved_family_is_separate_and_all_variants_keep_test_group():
    train, _ = data.build_training_intervention()
    probes = data.build_development_probes()
    sealed = data.build_sealed_prefix_suite()
    partition = data.validate_partition(train, probes, sealed)
    assert len(probes.cases) == 56 and len(sealed.cases) == 12
    assert len(partition["development_probes"]["ancestry_overlap_with_train"]) == 2
    assert partition["development_probes"]["independent_holdout"] is False
    assert partition["test"]["scored_cases"] == 0
    for case in sealed.cases:
        context = RuntimeContext(task=case.example.state.task)
        executor = _executor()
        for transition in case.example.trajectory.transitions:
            assert context.snapshot(transition.step).to_dict() == transition.state_before
            assert executor.execute(transition.action, context) == transition.observation
        assert context.snapshot(len(case.example.trajectory.transitions)) == case.example.state
    leaked = BenchmarkSuite("bad", tuple(replace(c, leakage_group="train-add-small-descendants") for c in sealed.cases))
    with pytest.raises(ValueError, match="leaks"):
        data.validate_partition(train, probes, leaked)
    visible_duplicate = replace(train.cases[0], split="dev", slice_name="numeric_shift",
                                example=replace(train.cases[0].example, teacher="different-label"))
    with pytest.raises(ValueError, match="appears in training"):
        data.validate_partition(train, BenchmarkSuite("bad-probe", (visible_duplicate,)), sealed)
    with pytest.raises(ValueError, match="dev probes only"):
        measure_state_diagnostics(None, None, suite=sealed)


def test_training_does_not_consume_probes_or_reserved_test_and_freezes_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    original_train = data._run_arm
    original_measure = data.measure_state_diagnostics
    calls = []

    def train_guard(*args, **kwargs):
        cases = args[6]
        assert all(c.split == "train" for c in cases)
        return original_train(*args, **kwargs)

    def measurement_guard(model, tokenizer, **kwargs):
        assert all(c.split == "dev" for c in kwargs["suite"].cases)
        calls.append("dev")
        def forbidden(*args, **kwargs):
            raise AssertionError("dev must not update model")
        with monkeypatch.context() as patch:
            patch.setattr(torch.optim.AdamW, "step", forbidden)
            return original_measure(model, tokenizer, **kwargs)

    monkeypatch.setattr(data, "_run_arm", train_guard)
    monkeypatch.setattr(data, "measure_state_diagnostics", measurement_guard)
    path = data.run_data_intervention(tmp_path, config=fit.DecisionFitStudyConfig(
        epochs=1, width=8, target_vocab_size=320, stability_window=1))
    study = json.loads((path / "study.json").read_text())
    assert calls == ["dev"] * 4
    assert study["partition"]["test"]["scored_cases"] == 0
    summaries = study["arms"]
    assert {s["fit"]["initial_model_sha256"] for s in summaries} == {study["initial_model_sha256"]}
    assert [s["fit"]["example_exposures"] for s in summaries] == [32]*4
    assert [s["fit"]["optimizer_updates"] for s in summaries] == [32]*4
    assert all(s["fit"]["reload_metrics_match"] for s in summaries)
    assert all(s["dev"]["weights_unchanged"] for s in summaries)
    tokenizer = load_tokenizer(path / "tokenizer")
    expected = data.train_aster_tokenizer(fit._serialized_texts(data.anchor_cases()),
        target_vocab_size=320, min_pair_frequency=1, name="AsterDecisionFitTokenizer", version="0")
    assert tokenizer.model.merges == expected.model.merges
    orders = []
    for s in summaries:
        arm_dir = path / "arms" / s["fit"]["arm_id"]
        _, _, manifest = fit.load_decision_fit_checkpoint(arm_dir / "checkpoint")
        assert manifest["tokenizer_sha256"] == study["tokenizer_sha256"]
        rows = [json.loads(line) for line in (arm_dir / "epoch-order.jsonl").read_text().splitlines()]
        ids = s["fit"]["case_ids"]
        orders.append([ids.index(i) for i in rows[0]["case_ids"]])
        rule = [e for e in s["dev"]["episodes"] if e["policy"] == "rule"]
        assert len(rule) == 8 and all(e["success"] for e in rule)
    assert all(order == orders[0] for order in orders)


def test_cached_failure_audit_checks_digests_without_new_inference(tmp_path, monkeypatch):
    from aster.benchmark import decision_failure as audit
    from aster.benchmark import state_diagnostics as diagnostic
    from aster.benchmark.suite import build_calculate_and_store_suite
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    fit_run = fit.run_logged_decision_fit_study(tmp_path, suite,
        config=fit.DecisionFitStudyConfig(epochs=1, width=8, target_vocab_size=320),
        arms=(fit.default_fit_arms(suite)[2],))
    checkpoint = fit_run / "arms/eight-fixed/checkpoint"
    dev_run = diagnostic.run_state_diagnostics(tmp_path, checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached audit must not perform new inference")
    monkeypatch.setattr(diagnostic, "score_candidates", forbidden)
    output = audit.audit_cached_failures(tmp_path, dev_run, checkpoint)
    result = json.loads((output / "audit.json").read_text())
    assert result["model_update"] is result["new_inference"] is False
    assert len((output / "token-cases.jsonl").read_text().splitlines()) == 48
    prediction_path = dev_run / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    rows[0]["input_sha256"][0] = "tampered"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="input digest"):
        audit.audit_cached_failures(tmp_path, dev_run, checkpoint)
