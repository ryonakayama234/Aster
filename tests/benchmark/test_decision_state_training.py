import json
from dataclasses import replace
import signal

import pytest

torch = pytest.importorskip("torch")
from aster.training import decision_state_training as experiment
from aster.training.decision_fit import DecisionFitStudyConfig
from aster.benchmark.state_representation import build_suite
from aster.training.decision_data import build_development_probes


def test_reachable_intervention_changes_visible_inputs_and_keeps_distribution():
    trains = experiment.build_training()
    dev, pairs = experiment.build_new_dev()
    original, _ = build_suite()
    audit = experiment.preflight(trains, dev, pairs, original, build_development_probes())
    assert len(audit["changed_slots"]) == 16
    assert audit["irrelevant_delay_is_input_alias"]
    assert audit["target_counts"]["baseline"] == audit["target_counts"]["state"]
    assert audit["input_collisions"] == 0
    primary = [p for p in audit["pairs"] if p["relation"] == "same_step_same_candidates_different_target"]
    assert len(primary) == 8 and sum(p["eligible_nonoverlap"] for p in primary) == 6
    assert sum(audit["overlap"]["new_dev"].values()) == 4
    # No repeated calculator execution: its whole generator family remains reserved.
    for suite in list(trains.values()) + [dev]:
        assert all(sum(t.action.name == "calculator" for t in c.example.trajectory.transitions) <= 1 for c in suite.cases)


def test_preflight_rejects_tampered_prefix_and_sealed_inputs():
    trains = experiment.build_training()
    dev, pairs = experiment.build_new_dev()
    original, _ = build_suite()
    old = build_development_probes()
    case = dev.cases[0]
    broken = replace(case.example, target_index=0 if case.example.target_index != 0 else 1)
    with pytest.raises(ValueError, match="Teacher replay"):
        experiment.preflight(trains, replace(dev, cases=(replace(case, example=broken),) + dev.cases[1:]), pairs, original, old)
    with pytest.raises(ValueError, match="sealed"):
        experiment.preflight(trains, replace(dev, cases=tuple(replace(c, split="test") for c in dev.cases)), pairs, original, old)


def test_debug_run_pairs_initialization_order_reload_and_evaluation(tmp_path):
    torch.set_num_threads(2)
    path = experiment.run_state_training(tmp_path, config=DecisionFitStudyConfig(epochs=1, width=8, stability_window=1))
    study = json.loads((path / "study.json").read_text())
    assert study["status"] == "debug/non-preregistered"
    assert study["same_epoch_orders"] and study["full_initial_weights_identical"]
    for arm in study["arms"]:
        assert arm["fit"]["optimizer_updates"] == 32
        assert arm["fit"]["initial_model_sha256"] == study["initial_model_sha256"]
        assert arm["fit"]["reload_metrics_match"]
        assert arm["new_dev"]["weights_unchanged"] and arm["new_dev"]["rng_unchanged"]
        assert arm["nonoverlap"]["primary_pairs"]["count"] == 6
    assert study["test_scored_cases"] == 0


def test_timeout_records_failure_restores_signal(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    def expire(*args, **kwargs):
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    monkeypatch.setattr(experiment, "_run_arm", expire)
    with pytest.raises(TimeoutError):
        experiment.run_state_training(tmp_path, config=DecisionFitStudyConfig(epochs=1), deadline_seconds=1)
    run = next((tmp_path / "runs").glob("*/run.json"))
    assert json.loads(run.read_text())["status"] == "failed"
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
