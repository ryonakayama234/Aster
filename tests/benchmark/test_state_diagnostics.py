"""Reachability, shortcut counterexamples, and frozen checkpoint contracts."""
import json

import pytest

torch = pytest.importorskip("torch")

from aster.agent.policy import RuleBasedPolicy
from aster.benchmark import state_diagnostics as diagnostic
from aster.benchmark.suite import _executor, build_calculate_and_store_suite
from aster.records.trajectory import Trajectory
from aster.runtime.context import RuntimeContext
from aster.training import decision_fit as fit


def test_probe_histories_replay_through_real_executor_and_keep_teacher_coverage():
    suite = diagnostic.build_state_diagnostic_suite()
    assert len(suite.cases) == 48
    assert not suite.cases_for("train") and not suite.cases_for("test")
    assert len({case.leakage_group for case in suite.cases}) == 2
    assert len({case.case_id for case in suite.cases}) == 48
    for case in suite.cases:
        example = case.example
        context = RuntimeContext(task=example.state.task)
        executor = _executor()
        for transition in example.trajectory.transitions:
            assert context.snapshot(transition.step).to_dict() == transition.state_before
            actual = executor.execute(transition.action, context)
            assert actual == transition.observation
            assert context.snapshot(transition.step + 1).to_dict() == transition.state_after
        assert context.snapshot(len(example.trajectory.transitions)) == example.state
        assert example.target in example.candidates
        target = RuleBasedPolicy().decide(example.state, example.trajectory,
                                         executor.registry.names() + ("stop",))
        assert target == example.target


def test_same_step_different_targets_and_same_target_different_steps():
    cases = {c.case_id: c for c in diagnostic.build_state_diagnostic_suite().cases}
    delayed_calculate = cases["add/seen/phase-0/delay-1"].example
    normal_put = cases["add/seen/phase-1/delay-0"].example
    delayed_put = cases["add/seen/phase-1/delay-1"].example
    assert delayed_calculate.state.step == normal_put.state.step == 1
    assert delayed_calculate.target.name == "calculator"
    assert normal_put.target.name == delayed_put.target.name == "memory.put"
    assert delayed_put.state.step == 2
    lookup = diagnostic.fit_step_lookup(build_calculate_and_store_suite().cases_for("train"))
    assert lookup[1] == "memory.put"  # Fails the reachable delayed-calculation case.
    with pytest.raises(ValueError, match="only fit on train"):
        diagnostic.fit_step_lookup(tuple(cases.values()))


def test_diagnostics_do_not_train_and_persist_independent_episodes(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    arm = fit.default_fit_arms(suite)[2]
    run = fit.run_logged_decision_fit_study(tmp_path, suite,
        config=fit.DecisionFitStudyConfig(epochs=1, width=8, target_vocab_size=320), arms=(arm,))
    checkpoint = run / "arms/eight-fixed/checkpoint"
    before = (checkpoint / "model.pt").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics must not train or fit temperature")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(diagnostic, "_task_specs", lambda: (
        ("add", "seen", {"kind": "calculate_and_store", "operation": "add", "left": 2, "right": 3, "store_as": "total"}),
    ))
    output = diagnostic.run_state_diagnostics(tmp_path, checkpoint, source_git_sha="test")
    report = json.loads((output / "diagnostics.json").read_text())
    assert report["weights_unchanged"] is True
    assert report["model_update"] is report["temperature_fit"] is False
    assert report["test"]["status"] == "sealed"
    assert report["overall"]["candidate_coverage"] == 1.0
    assert len(report["permutation_checks"]) == 4
    assert len(report["episodes"]) == 2
    assert next(ep for ep in report["episodes"] if ep["policy"] == "rule")["success"]
    assert (checkpoint / "model.pt").read_bytes() == before
    run_record = json.loads((output / "run.json").read_text())
    assert run_record["status"] == "completed"
    # Permutation evaluation is equal up to independent row arithmetic, not an
    # ability to handle new semantics. Every reversed case keeps the same target.
    saved = json.loads((output / "diagnostic-suite.json").read_text())
    assert all(c["split"] == "dev" for c in saved["cases"])


def test_loader_rejects_single_example_checkpoint(tmp_path):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    run = fit.run_logged_decision_fit_study(tmp_path, suite,
        config=fit.DecisionFitStudyConfig(epochs=1, width=8, target_vocab_size=320),
        arms=(fit.default_fit_arms(suite)[0],))
    with pytest.raises(ValueError, match="eight-example"):
        diagnostic.run_state_diagnostics(tmp_path, run / "arms/one-fixed/checkpoint")
