from dataclasses import replace

import pytest

from aster.benchmark.fixed_v1_diagnostics import (
    build_cases, preflight, replay, signature, train_evidence,
)
from aster.records.transition import Action


def test_v1_case_contract_and_same_candidate_counterexamples():
    cases, pairs = build_cases()
    audit = preflight(cases, pairs, set(), {})
    assert audit["rule_success"] == audit["candidate_coverage"] == 128
    assert audit["primary_pairs"] == 32
    assert audit["same_step_primary_pairs"] == 16
    assert audit["candidate_only_oracle_correct"] < 128
    targets = {c["factors"]["mutation"]: c["example"].target.name or "stop"
               for c in cases if c["group"] == "completion"}
    assert targets == {"wrong_overwrite": "memory.put", "same_value_write": "memory.get",
                       "unrelated_write": "stop", "repair_unconfirmed": "memory.get"}


def test_replay_rejects_state_tampering_and_old_contract():
    cases, _ = build_cases()
    example = cases[1]["example"]
    tampered = replace(example, state=replace(example.state, memory={"total": -123}))
    with pytest.raises(ValueError, match="replay mismatch"):
        replay(tampered)
    from aster.benchmark.state_representation import make_example
    old = make_example(example.state.task, 1, 0)
    with pytest.raises(ValueError, match="prefix replay mismatch"):
        replay(old)


def test_old_new_input_teacher_collision_is_rejected():
    cases, pairs = build_cases()
    sig = signature(cases[0]["example"])
    from aster.benchmark.state_representation import canonical
    with pytest.raises(ValueError, match="different old/new teacher"):
        preflight(cases, pairs, {sig}, {sig: {canonical(Action.stop().to_dict())}})


def test_source_checkpoint_set_must_be_complete(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError, match="exactly six"):
        train_evidence(tmp_path)


def test_failed_tool_single_stop_candidate_has_no_competitor_margin(monkeypatch):
    from types import SimpleNamespace
    import torch
    from aster.benchmark import fixed_v1_diagnostics as module
    from aster.training.decision_tokenizer import build_byte_decision_tokenizer
    session = module.Session(dict(kind="calculate_and_store",operation="add",left=2,right=3,store_as="total"))
    session.execute(Action.tool("memory.get",key="missing"))
    example = session.example()
    assert len(example.candidates) == 1
    monkeypatch.setattr(module,"score_candidates",lambda *a,**k:torch.tensor([0.0]))
    model=SimpleNamespace(backbone=SimpleNamespace(config=SimpleNamespace(context_length=2048)))
    result=module.score(model,build_byte_decision_tokenizer(),example)
    assert result["correct"] and result["margin"] is None
