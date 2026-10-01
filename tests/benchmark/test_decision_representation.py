import json
from dataclasses import replace
import signal

import pytest

torch = pytest.importorskip("torch")
from aster.training import decision_representation as comparison
from aster.training.decision_fit import DecisionFitStudyConfig, load_decision_fit_checkpoint
from aster.training.decision_tokenizer import build_byte_decision_tokenizer, build_comparison_training_suite
from aster.benchmark.state_representation import build_suite, compact_input
from aster.inference.decide import encode_candidate_batch


def test_serializer_reaches_candidate_encoding_and_no_truncation():
    suite,_ = build_suite()
    ex=suite.cases[3].example
    tokenizer=build_byte_decision_tokenizer()
    model=comparison._new_model(tokenizer, DecisionFitStudyConfig())
    tokens,lengths=encode_candidate_batch(model,tokenizer,ex.state,ex.trajectory,ex.candidates,serializer=compact_input)
    for i,candidate in enumerate(ex.candidates):
        assert tokenizer.decode(tokens[i,:lengths[i]].tolist())==compact_input(ex.state,ex.trajectory,candidate)
    with pytest.raises(ValueError,match="exceeds"):
        encode_candidate_batch(model,tokenizer,ex.state,ex.trajectory,ex.candidates,serializer=lambda *_:"x"*2049)


def test_audit_rejects_sealed_inputs_and_teacher_collisions():
    suite,_=build_suite()
    tokenizer=build_byte_decision_tokenizer()
    with pytest.raises(ValueError,match="sealed"):
        comparison.audit_inputs([replace(suite.cases[0],split="test")],tokenizer,compact_input,2048)
    with pytest.raises(ValueError,match="conflicting"):
        comparison.audit_inputs(suite.cases,tokenizer,lambda *_:"same",2048)


def test_lookup_is_train_only_and_unseen_abstains():
    train,_=build_comparison_training_suite(); dev,_=build_suite()
    result=comparison.candidate_lookup(train,dev)
    assert result["fit_cases"]==32 and result["covered"]<48
    for case in dev.cases:
        if case.slice_name=="joint":
            assert not result["predictions"][case.case_id]["covered"]
    with pytest.raises(ValueError,match="train only"):
        comparison.candidate_lookup(dev,dev)


def test_debug_paired_reload_and_serializer_identity(tmp_path):
    torch.set_num_threads(2)
    config=DecisionFitStudyConfig(epochs=1,learning_rate=1e-3,width=8,stability_window=1)
    path=comparison.run_comparison(tmp_path,config=config,source_git_sha="test")
    study=json.loads((path/"study.json").read_text())
    assert study["status"]=="debug/non-preregistered"
    assert study["full_initial_weights_identical"] and study["same_epoch_orders"]
    assert study["test_scored_cases"]==0
    assert {a["fit"]["initial_model_sha256"] for a in study["arms"]}=={study["initial_model_sha256"]}
    assert all(a["fit"]["optimizer_updates"]==32 and a["fit"]["reload_metrics_match"] for a in study["arms"])
    assert all(a["dev"]["weights_unchanged"] and a["dev"]["rng_unchanged"] for a in study["arms"])
    with pytest.raises(ValueError,match="serializer mismatch"):
        load_decision_fit_checkpoint(path/"arms/compact/checkpoint")
    load_decision_fit_checkpoint(path/"arms/compact/checkpoint",expected_input_serializer_id="aster-decision-compact-input-0")
    episodes=[json.loads(s) for s in (path/"arms/compact/prefix-episodes.jsonl").read_text().splitlines()]
    for episode in episodes:
        assert [t["step"] for t in episode["trajectory"]]==list(range(len(episode["trajectory"])))
        assert not episode["fallback"]


def test_timeout_records_failed_run_and_restores_handler(tmp_path,monkeypatch):
    previous=signal.getsignal(signal.SIGALRM)
    def expire(*a,**kw):
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM,None)
    monkeypatch.setattr(comparison,"_run_arm",expire)
    with pytest.raises(TimeoutError):
        comparison.run_comparison(tmp_path,config=DecisionFitStudyConfig(epochs=1),deadline_seconds=1)
    runs=list((tmp_path/"runs").glob("*/run.json"))
    assert len(runs)==1 and json.loads(runs[0].read_text())["status"]=="failed"
    assert signal.getsignal(signal.SIGALRM)==previous and signal.getitimer(signal.ITIMER_REAL)[0]==0
