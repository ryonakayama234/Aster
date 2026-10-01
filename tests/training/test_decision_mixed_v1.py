import json

import pytest
import torch

from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.fixed_v1_diagnostics import build_cases
from aster.benchmark.mixed_v1_preflight import ARMS, audit, build_training
from aster.training.decision_fit import DecisionFitStudyConfig, _new_model
from aster.training.decision_mixed_v1 import load_v1_checkpoint, train_arm, validate_registered
from aster.training.decision_tokenizer import build_byte_decision_tokenizer


def test_registration_rejects_slot_or_case_changes():
    cases, slots = build_training()
    actual = audit(cases, slots, *build_cases())
    original = json.loads(json.dumps(actual))
    validate_registered(actual, original)
    original["arms"][ARMS[0]]["slot_sha256"] = "tampered"
    with pytest.raises(ValueError, match="slots mismatch"):
        validate_registered(actual, original)
    original = json.loads(json.dumps(actual))
    original["case_manifest"][0]["signature"] = "tampered"
    with pytest.raises(ValueError, match="evidence mismatch"):
        validate_registered(actual, original)


def test_unique_epoch_evaluation_and_contract_guard(tmp_path):
    torch.set_num_threads(2)
    raw, _ = build_training()
    # Two different train decisions, each presented twice. Evaluate unique cases once.
    cases = tuple(BenchmarkCase(c["case_id"],"train",c["group"],c["leakage_group"],c["example"]) for c in raw[:2])
    suite = BenchmarkSuite("debug-unique-evaluation",cases)
    slots = [c.case_id for c in cases]*2
    tk=build_byte_decision_tokenizer()
    cfg=DecisionFitStudyConfig(epochs=1,learning_rate=0.001)
    initial={k:v.detach().clone() for k,v in _new_model(tk,cfg).state_dict().items()}
    path=tmp_path/"debug-arm"
    _,_,fit=train_arm(path,suite,slots,tk,initial,cfg,source={"git_sha":"debug"})
    assert fit["updates"]==4 and fit["unique"]==2
    curves=[json.loads(line) for line in (path/"learning-curve.jsonl").read_text().splitlines()]
    assert [r["metrics"]["examples"] for r in curves]==[2,2]
    contract=json.loads((path/"training-contract.json").read_text())
    contract["goal_contract"]="calculate-and-store-v0"
    (path/"training-contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError,match="contract mismatch"):
        load_v1_checkpoint(path/"checkpoint")
