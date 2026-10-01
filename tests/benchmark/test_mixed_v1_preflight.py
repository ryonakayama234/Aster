from dataclasses import replace

import pytest

from aster.benchmark.fixed_v1_diagnostics import build_cases
from aster.benchmark.mixed_v1_preflight import ARMS, audit, build_training, run_preflight
from aster.records.transition import Action


def test_executed_train_support_and_partition():
    train, slots = build_training()
    dev, pairs = build_cases()
    result = audit(train, slots, dev, pairs)
    assert result["independent_terminal_checks"] == 176
    assert result["train_unique_inputs"] == 48
    assert result["arms"][ARMS[0]]["normal_slots"] == 64
    assert result["arms"][ARMS[1]]["normal_slots"] == 48
    assert result["dev"]["train_overlap"] > 0
    assert result["dev_nonoverlap_pairs"] < 32
    assert result["train_pairs"]
    assert result["max_candidate_tokens"] <= 2048


def test_tampered_state_is_rejected():
    train, slots = build_training()
    example = train[1]["example"]
    train[1]["example"] = replace(example, state=replace(example.state, memory={"total": -123}))
    with pytest.raises(ValueError, match="replay mismatch"):
        audit(train, slots, *build_cases())


def test_wrong_teacher_or_action_frequency_is_rejected():
    train, slots = build_training()
    slots[ARMS[1]][0] = slots[ARMS[1]][1]
    with pytest.raises(ValueError, match="support mismatch|frequency mismatch"):
        audit(train, slots, *build_cases())
    train, slots = build_training()
    example = train[0]["example"]
    train[0]["example"] = replace(example, candidates=(Action.stop(),), target_index=0)
    with pytest.raises(ValueError, match="replay mismatch"):
        audit(train, slots, *build_cases())


def test_context_failure_and_successful_saved_run(tmp_path):
    train, slots = build_training()
    with pytest.raises(ValueError, match="context failure"):
        audit(train, slots, *build_cases(), context_length=10)
    path = run_preflight(tmp_path)
    import json
    assert json.loads((path/"run.json").read_text())["status"] == "completed"
    result = json.loads((path/"summary.json").read_text())
    assert result["model_scored"] == result["updates"] == result["test_scored"] == 0
