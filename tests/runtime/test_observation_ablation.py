"""OBS v0b input-only controls; no held-out capability claims."""
from __future__ import annotations

from json import loads
from types import SimpleNamespace

import pytest

from aster.runtime.observation_ablation import (
    BRANCHES, CONDITIONS, build_probes, preflight, run_probe,
)


def test_observation_interventions_alter_no_other_model_visible_field():
    probes = build_probes()
    assert tuple(probes) == CONDITIONS
    for branch in BRANCHES:
        for index in range(2):
            original = loads(probes["factual"][branch][index])
            for name in CONDITIONS:
                changed = loads(probes[name][branch][index])
                assert original["candidate"] == changed["candidate"]
                assert original["state"] == changed["state"]
                assert original["history"][0]["action"] == changed["history"][0]["action"]
                assert "evaluation" not in changed["history"][0]
                assert len(changed["history"]) == 1


def test_mask_erases_only_observation_and_is_byte_identical_across_branches():
    probes = build_probes()
    for index in range(2):
        a, b = (probes["masked"][branch][index] for branch in BRANCHES)
        assert a == b
        assert loads(a)["history"][0]["observation"] is None
        assert probes["swapped_offline"]["success"][index] == (
            probes["factual"]["transient_failure"][index]
        )
        assert probes["swapped_offline"]["transient_failure"][index] == (
            probes["factual"]["success"][index]
        )


def test_status_and_text_controls_are_distinct_and_explicitly_offline():
    probes = build_probes()
    for branch in BRANCHES:
        for index in range(2):
            factual = probes["factual"][branch][index]
            neutral = probes["neutral_char_length"][branch][index]
            assert len(neutral) == len(factual)
            assert len(neutral.encode("utf-8")) == len(factual.encode("utf-8"))
            status = loads(probes["status_only"][branch][index])["history"][0]["observation"]
            assert status["output"] is None and status["error"] is None
            assert status["ok"] is (branch == "success")
            assert status["accepted"] is True
    for index in range(2):
        obs = loads(probes["message_redacted"]["transient_failure"][index])["history"][0]["observation"]
        assert obs["error"]["code"] == "transient_read_error"
        assert obs["error"]["message"] == "[REDACTED]"
        assert probes["message_redacted"]["success"][index] == probes["factual"]["success"][index]


def test_preflight_audits_each_candidate_and_blocks_overflow():
    class FakeTokenizer:
        def encode(self, text, *, add_bos, add_eos):
            assert add_bos and add_eos
            return [256, *text.encode("utf-8"), 257]

    model = SimpleNamespace(backbone=SimpleNamespace(config=SimpleNamespace(context_length=999999)))
    checked = preflight(model, FakeTokenizer())
    assert checked["status"] == "PASS"
    assert len(checked["inputs"]) == len(CONDITIONS)
    assert all(len(checked["inputs"][name][branch]) == 2 for name in CONDITIONS for branch in BRANCHES)
    assert checked["inputs"]["masked"]["success"] == checked["inputs"]["masked"]["transient_failure"]
    model.backbone.config.context_length = 2
    blocked = preflight(model, FakeTokenizer())
    assert blocked["status"] == "BLOCKED_CONTEXT"
    assert blocked["max_input_length"] > 2


def test_frozen_probe_scoring_and_no_tool_continuation():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_id = 257

        def encode(self, value, *, add_bos, add_eos):
            assert add_bos and add_eos
            return [256, *value.encode("utf-8"), 257]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.backbone = SimpleNamespace(config=SimpleNamespace(context_length=999999))
            self.training = True

        def forward(self, ids, lengths):
            scores = []
            for seq, length in zip(ids, lengths):
                value = loads(bytes(seq[1:int(length.item()) - 1].tolist()).decode("utf-8"))
                obs = value["history"][0]["observation"]
                fail = isinstance(obs, dict) and obs.get("ok") is False
                is_retry = value["candidate"]["kind"] == "tool"
                scores.append(2.0 if fail == is_retry else -2.0)
            return torch.tensor(scores, dtype=torch.float32)

    model = FakeModel()
    was_training = model.training
    initial_weight = model.weight.detach().clone()
    result = run_probe(model, FakeTokenizer())
    assert result["status"] == "DIAGNOSTIC_ONLY"
    assert result["factual_first_decision"] == "REPLICATED"
    assert result["choices"]["factual"]["success"]["selected_index"] == 1
    assert result["choices"]["factual"]["transient_failure"]["selected_index"] == 0
    assert result["choices"]["masked"]["success"] == result["choices"]["masked"]["transient_failure"]
    assert result["control_assertions"]["same_masked_inputs_same_scores"]
    assert result["model_weights_unchanged"] is True
    assert result["training_updates"] == 0
    assert torch.equal(model.weight.detach(), initial_weight)
    assert model.training == was_training

    model.backbone.config.context_length = 1
    blocked = run_probe(model, FakeTokenizer())
    assert blocked["status"] == "BLOCKED_CONTEXT"
    assert "choices" not in blocked
