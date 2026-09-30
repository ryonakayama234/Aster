"""Tokenizer-comparison protocol, shared initialization, and isolation tests."""
import json

import pytest

torch = pytest.importorskip("torch")

from aster.tokenizer import byte as raw_byte
from aster.tokenizer.artifact import save_tokenizer, train_aster_tokenizer
from aster.training import decision_tokenizer as comparison
from aster.training.decision_data import anchor_cases
from aster.training.decision_fit import (
    DecisionFitStudyConfig,
    _directory_digest,
    _serialized_texts,
)


def _fixture_bpe(path):
    tokenizer = train_aster_tokenizer(
        _serialized_texts(anchor_cases()),
        target_vocab_size=320,
        min_pair_frequency=1,
        name="AsterDecisionFitTokenizer",
        version="0",
    )
    save_tokenizer(tokenizer, path)
    return tokenizer, _directory_digest(path)


def test_byte_decision_tokenizer_matches_raw_byte_codec_and_roundtrips():
    tokenizer = comparison.build_byte_decision_tokenizer()
    text = '日本語 / {"left": 6, "right": 10}'
    raw = raw_byte.encode(text)
    assert tokenizer.encode(text) == raw
    assert tokenizer.decode(raw) == raw_byte.decode(raw) == text
    assert tokenizer.vocab_size == 258
    assert tokenizer.bos_id == 256 and tokenizer.eos_id == 257
    assert tokenizer.manifest.tokenizer_type == "byte"


def test_comparison_suite_is_exact_32_unique_numbers_and_keys_decisions():
    suite, case_ids = comparison.build_comparison_training_suite()
    assert len(suite.cases) == len(case_ids) == 32
    assert len(set(case_ids)) == 32
    assert all(case.split == "train" for case in suite.cases)
    assert all(case.case_id.startswith("numbers-and-keys/") for case in suite.cases)


def test_shared_initialization_copies_common_tensors_and_semantic_rows(tmp_path):
    bpe, _ = _fixture_bpe(tmp_path / "bpe")
    byte = comparison.build_byte_decision_tokenizer()
    config = DecisionFitStudyConfig(
        epochs=1,
        width=8,
        heads=2,
        layers=1,
        target_vocab_size=320,
        min_pair_frequency=1,
    )
    bpe_state, byte_state, manifest = comparison._shared_initial_states(
        bpe, byte, config
    )
    assert manifest["full_initial_weights_identical"] is False
    assert "head.projection.weight" in manifest["full_shared_parameter_names"]
    assert torch.equal(
        bpe_state["head.projection.weight"], byte_state["head.projection.weight"]
    )
    assert torch.equal(
        bpe_state["backbone.token_embedding.weight"][:256],
        byte_state["backbone.token_embedding.weight"][:256],
    )
    assert torch.equal(
        bpe_state["backbone.token_embedding.weight"][bpe.bos_id],
        byte_state["backbone.token_embedding.weight"][byte.bos_id],
    )
    assert bpe_state["backbone.token_embedding.weight"].shape != (
        byte_state["backbone.token_embedding.weight"].shape
    )


def test_one_epoch_debug_run_keeps_reserved_test_sealed_and_orders_paired(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    _, bpe_sha = _fixture_bpe(tmp_path / "saved-bpe")
    original_audit = comparison._audit_tokenizer
    audited_splits = []

    def guarded_audit(tokenizer, cases, *, context_length):
        audited_splits.extend(case.split for case in cases)
        assert all(case.split != "test" for case in cases)
        return original_audit(tokenizer, cases, context_length=context_length)

    monkeypatch.setattr(comparison, "_audit_tokenizer", guarded_audit)
    config = DecisionFitStudyConfig(
        epochs=1,
        learning_rate=1e-3,
        seed=42,
        target_vocab_size=320,
        min_pair_frequency=1,
        context_length=2048,
        width=8,
        heads=2,
        layers=1,
        stability_window=1,
    )
    run = comparison.run_tokenizer_comparison(
        tmp_path,
        tmp_path / "saved-bpe",
        config=config,
        source_git_sha="test",
        expected_bpe_sha256=bpe_sha,
    )
    study = json.loads((run / "study.json").read_text(encoding="utf-8"))
    assert study["protocol_status"] == "debug/non-preregistered"
    assert study["test"]["scored_cases"] == 0
    assert study["partition"]["test"]["scored_cases"] == 0
    assert set(audited_splits) <= {"train", "dev"}
    assert study["same_epoch_orders"] is True
    assert {arm["arm_id"] for arm in study["arms"]} == {"bpe", "byte"}
    assert all(arm["fit"]["optimizer_updates"] == 32 for arm in study["arms"])
    assert all(arm["fit"]["example_exposures"] == 32 for arm in study["arms"])
    assert all(arm["fit"]["reload_metrics_match"] for arm in study["arms"])
    assert all(arm["dev"]["weights_unchanged"] for arm in study["arms"])
    assert all(
        arm["dev"]["detailed_episode_weights_unchanged"] for arm in study["arms"]
    )
    bpe_arm = next(arm for arm in study["arms"] if arm["arm_id"] == "bpe")
    byte_arm = next(arm for arm in study["arms"] if arm["arm_id"] == "byte")
    assert bpe_arm["token_budget"]["vocab_size"] != byte_arm["token_budget"]["vocab_size"]
    assert (
        bpe_arm["parameter_counts"]["total_parameters"]
        != byte_arm["parameter_counts"]["total_parameters"]
    )
