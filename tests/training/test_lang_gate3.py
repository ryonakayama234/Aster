"""Synthetic Gate 3 metric and sealed-test boundary tests; no private data."""
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from aster.corpus.pipeline import digest, json_bytes
from aster.tokenizer.artifact import train_aster_tokenizer
from aster.training.dataset import Window, prepare_windows
from aster.training.lang_gate3 import (
    TokenTotals, preflight, score_model, score_unigram, unigram_counts,
)
from aster.training.tokenizer_run import verify_view


@pytest.fixture()
def tokenizer():
    return train_aster_tokenizer(
        ["猫と犬。日本語のバイト。", "猫と犬。"], target_vocab_size=256
    )


def window(tokenizer, text: str, domain: str = "prose", *, record: str = "a"):
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return Window(tuple(ids), record, domain, 0)


def test_unigram_add_one_fits_only_train_targets(tokenizer):
    train = [window(tokenizer, "猫"), window(tokenizer, "猫")]
    dev = [window(tokenizer, "犬", record="b")]
    counts = unigram_counts(train, tokenizer)
    assert sum(counts) == sum(len(w.ids) - 1 for w in train)
    assert counts[tokenizer.bos_id] == 0
    assert counts[tokenizer.eos_id] == 2
    result, fingerprint = score_unigram(train, dev, tokenizer)
    expected = 0.0
    for token_id in dev[0].ids[1:]:
        expected += math.log(sum(counts) + tokenizer.vocab_size) - math.log(counts[token_id] + 1)
    assert result["nll_sum"] == pytest.approx(expected)
    assert result["targets"] == len(dev[0].ids) - 1
    assert result["text_bytes"] == len("犬".encode("utf-8"))
    assert result["text_targets"] == 3
    assert result["eos_targets"] == 1
    assert result["text_nll_sum"] + result["eos_nll_sum"] == pytest.approx(expected)
    assert result["text_bits_per_byte"] == pytest.approx(
        result["text_nll_sum"] / (3 * math.log(2))
    )
    assert len(fingerprint) == 64
    # Different dev content never changes the fitted frequency table.
    other, same_digest = score_unigram(train, [window(tokenizer, "犬犬")], tokenizer)
    assert other["targets"] != result["targets"]
    assert same_digest == fingerprint


def test_text_byte_count_and_domain_weighting(tokenizer):
    ja = window(tokenizer, "猫", "ja", record="a")
    en = window(tokenizer, "a", "en", record="b")
    scores = TokenTotals()
    scores.add(ja, [1.0] * (len(ja.ids) - 1), tokenizer)
    scores.add(en, [2.0] * (len(en.ids) - 1), tokenizer)
    result = scores.result()
    assert result["text_bytes"] == 4
    assert result["targets"] == 6
    assert result["eos_targets"] == 2
    assert result["text_targets"] == 4
    assert result["nll_sum"] == pytest.approx(4 + 4)
    assert result["nll"] == pytest.approx(8 / 6)
    assert result["by_domain"]["ja"]["text_bytes"] == 3
    assert result["by_domain"]["en"]["text_bytes"] == 1
    assert result["text_nll_sum"] == pytest.approx(5)
    assert result["text_bits_per_byte"] == pytest.approx(5 / (4 * math.log(2)))


def test_score_model_ignores_right_padding(tokenizer):
    class FlatModel(nn.Module):
        def forward(self, x):
            return torch.zeros((*x.shape, tokenizer.vocab_size), dtype=torch.float32)

    long = window(tokenizer, "猫と犬", record="a")
    short = window(tokenizer, "x", record="b")
    together = score_model(FlatModel(), [long, short], tokenizer, batch_size=2)
    separate = score_model(FlatModel(), [long, short], tokenizer, batch_size=1)
    assert together["targets"] == sum(len(w.ids) - 1 for w in (long, short))
    assert together["targets"] < 2 * (len(long.ids) - 1)
    assert together["text_bytes"] == len("猫と犬x".encode("utf-8"))
    assert together["nll"] == pytest.approx(math.log(tokenizer.vocab_size), rel=1e-6)
    assert together["nll_sum"] == pytest.approx(separate["nll_sum"], rel=1e-6)
    assert together["by_domain"]["prose"]["targets"] == together["targets"]


def test_score_rejects_bad_mask_and_special_tokens(tokenizer):
    item = window(tokenizer, "猫")
    scorer = TokenTotals()
    with pytest.raises(ValueError, match="One NLL"):
        scorer.add(item, [0.0], tokenizer)
    illegal = Window((tokenizer.bos_id, tokenizer.bos_id), "b", "prose", 0)
    with pytest.raises(ValueError, match="BOS"):
        scorer.add(illegal, [1.0], tokenizer)
    with pytest.raises(ValueError, match="Invalid target NLL"):
        scorer.add(item, [float("nan")] * (len(item.ids) - 1), tokenizer)


def create_view(root, records):
    body_map = {}
    samples = []
    for ident, split, group, text in records:
        body = text.encode("utf-8")
        filename = f"{split}/{ident}.txt"
        body_map[filename] = body
        samples.append({
            "record_id": ident, "split": split, "domain": "prose",
            "text_path": filename, "text_sha256": digest(body),
            "leakage_group": group,
        })
    body_map["samples.jsonl"] = (
        "".join(json.dumps(row) + "\n" for row in samples)
    ).encode()
    manifest = json_bytes({
        "schema_version": "aster-training-view-0",
        "files": {path: digest(body) for path, body in body_map.items()},
    })
    view = root / digest(manifest)
    for filename, body in body_map.items():
        path = view / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    (view / "manifest.json").write_bytes(manifest)
    return view


def test_prepare_windows_does_not_read_sealed_test_bytes(tmp_path, monkeypatch, tokenizer):
    view = create_view(tmp_path, [
        ("train1", "train", "gtrain", "猫"),
        ("dev1", "dev", "gdev", "犬"),
        ("test1", "test", "gtest", "封印されている本文"),
    ])
    original = Path.read_bytes

    def guarded(path):
        if path.parent.name == "test":
            raise AssertionError("Sealed test bytes must not be opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    windows = prepare_windows(view, tokenizer, 4)
    assert windows["train"] and windows["dev"]
    assert all(w.record_id != "test1" for split in windows.values() for w in split)


def test_split_metadata_prevents_test_leakage_without_opening_test(tmp_path, tokenizer):
    view = create_view(tmp_path, [
        ("train1", "train", "shared_group", "猫"),
        ("test1", "test", "shared_group", "犬"),
        ("dev1", "dev", "dev_group", "鳥"),
    ])
    with pytest.raises(ValueError, match="Cross-split leakage"):
        verify_view(view, skip_test_bytes=True)


def test_frozen_config_drift_rejected_before_local_artifacts(tmp_path):
    config = tmp_path / "configs/tinylm-pilot-v0.json"
    config.parent.mkdir()
    config.write_text('{"mode":"pilot","steps":201}', encoding="utf-8")
    with pytest.raises(ValueError, match="config digest mismatch"):
        preflight(tmp_path)
