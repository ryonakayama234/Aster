import json

import pytest

from aster.tokenizer.artifact import (
    load_tokenizer,
    save_tokenizer,
    train_aster_tokenizer,
)


def test_special_tokens_do_not_collide_with_bpe_vocab():
    tokenizer = train_aster_tokenizer(
        ["banana banana", "こんにちは", "Aster 🚀"],
        target_vocab_size=300,
    )

    bpe_ids = set(tokenizer.model.vocab)
    special_ids = set(tokenizer.special_tokens.values())

    assert bpe_ids.isdisjoint(special_ids)
    assert tokenizer.bos_id < tokenizer.eos_id


def test_encode_can_add_bos_and_eos():
    tokenizer = train_aster_tokenizer(
        ["hello hello"],
        target_vocab_size=280,
    )

    ids = tokenizer.encode(
        "hello",
        add_bos=True,
        add_eos=True,
    )

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello"


def test_decode_rejects_special_tokens_when_not_skipping():
    tokenizer = train_aster_tokenizer(
        ["hello hello"],
        target_vocab_size=280,
    )

    ids = tokenizer.encode("hello", add_bos=True)

    with pytest.raises(ValueError):
        tokenizer.decode(ids, skip_special_tokens=False)


def test_save_and_load_preserve_tokenizer_behavior(tmp_path):
    tokenizer = train_aster_tokenizer(
        [
            "hello hello",
            "こんにちはこんにちは",
            "Aster 🚀 Aster 🚀",
            '{"tool": "python"}',
        ],
        target_vocab_size=320,
    )

    artifact_dir = tmp_path / "AsterTokenizer-v0.1"
    save_tokenizer(tokenizer, artifact_dir)
    loaded = load_tokenizer(artifact_dir)

    texts = [
        "hello",
        "こんにちは",
        "Aster 🚀",
        '{"tool": "python"}',
        "",
    ]

    for text in texts:
        before = tokenizer.encode(text, add_bos=True, add_eos=True)
        after = loaded.encode(text, add_bos=True, add_eos=True)

        assert before == after
        assert loaded.decode(after) == text

    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.manifest == tokenizer.manifest


def test_save_creates_readable_artifact_files(tmp_path):
    tokenizer = train_aster_tokenizer(
        ["banana banana banana"],
        target_vocab_size=280,
    )

    artifact_dir = tmp_path / "AsterTokenizer-v0.1"
    save_tokenizer(tokenizer, artifact_dir)

    expected_files = {
        "manifest.json",
        "vocab.json",
        "merges.json",
        "special_tokens.json",
    }

    assert {path.name for path in artifact_dir.iterdir()} == expected_files

    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["name"] == "AsterTokenizer"
    assert manifest["version"] == "0.1"
    assert manifest["tokenizer_type"] == "byte_bpe"
    assert manifest["base_vocab_size"] == 256
