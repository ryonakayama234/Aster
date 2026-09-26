import json

import pytest

from aster.tokenizer.artifact import (
    load_tokenizer,
    save_tokenizer,
    train_aster_tokenizer,
)


def _saved_artifact(tmp_path):
    tokenizer = train_aster_tokenizer(
        ["banana banana banana", "こんにちはこんにちは", "Aster 🚀 Aster 🚀"],
        target_vocab_size=280,
    )
    artifact_dir = tmp_path / "AsterTokenizer-v0.1"
    save_tokenizer(tokenizer, artifact_dir)
    return artifact_dir


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


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


def test_load_rejects_string_in_manifest_integer_field(tmp_path):
    artifact_dir = _saved_artifact(tmp_path)
    path = artifact_dir / "manifest.json"
    manifest = _read_json(path)
    manifest["base_vocab_size"] = "256"
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="base_vocab_size must be an integer"):
        load_tokenizer(artifact_dir)


@pytest.mark.parametrize("bad_value", [1.5, True])
def test_load_rejects_non_integer_merge_ids(tmp_path, bad_value):
    artifact_dir = _saved_artifact(tmp_path)
    path = artifact_dir / "merges.json"
    merges = _read_json(path)
    assert merges
    merges[0]["left"] = bad_value
    _write_json(path, merges)

    with pytest.raises(ValueError, match="left must be an integer"):
        load_tokenizer(artifact_dir)


def test_load_rejects_boolean_special_token_id(tmp_path):
    artifact_dir = _saved_artifact(tmp_path)
    path = artifact_dir / "special_tokens.json"
    special_tokens = _read_json(path)
    special_tokens["<|bos|>"] = True
    _write_json(path, special_tokens)

    with pytest.raises(ValueError, match="Special-token ID.*must be an integer"):
        load_tokenizer(artifact_dir)


def test_load_rejects_noncanonical_vocab_key(tmp_path):
    artifact_dir = _saved_artifact(tmp_path)
    path = artifact_dir / "vocab.json"
    vocab = _read_json(path)
    token_hex = vocab.pop("1")
    vocab["01"] = token_hex
    _write_json(path, vocab)

    with pytest.raises(ValueError, match="canonical non-negative integer string"):
        load_tokenizer(artifact_dir)


def test_load_rejects_wrong_top_level_json_shape(tmp_path):
    artifact_dir = _saved_artifact(tmp_path)
    _write_json(artifact_dir / "manifest.json", [])

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_tokenizer(artifact_dir)
