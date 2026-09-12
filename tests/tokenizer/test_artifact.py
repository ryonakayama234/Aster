from aster.tokenizer.artifact import (
    load_bpe,
    save_bpe,
)
from aster.tokenizer.bpe import train_bpe


def test_save_and_load_bpe(tmp_path):
    texts = [
        "hello hello",
        "こんにちは",
        "Aster tokenizer",
    ]

    original = train_bpe(
        texts,
        target_vocab_size=300,
    )

    artifact_dir = tmp_path / "tokenizer"

    save_bpe(
        original,
        artifact_dir,
    )

    loaded = load_bpe(
        artifact_dir,
    )

    assert loaded.merges == original.merges
    assert loaded.vocab == original.vocab

def test_loaded_bpe_has_same_behavior(tmp_path):
    texts = [
        "banana banana",
        "こんにちは",
        "Aster 🚀",
    ]

    original = train_bpe(
        texts,
        target_vocab_size=300,
    )

    artifact_dir = tmp_path / "tokenizer"

    save_bpe(original, artifact_dir)
    loaded = load_bpe(artifact_dir)

    test_text = "Aster こんにちは 🚀"

    assert loaded.encode(test_text) == original.encode(test_text)

    ids = loaded.encode(test_text)

    assert loaded.decode(ids) == test_text