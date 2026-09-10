from aster.tokenizer.bpe import (
    count_pairs,
    merge_pair,
    train_bpe,
)


def test_count_pairs():
    sequences = [
        [1, 2, 3, 2, 3],
    ]

    counts = count_pairs(sequences)

    assert counts[(1, 2)] == 1
    assert counts[(2, 3)] == 2
    assert counts[(3, 2)] == 1


def test_merge_pair():
    ids = [1, 2, 1, 2, 3]

    merged = merge_pair(
        ids,
        pair=(1, 2),
        new_id=256,
    )

    assert merged == [256, 256, 3]


def test_round_trip():
    texts = [
        "hello",
        "こんにちは",
        "Aster 🚀",
        "😀😃😄",
        "",
    ]

    model = train_bpe(
        texts,
        target_vocab_size=300,
    )

    for text in texts:
        ids = model.encode(text)
        restored = model.decode(ids)

        assert restored == text


def test_bpe_compresses_repeated_text():
    text = "banana banana banana"

    model = train_bpe(
        [text],
        target_vocab_size=300,
    )

    byte_ids = list(text.encode("utf-8"))
    bpe_ids = model.encode(text)

    assert len(bpe_ids) < len(byte_ids)


def test_training_is_deterministic():
    texts = [
        "banana banana",
        "bandana banana",
    ]

    model_a = train_bpe(texts, target_vocab_size=300)
    model_b = train_bpe(texts, target_vocab_size=300)

    assert model_a.merges == model_b.merges
    assert model_a.vocab == model_b.vocab