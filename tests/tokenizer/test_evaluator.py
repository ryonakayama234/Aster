import pytest

from aster.tokenizer.bpe import train_bpe
from aster.tokenizer.evaluator import evaluate_tokenizer


def test_evaluate_tokenizer_counts_metrics():
    model = train_bpe(
        ["banana banana banana", "こんにちは Aster"],
        target_vocab_size=300,
    )

    texts = [
        "banana banana",
        "こんにちは",
        "Aster 🚀",
    ]

    result = evaluate_tokenizer(model, texts)

    assert result.documents == 3
    assert result.characters == sum(len(text) for text in texts)
    assert result.utf8_bytes == sum(len(text.encode("utf-8")) for text in texts)
    assert result.tokens == sum(len(model.encode(text)) for text in texts)
    assert result.round_trip_failures == 0
    assert result.round_trip_rate == 1.0
    assert result.tokens_per_character > 0
    assert result.tokens_per_byte > 0


def test_evaluate_tokenizer_detects_round_trip_failure():
    class BrokenTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, ids: list[int]) -> str:
            return "broken"

    result = evaluate_tokenizer(
        BrokenTokenizer(),
        ["hello", "こんにちは"],
    )

    assert result.round_trip_failures == 2
    assert result.round_trip_rate == 0.0


def test_evaluate_tokenizer_rejects_empty_dataset():
    model = train_bpe(
        ["hello"],
        target_vocab_size=300,
    )

    with pytest.raises(ValueError, match="texts must not be empty"):
        evaluate_tokenizer(model, [])
