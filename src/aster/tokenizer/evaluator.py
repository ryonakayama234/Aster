from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenizerEvalResult:
    documents: int
    characters: int
    utf8_bytes: int
    tokens: int
    round_trip_failures: int

    @property
    def round_trip_rate(self) -> float:
        return (self.documents - self.round_trip_failures) / self.documents

    @property
    def tokens_per_character(self) -> float:
        if self.characters == 0:
            return 0.0
        return self.tokens / self.characters

    @property
    def tokens_per_byte(self) -> float:
        if self.utf8_bytes == 0:
            return 0.0
        return self.tokens / self.utf8_bytes


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    texts: list[str],
) -> TokenizerEvalResult:
    if not texts:
        raise ValueError("texts must not be empty")

    characters = 0
    utf8_bytes = 0
    tokens = 0
    round_trip_failures = 0

    for text in texts:
        ids = tokenizer.encode(text)
        restored = tokenizer.decode(ids)

        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        tokens += len(ids)

        if restored != text:
            round_trip_failures += 1

    return TokenizerEvalResult(
        documents=len(texts),
        characters=characters,
        utf8_bytes=utf8_bytes,
        tokens=tokens,
        round_trip_failures=round_trip_failures,
    )
