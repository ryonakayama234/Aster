from __future__ import annotations

from pathlib import Path

from aster.tokenizer.artifact import (
    AsterTokenizer,
    save_tokenizer,
    train_aster_tokenizer,
)


def load_canonical_texts(canonical_dir: str | Path) -> list[str]:
    canonical_path = Path(canonical_dir)

    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical directory does not exist: {canonical_path}")

    text_files = sorted(
        path
        for path in canonical_path.rglob("*.txt")
        if path.is_file()
    )

    if not text_files:
        raise ValueError(
            f"No .txt canonical corpus files found under: {canonical_path}"
        )

    return [
        path.read_text(encoding="utf-8")
        for path in text_files
    ]


def build_tokenizer_v0_1(
    canonical_dir: str | Path,
    output_dir: str | Path,
    *,
    target_vocab_size: int = 512,
    min_pair_frequency: int = 2,
) -> AsterTokenizer:
    texts = load_canonical_texts(canonical_dir)

    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=target_vocab_size,
        min_pair_frequency=min_pair_frequency,
        name="AsterTokenizer",
        version="0.1",
    )

    save_tokenizer(tokenizer, output_dir)
    return tokenizer
