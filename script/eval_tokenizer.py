from argparse import ArgumentParser
from pathlib import Path

from aster.tokenizer.bpe import train_bpe
from aster.tokenizer.evaluator import evaluate_tokenizer


SUPPORTED_SUFFIXES = {".txt", ".py", ".jsonl"}


def read_texts(root: Path) -> list[str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
    )
    return [path.read_text(encoding="utf-8") for path in paths]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    args = parser.parse_args()

    train_root = Path("data/tokenizer/train")
    eval_root = Path("data/tokenizer/eval")

    train_texts = read_texts(train_root)
    if not train_texts:
        raise RuntimeError(f"no training texts found under {train_root}")

    model = train_bpe(
        train_texts,
        target_vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )

    print(
        f"{'category':<16}"
        f"{'docs':>6}"
        f"{'chars':>8}"
        f"{'bytes':>8}"
        f"{'tokens':>8}"
        f"{'tok/char':>10}"
        f"{'tok/byte':>10}"
        f"{'roundtrip':>12}"
    )

    for category_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        texts = read_texts(category_dir)
        if not texts:
            continue

        result = evaluate_tokenizer(model, texts)

        print(
            f"{category_dir.name:<16}"
            f"{result.documents:>6}"
            f"{result.characters:>8}"
            f"{result.utf8_bytes:>8}"
            f"{result.tokens:>8}"
            f"{result.tokens_per_character:>10.3f}"
            f"{result.tokens_per_byte:>10.3f}"
            f"{result.round_trip_rate:>11.1%}"
        )


if __name__ == "__main__":
    main()
