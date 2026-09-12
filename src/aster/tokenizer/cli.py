from __future__ import annotations

import argparse
from pathlib import Path

from aster.tokenizer.build import build_tokenizer_v0_1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aster-tokenizer",
        description="Build versioned Aster tokenizer artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_v0_1 = subparsers.add_parser(
        "build-v0.1",
        help="Train and save AsterTokenizer-v0.1 from canonical .txt files.",
    )
    build_v0_1.add_argument(
        "--canonical-dir",
        type=Path,
        default=Path("data/canonical"),
    )
    build_v0_1.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizers/AsterTokenizer-v0.1"),
    )
    build_v0_1.add_argument(
        "--target-vocab-size",
        type=int,
        default=512,
    )
    build_v0_1.add_argument(
        "--min-pair-frequency",
        type=int,
        default=2,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-v0.1":
        tokenizer = build_tokenizer_v0_1(
            args.canonical_dir,
            args.output_dir,
            target_vocab_size=args.target_vocab_size,
            min_pair_frequency=args.min_pair_frequency,
        )

        print(f"saved: {args.output_dir}")
        print(f"bpe vocab: {len(tokenizer.model.vocab)}")
        print(f"special tokens: {tokenizer.special_tokens}")
        print(f"total vocab: {tokenizer.vocab_size}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
