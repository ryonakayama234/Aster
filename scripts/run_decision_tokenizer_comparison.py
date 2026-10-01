#!/usr/bin/env python
"""Run the preregistered DecisionModel BPE-vs-byte comparison for Issue #28."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch

from aster.training.decision_fit import DecisionFitStudyConfig
from aster.training.decision_tokenizer import EXPECTED_BPE_SHA256, run_tokenizer_comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--bpe-tokenizer",
        type=Path,
        required=True,
        help="Saved #27 BPE tokenizer artifact directory; it is loaded, never retrained.",
    )
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--expected-bpe-sha256", default=EXPECTED_BPE_SHA256)
    args = parser.parse_args()

    if args.threads != 2:
        parser.error("Issue #28 measurement protocol requires --threads 2")
    torch.set_num_threads(args.threads)
    root = args.root.resolve()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        sha = sha + ":working-tree" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        sha = None

    config = DecisionFitStudyConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        target_vocab_size=512,
        min_pair_frequency=1,
        context_length=2048,
        width=16,
        heads=2,
        layers=1,
        stability_window=3,
    )
    output = run_tokenizer_comparison(
        root,
        args.bpe_tokenizer.resolve(),
        config=config,
        source_git_sha=sha,
        expected_bpe_sha256=args.expected_bpe_sha256 or None,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
