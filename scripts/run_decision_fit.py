"""Run controlled train-only fit experiments for AsterDecision-v0."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.training.decision_fit import (
    DecisionFitStudyConfig,
    default_fit_arms,
    run_logged_decision_fit_study,
)


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-full-batch",
        action="store_true",
        help="also run the optional 8-example full-batch arm",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    torch.set_num_threads(args.threads)
    suite = build_calculate_and_store_suite()
    run_path = run_logged_decision_fit_study(
        root,
        suite,
        config=DecisionFitStudyConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        arms=default_fit_arms(
            suite,
            include_full_batch=args.include_full_batch,
        ),
        source_git_sha=_git_sha(root),
    )
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
