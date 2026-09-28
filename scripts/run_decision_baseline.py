"""Run the CPU AsterDecision baseline and reveal its sealed test separately."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.training.decision_baseline import (
    DecisionBaselineConfig,
    run_logged_decision_baseline,
    run_logged_decision_final_test,
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train/calibrate/dev-evaluate; keep test sealed")
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--seed", type=int, default=42)

    test = subparsers.add_parser("test", help="reveal test for one saved model artifact")
    test.add_argument("--artifact", type=Path, required=True)

    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(args.threads)
    suite = build_calculate_and_store_suite()

    if args.command == "train":
        run_path = run_logged_decision_baseline(
            root,
            suite,
            config=DecisionBaselineConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
            source_git_sha=_git_sha(root),
        )
    else:
        artifact = args.artifact
        if not artifact.is_absolute():
            artifact = (root / artifact).resolve()
        run_path = run_logged_decision_final_test(root, artifact, suite)

    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
