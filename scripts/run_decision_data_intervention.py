"""Compare four equal-budget training-data conditions; test stays sealed."""
import argparse
from pathlib import Path
import subprocess

import torch

from aster.training.decision_data import run_data_intervention
from aster.training.decision_fit import DecisionFitStudyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.root.resolve()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        sha = sha + ":working-tree" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        sha = None
    print(run_data_intervention(root, config=DecisionFitStudyConfig(epochs=args.epochs,
        learning_rate=args.learning_rate, seed=args.seed), source_git_sha=sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
