"""Inspect a saved eight-example fit checkpoint; no training or calibration."""
import argparse
from pathlib import Path
import subprocess

import torch

from aster.benchmark.state_diagnostics import run_state_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(args.threads)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    print(run_state_diagnostics(root, args.checkpoint.resolve(), source_git_sha=sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
