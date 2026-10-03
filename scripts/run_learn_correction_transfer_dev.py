"""Run the fixed LEARN-v0 correction-transfer development probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aster.training.learn_probe import DEFAULT_ACT_RUN_ID, run_learn_dev_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--act-run-id", default=DEFAULT_ACT_RUN_ID)
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_path = run_learn_dev_probe(
        args.root,
        act_run_id=args.act_run_id,
    )
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
