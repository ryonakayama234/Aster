#!/usr/bin/env python3
"""Run or resume DIAG v4 — Zero-init Residual Decision Head Trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aster.training.diag_residual_head import run_residual_head_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_path = run_residual_head_campaign(args.root)
    result = json.loads(
        (run_path / "diag-residual-head-results.json").read_text(encoding="utf-8")
    )
    print(run_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
