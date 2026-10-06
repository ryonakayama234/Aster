#!/usr/bin/env python3
"""Materialize DIAG v3 pre-outcome confounding/collinearity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.training.diag_family_preoutcome import run_preoutcome_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--feature-run-id", required=True)
    args = parser.parse_args()

    run_path = run_preoutcome_audit(args.root, args.feature_run_id)
    result = json.loads(
        (run_path / "diag-family-preoutcome-audit.json").read_text(encoding="utf-8")
    )
    print(run_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
