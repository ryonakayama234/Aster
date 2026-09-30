"""Inspect cached dev errors and token pieces using their original checkpoint."""
import argparse
from pathlib import Path

from aster.benchmark.decision_failure import audit_cached_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(audit_cached_failures(args.root, args.diagnostic_run, args.checkpoint))


if __name__ == "__main__":
    main()
