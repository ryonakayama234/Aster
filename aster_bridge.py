#!/usr/bin/env python3
"""Compatibility entry point for the local Aster service."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aster.service.http import main


if __name__ == "__main__":
    raise SystemExit(main())
