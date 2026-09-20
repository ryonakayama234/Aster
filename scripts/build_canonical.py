"""Refresh the inventory and build canonical candidates with one command."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts/inventory_corpus.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "aster.corpus.pipeline", "--root", str(ROOT)], cwd=ROOT, check=True)
