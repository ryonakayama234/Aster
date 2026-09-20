"""Build the pinned pilot view and run record; never silently refresh canonical."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    subprocess.run([sys.executable, '-m', 'aster.training.view', '--root', str(ROOT), *sys.argv[1:]], cwd=ROOT, check=True)
