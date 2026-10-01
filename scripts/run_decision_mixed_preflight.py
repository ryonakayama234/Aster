"""Generate and audit v1 mixed train cases without scoring or training a model."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from aster.benchmark.mixed_v1_preflight import run_preflight
from aster.benchmark.state_representation import digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=repo, text=True).splitlines()
    hashes = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest()
              for p in sorted(set(paths)) if p.endswith(".py") and (repo/p).is_file()}
    source = dict(git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
                  python_snapshot_sha256=digest(hashes), python_file_sha256=hashes,
                  protocol_sha256=hashlib.sha256((repo/"docs/DecisionMixedPreflight-v0.md").read_bytes()).hexdigest())
    print(run_preflight(args.root.resolve(), source=source))
