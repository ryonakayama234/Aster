"""Run fixed raw/compact arms; never open sealed tests."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch

from aster.benchmark.state_representation import digest
from aster.training.decision_fit import DecisionFitStudyConfig
from aster.training.decision_representation import run_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=(42,43,44), default=42)
    parser.add_argument("--epochs", type=int, default=50, help="Other values are debug only")
    args = parser.parse_args()
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=repo, text=True).splitlines()
    hashes = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(set(paths))
              if p.endswith(".py") and (repo/p).is_file()}
    protocol_hash = hashlib.sha256((repo / "docs/DecisionRepresentationComparison-v0.md").read_bytes()).hexdigest()
    source = {"git_sha": sha, "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],cwd=repo,text=True).strip()),
              "python_snapshot_sha256": digest(hashes), "python_file_sha256": hashes, "protocol_sha256": protocol_hash}
    config = DecisionFitStudyConfig(epochs=args.epochs, learning_rate=1e-3, seed=args.seed)
    print(run_comparison(args.root.resolve(), config=config, source_git_sha=sha, source_provenance=source), flush=True)
