"""Compare fixed compact/byte training data; leave all tests sealed."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch

from aster.benchmark.state_representation import digest
from aster.training.decision_fit import DecisionFitStudyConfig
from aster.training.decision_state_training import run_state_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--epochs", type=int, default=50, help="Other values are debug only")
    args = parser.parse_args()
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=repo, text=True).splitlines()
    hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sorted(set(paths))
              if p.endswith(".py") and (repo / p).is_file()}
    source = {"git_sha": sha, "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
              "python_snapshot_sha256": digest(hashes), "python_file_sha256": hashes,
              "protocol_sha256": hashlib.sha256((repo / "docs/DecisionStateTraining-v0.md").read_bytes()).hexdigest()}
    print(run_state_training(args.root.resolve(), config=DecisionFitStudyConfig(epochs=args.epochs, learning_rate=1e-3, seed=args.seed),
                             source_git_sha=sha, source_provenance=source), flush=True)
