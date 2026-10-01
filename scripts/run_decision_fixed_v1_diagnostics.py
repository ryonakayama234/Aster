"""Diagnose frozen PR #33 checkpoints under the current-state v1 contract."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch

from aster.benchmark.fixed_v1_diagnostics import run_diagnostics
from aster.benchmark.state_representation import digest

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--bundle",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    repo=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    paths=subprocess.check_output(["git","ls-files","-c","-o","--exclude-standard"],cwd=repo,text=True).splitlines()
    hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()}
    source=dict(git_sha=sha,dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()),
                python_snapshot_sha256=digest(hashes),python_file_sha256=hashes,
                protocol_sha256=hashlib.sha256((repo/"docs/DecisionFixedV1Diagnostics-v0.md").read_bytes()).hexdigest())
    print(run_diagnostics(args.root.resolve(),args.bundle.resolve(),source=source),flush=True)
