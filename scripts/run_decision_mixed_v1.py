"""Run the preregistered normal/recovery comparison, or a separate wiring debug."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import torch

from aster.benchmark.state_representation import digest
from aster.training.decision_mixed_v1 import run_study

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--debug",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    repo=Path(__file__).resolve().parents[1]
    paths=subprocess.check_output(["git","ls-files","-c","-o","--exclude-standard"],cwd=repo,text=True).splitlines()
    hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()}
    source=dict(git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()),
                python_snapshot_sha256=digest(hashes),python_file_sha256=hashes,
                protocol_sha256=hashlib.sha256((repo/"docs/DecisionMixedMeasurement-v0.md").read_bytes()).hexdigest())
    print(run_study(args.root.resolve(),repo/"reports/decision-mixed-v1-preflight-v0.json",source=source,debug=args.debug),flush=True)
