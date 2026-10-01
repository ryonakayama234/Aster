"""Register the input preflight, then run a digest-guarded paired comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from aster.benchmark.state_representation import digest
from aster.training.decision_relations import prepare, run_study

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("preflight","debug","measure"))
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--registration",type=Path,default=Path("reports/decision-relations-preflight-v0.json"))
    parser.add_argument("--expected-digest")
    args=parser.parse_args()
    torch.set_num_threads(2)
    repo=Path(__file__).resolve().parents[1]
    original=json.loads((repo/"reports/decision-mixed-v1-preflight-v0.json").read_text())
    if args.mode=="preflight":
        report=prepare(original)[-1]
        args.registration.parent.mkdir(parents=True,exist_ok=True)
        args.registration.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        print(json.dumps(dict(registration=str(args.registration),digest=digest(report),
                             common_nonoverlap_pairs=report['common_nonoverlap_pairs'],
                             representations={n:{k:v for k,v in r.items() if k not in ('manifest','overlap')} for n,r in report['representations'].items()})),flush=True)
    else:
        registration=json.loads(args.registration.read_text())
        if args.expected_digest!=digest(registration):
            raise ValueError("Explicit expected registration digest required/mismatch")
        paths=subprocess.check_output(["git","ls-files","-c","-o","--exclude-standard"],cwd=repo,text=True).splitlines()
        hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()}
        source=dict(git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                    dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()),
                    python_file_sha256=hashes,python_snapshot_sha256=digest(hashes),
                    protocol_sha256=hashlib.sha256((repo/"docs/DecisionRelationsComparison-v0.md").read_bytes()).hexdigest())
        print(run_study(args.root.resolve(),original,registration,source=source,debug=args.mode=="debug"),flush=True)
