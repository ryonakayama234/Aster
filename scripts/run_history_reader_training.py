"""Run preregistered history-reader training using an existing frozen classifier."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from aster.training.history_reader import run_study

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('debug', 'measure'))
    p.add_argument('--classifier', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--expected-suite', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    registration = json.loads((repo/'reports/history-reader-preflight-v0.json').read_text())
    if registration['suite_sha256'] != args.expected_suite:
        raise ValueError('Explicit reader suite digest mismatch')
    paths = subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard'],cwd=repo,text=True).splitlines()
    source = dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  git_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),
                  dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()),
                  python_file_sha256={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sorted(set(paths)) if s.endswith('.py') and (repo/s).is_file()},
                  protocol_sha256=hashlib.sha256((repo/'docs/HistoryFactReaderTraining-v0.md').read_bytes()).hexdigest())
    torch.set_num_threads(2)
    print(run_study(args.root.resolve(),args.classifier.resolve(),registration,source,debug=args.mode=='debug'),flush=True)
