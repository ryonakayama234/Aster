"""Register numeric-feature diagnostics, then measure the learned action classifier."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from aster.benchmark.state_representation import digest
from aster.training.decision_action_types import prepare,run_study

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('preflight','debug','measure'))
    p.add_argument('--expected-digest');p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];registration=repo/'reports/decision-action-types-preflight-v0.json';torch.set_num_threads(2)
    if a.mode=='preflight':
        report=prepare()[-1];registration.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'digest':digest(report),'train_patterns':report['train_unique_feature_patterns'],'dev_feature_overlap':report['dev_feature_train_overlap']}))
    else:
        registered=json.loads(registration.read_text())
        if digest(registered)!=a.expected_digest:raise ValueError('Explicit expected registration digest required/mismatch')
        paths=subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard'],cwd=repo,text=True).splitlines()
        hashes={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sorted(set(paths)) if s.endswith('.py') and (repo/s).is_file()}
        source=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()),
                    python_file_sha256=hashes,python_snapshot_sha256=digest(hashes),
                    protocol_sha256=hashlib.sha256((repo/'docs/DecisionActionTypes-v0.md').read_bytes()).hexdigest())
        print(run_study(a.root.resolve(),registered,source=source,debug=a.mode=='debug'),flush=True)
