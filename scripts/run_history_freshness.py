"""Generate, measure and independently audit the preregistered F-only study."""
import argparse
import json
from pathlib import Path

from aster.benchmark.history_freshness import audit_freshness,build_freshness

if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('preflight','debug','measure','audit'))
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--run',type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    registration_path = repo/'reports/history-freshness-preflight-v0.json'
    if args.mode=='preflight':
        rows,pairs = build_freshness()
        registration_path.write_text(json.dumps(audit_freshness(rows,pairs),ensure_ascii=False,indent=2)+'\n')
        print(registration_path)
    else:
        import torch
        from aster.training.history_freshness import audit_saved,run_study
        torch.set_num_threads(2)
        if args.mode=='audit':
            if args.run is None:
                p.error('--run required for audit')
            print(json.dumps(audit_saved(args.run),indent=2))
        else:
            registration = json.loads(registration_path.read_text())
            print(run_study(repo,args.root.resolve(),registration,debug=args.mode=='debug'),flush=True)
