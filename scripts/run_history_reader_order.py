"""Register or measure a matched-budget order intervention."""
import argparse
import hashlib
import json
from pathlib import Path

from run_history_reader_contrasts import source_metadata
from aster.benchmark.history_order import audit_order,build_order

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('register','measure'))
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--previous-run',type=Path)
    p.add_argument('--registration',type=Path)
    p.add_argument('--expected-suite')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = source_metadata(repo)
    source['protocol_sha256'] = hashlib.sha256((repo/'docs/HistoryReaderOrderIntervention-v0.md').read_bytes()).hexdigest()
    if a.mode == 'register':
        result = audit_order(*build_order())
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        if a.previous_run is None or a.registration is None or a.expected_suite is None:
            p.error('measure requires --previous-run --registration --expected-suite')
        registration = json.loads(a.registration.read_text())
        if registration['suite_sha256'] != a.expected_suite or source['dirty']:
            raise ValueError('Suite mismatch or dirty formal source')
        import torch
        from aster.training.history_order import run_intervention
        torch.set_num_threads(2)
        print(run_intervention(a.root.resolve(),a.previous_run.resolve(),registration,source),flush=True)
