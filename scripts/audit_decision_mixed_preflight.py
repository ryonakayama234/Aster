"""Recompute saved preflight input/target, slot and overlap evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from aster.benchmark.state_representation import compact_input, digest, canonical
from aster.records.decision import DecisionExample


def audit_saved(path):
    summary = json.loads((path/'summary.json').read_text())
    slots = json.loads((path/'slots.json').read_text())
    pairs = json.loads((path/'pairs.json').read_text())
    rows = [json.loads(line) for name in ('train-cases.jsonl','dev-cases.jsonl')
            for line in (path/name).read_text().splitlines()]
    manifest = {r['case_id']:r for r in summary['case_manifest']}
    assert len(manifest) == len(rows) == 176
    signatures, targets, by_id = {}, {}, {}
    for row in rows:
        case_id = row['case_id']
        example = DecisionExample.from_dict(row['example'])
        texts = [compact_input(example.state,example.trajectory,a) for a in example.candidates]
        sig = digest(sorted(texts))
        assert sig == manifest[case_id]['signature']
        assert digest(row['example']) == manifest[case_id]['example_sha256']
        assert example.target.to_dict() == manifest[case_id]['target']
        assert [len(t.encode('utf-8'))+2 for t in texts] == manifest[case_id]['candidate_token_lengths']
        target = canonical(example.target.to_dict())
        assert targets.setdefault(sig,target) == target
        signatures[case_id] = sig
        by_id[case_id] = row
    union = {signatures[r['case_id']] for r in rows if r.get('split')=='train'}
    dev = [r for r in rows if r.get('split')!='train']
    assert len(union)==summary['train_unique_inputs']==48
    assert sum(signatures[r['case_id']] in union for r in dev)==summary['dev']['train_overlap']
    nonoverlap = sum(all(signatures[p[k]] not in union for k in ('left','right')) for p in pairs)
    assert nonoverlap==summary['dev_nonoverlap_pairs']
    assert digest(pairs)==summary['pairs_sha256']
    for name, ids in slots.items():
        assert digest(ids)==summary['arms'][name]['slot_sha256']
        frequency = Counter()
        for case_id in ids:
            target = manifest[case_id]['target']
            frequency[target['name'] if target['kind']=='tool' else 'stop']+=1
        assert dict(frequency)==summary['arms'][name]['action_slots']
        assert {i:n*25 for i,n in Counter(ids).items()}==summary['arms'][name]['case_exposures_25_epochs']
    for key, filename in (('train_suite_sha256','train-cases.jsonl'), ('dev_suite_sha256','dev-cases.jsonl')):
        value=[json.loads(line) for line in (path/filename).read_text().splitlines()]
        assert digest(value)==summary[key]
    return {'saved_cases':len(rows),'nonoverlap_pairs':nonoverlap,'slots_checked':sum(map(len,slots.values()))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_saved(args.run),sort_keys=True))
