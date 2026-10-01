"""Post-hoc train-only relation-pattern lookup; not a neural or holdout score."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from aster.benchmark.state_representation import canonical
from aster.records.decision import DecisionExample
from aster.records.decision_relations import observable_relations


def action_class(action):
    return action.name if action.kind == 'tool' else 'stop:'+str(action.arguments.get('reason'))


def pattern(example):
    facts=observable_relations(example.state,example.trajectory)
    return canonical({k:v for k,v in facts.items() if k not in ('memory_value','calculation_value')})


def analyze(run):
    lookup=defaultdict(set)
    train=[DecisionExample.from_dict(json.loads(line)['example']) for line in (run/'train-cases.jsonl').read_text().splitlines()]
    for ex in train:
        lookup[pattern(ex)].add(action_class(ex.target))
    if any(len(v)!=1 for v in lookup.values()):
        raise ValueError('Train relation-pattern label conflict')
    rows=[]
    for line in (run/'dev-cases.jsonl').read_text().splitlines():
        case=json.loads(line); ex=DecisionExample.from_dict(case['example'])
        labels=lookup.get(pattern(ex),set()); covered=len(labels)==1
        predicted=next(iter(labels)) if covered else None
        matches=[a for a in ex.candidates if action_class(a)==predicted]
        rows.append(dict(case_id=case['case_id'],group=case['group'],covered=covered,
                         correct=covered and len(matches)==1 and matches[0]==ex.target,
                         pattern=pattern(ex),predicted_action_class=predicted))
    study=json.loads((run/'study.json').read_text())
    model_checks=[]
    for model in study['models']:
        saved=[json.loads(line) for line in (run/f"seed-{model['seed']}"/model['arm']/'dev-predictions.jsonl').read_text().splitlines()]
        model_checks.append(dict(seed=model['seed'],arm=model['arm'],
            covered_count=sum(r['covered'] for r in rows),
            correct_on_covered=sum(r['covered'] and p['correct'] for r,p in zip(rows,saved,strict=True))))
    return dict(kind='posthoc train-only lookup on externally extracted boolean relations; not neural accuracy',
        preregistered=False,train_patterns=len(lookup),train_pattern_label_collisions=0,
        count=len(rows),covered=sum(r['covered'] for r in rows),correct=sum(r['correct'] for r in rows),
        groups={g:dict(count=sum(r['group']==g for r in rows),covered=sum(r['group']==g and r['covered'] for r in rows),
                       correct=sum(r['group']==g and r['correct'] for r in rows)) for g in ('normal','factorial','completion')},
        train_lookup={k:next(iter(v)) for k,v in lookup.items()},model_checks=model_checks,rows=rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args=parser.parse_args(); print(json.dumps(analyze(args.run),ensure_ascii=False,indent=2,sort_keys=True))
