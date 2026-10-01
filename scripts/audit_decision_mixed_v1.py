"""Independently recompute saved train metrics and replay v1 model trajectories."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from aster.benchmark.fixed_v1_diagnostics import Session
from aster.benchmark.state_representation import compact_input, digest
from aster.records.decision import DecisionExample
from aster.records.transition import Action


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_score(row):
    values=row['scores']
    predicted=max(range(len(values)),key=lambda i:values[i])
    assert row['selected']==row['candidates'][predicted]
    assert row['target'] in row['candidates']
    ti=row['candidates'].index(row['target'])
    assert row['correct']==(predicted==ti)
    alternatives=[v for i,v in enumerate(values) if i!=ti]
    margin=values[ti]-max(alternatives) if alternatives else None
    assert margin==row['margin']
    assert [list(t.encode('utf-8')) for t in row['inputs']]==[[i for i in ids if i<256] for ids in row['token_ids']]
    assert digest(row['inputs'])==row['input_sha256']


def audit(run):
    study=json.loads((run/'study.json').read_text())
    cases=read_rows(run/'dev-cases.jsonl')
    pairs=json.loads((run/'pairs.json').read_text())
    totals=dict(train_predictions=0,dev_predictions=0,episodes=0,visited_predictions=0,transitions=0)
    train_data={r['case_id']:DecisionExample.from_dict(r['example']) for r in read_rows(run/'train-cases.jsonl')}
    for model in study['models']:
        assert model['status']=='completed'
        path=run/f"seed-{model['seed']}"/model['arm']
        curve=read_rows(path/'learning-curve.jsonl')
        by_epoch=defaultdict(list)
        for row in read_rows(path/'epoch-predictions.jsonl'):
            by_epoch[row['epoch']].append(row)
            ex=train_data[row['case_id']]
            assert row['candidates']==[a.to_dict() for a in ex.candidates] and row['target_action']==ex.target.to_dict()
            assert row['target_index']==ex.target_index
            assert row['predicted_index']==max(range(len(row['logits'])),key=lambda i:row['logits'][i])
            assert row['predicted_action']==row['candidates'][row['predicted_index']]
            totals['train_predictions']+=1
        for point in curve:
            rows=by_epoch[point['epoch']]
            assert len(rows)==len({r['case_id'] for r in rows})==model['fit']['unique']
            for key,subset in [('metrics',rows),('common_normal_metrics',[r for r in rows if '/normal-' in r['case_id']])]:
                correct=sum(r['predicted_index']==r['target_index'] for r in subset)
                nll=0; margins=[]
                for r in subset:
                    v=r['logits']; ti=r['target_index']; mx=max(v)
                    nll+=mx-v[ti]+math.log(sum(math.exp(x-mx) for x in v))
                    margins.append(v[ti]-max(x for i,x in enumerate(v) if i!=ti))
                m=point[key]
                assert m['correct']==correct and m['accuracy']==correct/len(subset)
                assert abs(m['nll']-nll/len(subset))<1e-10
                assert m['min_target_margin']==min(margins)
                assert m['all_target_margins_positive']==all(x>0 for x in margins)
        assert model['fit']['stable_fit']==all(c['metrics']['accuracy']==1 and c['metrics']['all_target_margins_positive'] for c in curve[-3:])
        dev=read_rows(path/'dev-predictions.jsonl')
        by_id={r['case_id']:r for r in dev}
        for c,r in zip(cases,dev,strict=True):
            ex=DecisionExample.from_dict(c['example'])
            assert c['case_id']==r['case_id'] and r['target']==ex.target.to_dict()
            assert r['inputs']==[compact_input(ex.state,ex.trajectory,a) for a in ex.candidates]
            check_score(r); totals['dev_predictions']+=1
        selected=[p for p in pairs if p['nonoverlap']]
        assert model['dev']['pairs']['nonoverlap_both_correct']==sum(by_id[p['left']]['correct'] and by_id[p['right']]['correct'] for p in selected)
        for c,e in zip(cases,read_rows(path/'prefix-episodes.jsonl'),strict=True):
            ex=DecisionExample.from_dict(c['example'])
            session=Session(ex.state.task)
            prefix=e['prefix_length']; assert prefix==len(ex.trajectory)
            for index,saved in enumerate(e['trajectory']):
                if index>=prefix:
                    r=e['decisions'][index-prefix]
                    visited=session.example()
                    assert r['state']==visited.state.to_dict()
                    assert r['target']==visited.target.to_dict()
                    assert r['inputs']==[compact_input(visited.state,visited.trajectory,a) for a in visited.candidates]
                    check_score(r); assert r['selected']==saved['action']
                    totals['visited_predictions']+=1
                transition=session.execute(Action.from_dict(saved['action']))
                assert transition.to_dict()==saved
                totals['transitions']+=1
            task=ex.state.task; expected=task['left']+task['right'] if task['operation']=='add' else task['left']-task['right']
            key=task['store_as']; last_put=-1; last_get=-1
            for i,t in enumerate(e['trajectory']):
                a=t['action']; obs=t['observation']; out=obs['output']
                if a['kind']=='tool' and a['arguments'].get('key')==key and obs['ok']:
                    if a['name']=='memory.put':last_put=i
                    elif a['name']=='memory.get' and isinstance(out,dict) and out.get('value')==expected:last_get=i
            stopped=bool(e['trajectory'] and e['trajectory'][-1]['action']['kind']=='stop')
            success=stopped and session.context.memory.get(key)==expected and 0<=last_put<last_get
            assert e['summary']['task_success']==success
            assert e['success']==(None if e['invalid'] else success)
            totals['episodes']+=1
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args=parser.parse_args(); print(json.dumps(audit(args.run),sort_keys=True))
