"""Independently recompute cached metrics and replay every bound action."""
import argparse,json,math
from pathlib import Path

from aster.benchmark.fixed_v1_diagnostics import Session
from aster.evaluator.episode import evaluate_episode
from aster.records.decision import DecisionExample
from aster.records.transition import Action
from aster.training.decision_action_types import TYPES,features,load_checkpoint,score


def read(path):return [json.loads(s) for s in path.read_text().splitlines()]


def audit(run):
    study=json.loads((run/'study.json').read_text())
    cases={r['case_id']:DecisionExample.from_dict(r['example']) for name in ('train','dev','fresh') for r in read(run/f'{name}-cases.jsonl')}
    totals=dict(train_predictions=0,dev_predictions=0,episodes=0,transitions=0,visited_predictions=0)
    for result in study['models']:
        path=run/f"seed-{result['seed']}";model,_=load_checkpoint(path/'checkpoint')
        train_rows=read(path/'train-predictions.jsonl')
        curves=read(path/'learning-curve.jsonl')
        for point in curves:
            rows=[r for r in train_rows if r['epoch']==point['epoch']]
            assert len(rows)==48 and len({r['case_id'] for r in rows})==48
            losses=[];correct=0;margins=[]
            for r in rows:
                ex=cases[r['case_id']];assert r['features']==features(ex)
                v=r['logits'];ti=TYPES.index(r['target_type']);pi=max(range(4),key=lambda i:v[i])
                assert r['target_type']==(ex.target.name if ex.target.kind=='tool' else 'stop')
                assert r['predicted_type']==TYPES[pi] and r['correct_type']==(pi==ti)
                margin=v[ti]-max(x for i,x in enumerate(v) if i!=ti)
                assert margin==r['margin'];margins.append(margin);correct+=pi==ti
                mx=max(v);losses.append(mx-v[ti]+math.log(sum(math.exp(x-mx) for x in v)))
                if point['epoch']==study['epochs']:assert score(model,ex)=={k:v for k,v in r.items() if k not in ('epoch','case_id')}
                totals['train_predictions']+=1
            assert point['correct']==correct and point['accuracy']==correct/48
            assert abs(point['nll']-sum(losses)/48)<1e-10 and point['min_margin']==min(margins)
        assert result['stable_fit']==(len(curves)>=3 and all(c['accuracy']==1 and c['min_margin']>0 for c in curves[-3:]))
        dev=read(path/'dev-predictions.jsonl')
        for r in dev:
            assert score(model,cases[r['case_id']])=={k:v for k,v in r.items() if k!='case_id'}
            totals['dev_predictions']+=1
        for name in ('episodes','new-episodes'):
            for e in read(path/f'{name}.jsonl'):
                ex=cases[e['case_id']];session=Session(ex.state.task);prefix=e['prefix_length']
                assert prefix==len(ex.trajectory)
                visited=0
                for index,t in enumerate(e['trajectory']):
                    if index>=prefix:
                        r=e['decisions'][visited];current=session.example();actual=score(model,current)
                        assert all(r[k]==v for k,v in actual.items()) and r['state']==current.state.to_dict()
                        assert r['selected']==t['action'];visited+=1;totals['visited_predictions']+=1
                    assert session.execute(Action.from_dict(t['action'])).to_dict()==t
                    totals['transitions']+=1
                # A binding failure has one proposal that was never executed.
                if e['failure']:
                    r=e['decisions'][visited];current=session.example();actual=score(model,current)
                    assert actual['binding_error']==e['failure'] and all(r[k]==v for k,v in actual.items())
                    totals['visited_predictions']+=1
                trajectory=session.recorder.trajectory();assert evaluate_episode(trajectory).to_dict()==e['summary']
                task=ex.state.task;key=task['store_as'];expected=task['left']+task['right'] if task['operation']=='add' else task['left']-task['right']
                puts=[i for i,t in enumerate(trajectory.transitions) if t.action.name=='memory.put' and t.action.arguments.get('key')==key and t.observation.ok]
                gets=[i for i,t in enumerate(trajectory.transitions) if t.action.name=='memory.get' and t.action.arguments.get('key')==key and t.observation.ok and isinstance(t.observation.output,dict) and t.observation.output.get('value')==expected]
                stopped=bool(trajectory.last and trajectory.last.action.kind=='stop')
                success=stopped and session.context.memory.get(key)==expected and bool(gets) and max(gets)>(max(puts) if puts else -1)
                assert e['summary']['task_success']==success
                assert e['success']==(None if e['invalid'] else success and e['failure'] is None)
                totals['episodes']+=1
        assert result['dev_action_correct']==sum(r['correct_action'] for r in dev)
        pairs=read(path/'pairs.jsonl');by_id={r['case_id']:r for r in dev}
        assert all(p['both_correct']==(by_id[p['left']]['correct_action'] and by_id[p['right']]['correct_action']) for p in pairs)
        assert result['pairs_correct']==sum(p['both_correct'] for p in pairs)
        episodes=read(path/'episodes.jsonl');new=read(path/'new-episodes.jsonl')
        assert result['prefix_success']==sum(e['success'] is True for e in episodes)
        assert result['new_initial_success']==sum(e['success'] is True for e in new)
    return totals


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.run),sort_keys=True))
