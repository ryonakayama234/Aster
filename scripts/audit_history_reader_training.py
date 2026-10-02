"""Audit saved reader logits and replay every recorded episode without retraining."""
import argparse
import json
import math
from pathlib import Path

from aster.benchmark.history_reader import build_cases, read_facts, replay
from aster.benchmark.state_representation import digest
from aster.training.history_reader import (CLASSES, FACTS, execute_proposal,
    select_action, terminal_check, load_classifier)
from aster.records.transition import Action
from aster.evaluator.episode import evaluate_episode


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def validate_prediction(row):
    actual = {k:CLASSES[max(range(3),key=lambda j:row['logits'][i][j])] for i,k in enumerate(FACTS)}
    if actual != row['predicted'] or (actual == row['truth']) != row['correct']:
        raise ValueError('Cached reader prediction mismatch')
    losses, margins = [], []
    for i,k in enumerate(FACTS):
        v = row['logits'][i]
        if not all(math.isfinite(x) for x in v):raise ValueError('Nonfinite cached logits')
        target = CLASSES.index(row['truth'][k]);peak = max(v)
        losses.append(peak+math.log(sum(math.exp(x-peak) for x in v))-v[target])
        margins.append(v[target]-max(x for j,x in enumerate(v) if j != target))
    if abs(sum(losses)/3-row['nll'])>1e-10 or any(abs(x-y)>1e-10 for x,y in zip(margins,row['margins'],strict=True)):
        raise ValueError('Cached loss/margin mismatch')


def audit_run(path):
    rows = lines(path/'cases.jsonl'); expected=build_cases()
    if rows != expected:
        raise ValueError('Saved cases changed')
    by_id = {r['case_id']:r for r in rows}
    classifier, _ = load_classifier(path/'fixed-classifier')
    counters = dict(train_predictions=0,final_predictions=0,episodes=0,transitions=0,visited=0)
    for arm in sorted(path.glob('seed-*')):
        predictions = lines(arm/'train-predictions.jsonl')
        curves = lines(arm/'learning-curve.jsonl')
        for row in predictions:
            validate_prediction(row)
            if row['truth'] != by_id[row['case_id']]['facts']:raise ValueError('Train truth mismatch')
        for c in curves:
            group=[r for r in predictions if r['epoch']==c['epoch']]
            if len(group)!=80 or c['correct']!=sum(r['correct'] for r in group) or abs(c['nll']-sum(r['nll'] for r in group)/80)>1e-10:
                raise ValueError('Curve/count mismatch')
            if c['min_margin']!=min(min(r['margins']) for r in group):raise ValueError('Curve margin mismatch')
        counters['train_predictions']+=len(predictions)
        final=lines(arm/'final-predictions.jsonl')
        for row in final:
            validate_prediction(row)
            if row['truth']!=by_id[row['case_id']]['facts']:raise ValueError('Final truth mismatch')
        counters['final_predictions']+=len(final)
        for episode in lines(arm/'episodes.jsonl'):
            session=replay(by_id[episode['case_id']])
            for visit in episode['visited']:
                text=session.input_text()
                if visit['input']!=text or visit['input_sha256']!=digest(text) or visit['truth']!=read_facts(text):
                    raise ValueError('Visited input/truth mismatch')
                predicted={k:CLASSES[max(range(3),key=lambda j:visit['logits'][i][j])] for i,k in enumerate(FACTS)}
                if visit['predicted']!=predicted or visit['fact_correct']!=(predicted==visit['truth']):
                    raise ValueError('Visited prediction mismatch')
                action,failure=select_action(classifier,predicted,session)
                selected=None if action is None else action.to_dict()
                if selected!=visit['selected'] or failure!=visit['failure'] or session.next_action().to_dict()!=visit['teacher']:
                    raise ValueError('Visited action/binding mismatch')
                if visit['action_correct']!=(selected==visit['teacher']):raise ValueError('Visited action correctness mismatch')
                if action is not None:execute_proposal(session,action)
                counters['visited']+=1
            trajectory=[t.to_dict() for t in session.recorder.trajectory().transitions]
            if trajectory!=episode['trajectory'] or evaluate_episode(session.recorder.trajectory()).to_dict()!=episode['summary']:
                raise ValueError('Episode replay mismatch')
            if episode['success'] is not None and terminal_check(session)!=episode['success']:
                raise ValueError('Independent terminal mismatch')
            counters['transitions']+=len(trajectory);counters['episodes']+=1
    for episode in lines(path/'oracle-episodes.jsonl'):
        session=replay(by_id[episode['case_id']])
        for saved in episode['trajectory'][len(by_id[episode['case_id']]['trajectory']):]:
            if execute_proposal(session,Action.from_dict(saved['action'])).to_dict()!=saved:
                raise ValueError('Oracle replay mismatch')
        if not terminal_check(session):raise ValueError('Oracle terminal mismatch')
    return counters


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(audit_run(a.run),sort_keys=True))
