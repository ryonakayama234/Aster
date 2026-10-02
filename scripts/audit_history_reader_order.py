"""Rescore saved logits, exposure controls, reloads and independently replay episodes."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from audit_history_reader_training import validate_prediction
from audit_history_reader_contrasts import independent_terminal,scalar_selected_type
from aster.benchmark.history_order import EPOCHS,audit_order,build_order,training_slots
from aster.benchmark.history_reader import read_facts,replay
from aster.benchmark.state_representation import digest
from aster.evaluator.episode import evaluate_episode
from aster.records.transition import Action
from aster.training.history_contrasts import lines
from aster.training.history_reader import (CLASSIFIER_HASH,CLASSES,FACTS,execute_proposal,
    load_classifier,load_reader,predict,select_action)
from aster.training.history_order import summarize


def replay_episode(ep,row,classifier,oracle=False):
    session = replay(row)
    if ep['prefix_length'] != len(row['trajectory']) or ep['fallback'] is not False:
        raise ValueError('Episode prefix/fallback mismatch')
    for v in ep['visited']:
        text = session.input_text()
        truth = read_facts(text)
        inferred = truth if oracle else {k:CLASSES[max(range(3),key=lambda j:v['logits'][i][j])] for i,k in enumerate(FACTS)}
        if (v['input'] != text or v['input_sha256'] != digest(text) or v['truth'] != truth
                or v['predicted'] != inferred or v['fact_correct'] != (inferred==truth)):
            raise ValueError('Episode input/label/logit mismatch')
        action,failure = select_action(classifier,inferred,session)
        selected = None if action is None else action.to_dict()
        if (v['selected'] != selected or v['teacher'] != session.next_action().to_dict()
                or v['action_correct'] != (selected==v['teacher']) or v['failure'] != failure):
            raise ValueError('Episode action mismatch')
        if action is not None:
            manual = scalar_selected_type(classifier,inferred)
            if manual != (action.name if action.kind=='tool' else 'stop'):
                raise ValueError('Independent scalar action mismatch')
            execute_proposal(session,action)
    actual = [t.to_dict() for t in session.recorder.trajectory().transitions]
    if actual != ep['trajectory'] or evaluate_episode(session.recorder.trajectory()).to_dict()!=ep['summary']:
        raise ValueError('Episode tool replay mismatch')
    invalid = next((v['invalid'] for v in [ep] if v['invalid']),None)
    if invalid:
        if invalid == 'failed_tool_outside_reader_scope' and session.recorder.trajectory().transitions[-1].observation.ok:
            raise ValueError('Unjustified tool invalid')
        if invalid == 'reader_context_exceeded' and len(session.input_text().encode())+2 <= 2048:
            raise ValueError('Unjustified context invalid')
        if invalid not in ('failed_tool_outside_reader_scope','reader_context_exceeded') or ep['success'] is not None:
            raise ValueError('Unknown invalid/success')
    elif ep['success'] != independent_terminal(session):
        raise ValueError('Independent terminal mismatch')
    if ep['failure'] != (ep['visited'][-1]['failure'] if ep['visited'] else None):
        raise ValueError('Episode failure summary mismatch')
    if ep['first_fact_error'] != next((i for i,v in enumerate(ep['visited']) if not v['fact_correct']),None):
        raise ValueError('First error mismatch')
    return len(actual),len(ep['visited'])


def audit_run(path):
    rows,pairs = build_order()
    if lines(path/'cases.jsonl') != rows or lines(path/'pairs.jsonl') != pairs:
        raise ValueError('Saved order suite changed')
    registration = audit_order(rows,pairs)
    if json.loads((path/'registration.json').read_text()) != registration:
        raise ValueError('Saved registration changed')
    study = json.loads((path/'study.json').read_text())
    classifier,_ = load_classifier(path/'fixed-classifier')
    by_id = {r['case_id']:r for r in rows}
    counters = dict(predictions=0,train_predictions=0,episodes=0,transitions=0,visited=0,oracle=0,reloaded=0)
    for seed in (42,43,44):
        summaries = []
        for name in ('frozen','repeat','order'):
            arm = path/f'seed-{seed}'/name
            summary = json.loads((arm/'summary.json').read_text())
            if summary != next(m for m in study['models'] if m['seed']==seed and m['arm']==name):
                raise ValueError('Canonical summary mismatch')
            predictions = lines(arm/'predictions.jsonl')
            episodes = lines(arm/'episodes.jsonl')
            if len(predictions)!=264 or len(episodes)!=264 or {p['case_id'] for p in predictions}!=set(by_id) or {e['case_id'] for e in episodes}!=set(by_id):
                raise ValueError('Prediction/episode coverage mismatch')
            model = load_reader(arm/'checkpoint',CLASSIFIER_HASH)
            for p in predictions:
                validate_prediction(p)
                row = by_id[p['case_id']]
                if p['truth']!=row['facts'] or p['input_sha256']!=digest(row['input']):
                    raise ValueError('Cached label/input mismatch')
                if predict(model,row['input']) != {k:p[k] for k in ('logits','predicted')}:
                    raise ValueError('Reload/cached logits mismatch')
                s = replay(row)
                action,failure = select_action(classifier,p['predicted'],s)
                selected = None if action is None else action.to_dict()
                if p['selected']!=selected or p['teacher']!=s.next_action().to_dict() or p['failure']!=failure or p['action_correct']!=(selected==p['teacher']):
                    raise ValueError('Prediction action mismatch')
            expected,paired = summarize(rows,pairs,predictions,episodes,registration)
            if any(summary[k]!=v for k,v in expected.items()) or lines(arm/'paired-predictions.jsonl')!=paired:
                raise ValueError('Summary/pair mismatch')
            if name!='frozen':
                slots = training_slots(rows,name)
                if json.loads((arm/'slots.json').read_text()) != [r['case_id'] for r in slots]:
                    raise ValueError('Saved training slots changed')
                orders = json.loads((arm/'order.json').read_text())
                if len(orders)!=EPOCHS or any(sorted(o['indices'])!=list(range(112)) or o['epoch']!=i+1 for i,o in enumerate(orders)):
                    raise ValueError('Exposure/order mismatch')
                tokens = EPOCHS*sum(len(r['input'].encode())+2 for r in slots)
                labels = dict(Counter(json.dumps(r['facts'],sort_keys=True) for r in slots))
                if summary['updates']!=4480 or summary['train_tokens']!=tokens or summary['train_class_counts']!=labels:
                    raise ValueError('Budget/class control mismatch')
                curve_predictions = lines(arm/'train-predictions.jsonl')
                curves = lines(arm/'learning-curve.jsonl')
                if [c['epoch'] for c in curves]!=[0,10,20,30,40] or len(curve_predictions)!=560:
                    raise ValueError('Learning curve coverage mismatch')
                for p in curve_predictions:
                    validate_prediction(p)
                    if p['truth']!=by_id[p['case_id']]['facts']:
                        raise ValueError('Curve truth mismatch')
                for c in curves:
                    ps = [p for p in curve_predictions if p['epoch']==c['epoch']]
                    if [p['case_id'] for p in ps]!=[r['case_id'] for r in slots] or c['correct']!=sum(p['correct'] for p in ps) or c['updates']!=c['epoch']*112:
                        raise ValueError('Curve exposure/metric mismatch')
                    if abs(c['nll']-sum(p['nll'] for p in ps)/112)>1e-10 or c['min_margin']!=min(min(p['margins']) for p in ps):
                        raise ValueError('Curve loss/margin mismatch')
                    for i,k in enumerate(FACTS):
                        confusion = [[sum(p['truth'][k] is a and p['predicted'][k] is b for p in ps) for b in CLASSES] for a in CLASSES]
                        if c['facts'][k]['confusion']!=confusion:
                            raise ValueError('Curve confusion mismatch')
                counters['train_predictions']+=len(curve_predictions)
            elif summary['updates']!=0 or summary['model_sha256']!=summary['parent_model_sha256']:
                raise ValueError('Frozen model changed')
            for ep in episodes:
                n,v = replay_episode(ep,by_id[ep['case_id']],classifier)
                counters['transitions']+=n
                counters['visited']+=v
                counters['episodes']+=1
            counters['predictions']+=len(predictions)
            counters['reloaded']+=len(predictions)
            summaries.append(summary)
        if (summaries[1]['parent_model_sha256']!=summaries[2]['parent_model_sha256']
                or summaries[1]['train_class_counts']!=summaries[2]['train_class_counts']
                or (path/f'seed-{seed}'/'repeat/order.json').read_bytes()!=(path/f'seed-{seed}'/'order/order.json').read_bytes()):
            raise ValueError('Arm matching changed')
    oracle = lines(path/'oracle-episodes.jsonl')
    if len(oracle)!=264 or {e['case_id'] for e in oracle}!=set(by_id):
        raise ValueError('Oracle coverage changed')
    for ep in oracle:
        replay_episode(ep,by_id[ep['case_id']],classifier,oracle=True)
        if ep['success'] is not True:
            raise ValueError('Oracle failed')
    counters['oracle']=len(oracle)
    if study['oracle']['count']!=264 or study['oracle']['success']!=264:
        raise ValueError('Oracle summary changed')
    return counters


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args()
    print(json.dumps(audit_run(a.run),sort_keys=True))
