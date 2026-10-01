"""Learn action types from numeric observable facts; bind existing arguments separately."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import signal
import time
from typing import Any

import torch
from torch import nn

from aster.benchmark.fixed_v1_diagnostics import Session, build_cases, replay
from aster.benchmark.mixed_v1_preflight import ARMS, build_training, independent_rule_check
from aster.benchmark.state_representation import canonical, digest
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.records.decision_relations import observable_relations
from aster.records.runlog import RunLog
from aster.records.transition import Action
from aster.training.decision_fit import _state_digest, _write_json, _write_jsonl

FEATURE_ID='aster-observed-cmf-0'
BINDER_ID='aster-action-type-candidate-binding-0'
SCHEMA='aster-action-types-study-0'
TYPES=('calculator','memory.put','memory.get','stop')
NEW_TASKS=(
    dict(kind='calculate_and_store',operation='add',left=123,right=-45,store_as='k_delta'),
    dict(kind='calculate_and_store',operation='add',left=-7,right=0,store_as='aux_total_v1'),
    dict(kind='calculate_and_store',operation='subtract',left=100,right=123,store_as='diff_new'),
    dict(kind='calculate_and_store',operation='subtract',left=-4,right=-9,store_as='slot_z'),
)


def action_type(action):
    return action.name if action.kind=='tool' else 'stop'


def features(example):
    facts=observable_relations(example.state,example.trajectory)
    if facts['last_tool_failed']:
        raise ValueError('last_tool_failure_outside_scope')
    return [int(facts['calculation_observed']),int(facts['memory_matches_calculation'] is True),
            int(facts['get_after_last_target_put'] is True and facts['get_matches_calculation'] is True)]


def bind_type(name, candidates):
    matching=[a for a in candidates if action_type(a)==name]
    if name=='stop':
        verified=[a for a in matching if a.arguments.get('reason')=='goal_verified']
        matching=verified if verified else matching
    if len(matching)!=1:
        raise ValueError('no_unique_candidate_for_action_type')
    return matching[0]


def load_checkpoint(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if (manifest['feature_id']!=FEATURE_ID or manifest['binder_id']!=BINDER_ID
            or manifest['goal_contract']!=CURRENT_GOAL_V1 or manifest['types']!=list(TYPES)):
        raise ValueError('Action-type checkpoint contract mismatch')
    weight=path/'weights.pt'
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=manifest['weights_file_sha256']:
        raise ValueError('Action-type checkpoint weights mismatch')
    model=nn.Linear(3,4)
    model.load_state_dict(torch.load(weight,weights_only=True,map_location='cpu'))
    if _state_digest(model.state_dict())!=manifest['model_sha256']:
        raise ValueError('Action-type checkpoint model identity mismatch')
    model.eval()
    return model,manifest


def score(model,example) -> dict[str, Any]:
    x=features(example)
    with torch.no_grad():
        logits=model(torch.tensor(x,dtype=torch.float32)).tolist()
    if not all(torch.isfinite(torch.tensor(logits))):
        raise ValueError('Nonfinite action-type scores')
    index=max(range(4),key=lambda i:logits[i])
    target=TYPES.index(action_type(example.target))
    try:
        selected=bind_type(TYPES[index],example.candidates)
        binding_error=None
    except ValueError as error:
        selected=None; binding_error=str(error)
    return dict(features=x,logits=logits,predicted_type=TYPES[index],target_type=TYPES[target],
                correct_type=index==target,selected=None if selected is None else selected.to_dict(),
                target=example.target.to_dict(),correct_action=selected==example.target,
                candidates=[a.to_dict() for a in example.candidates],binding_error=binding_error,
                margin=logits[target]-max(v for i,v in enumerate(logits) if i!=target))


def rollout(model,example) -> dict[str, Any]:
    session=replay(example); decisions=[]; failure=None; invalid=None
    for _ in range(8):
        ex=session.example()
        try:
            row=score(model,ex)
        except ValueError as error:
            invalid=str(error); break
        row['state']=ex.state.to_dict(); row['step']=ex.state.step
        decisions.append(row)
        if row['binding_error']:
            failure=row['binding_error']; break
        last=session.execute(Action.from_dict(row['selected']))
        if last.evaluation.terminal:
            break
    trajectory=session.recorder.trajectory()
    summary=evaluate_episode(trajectory).to_dict()
    return dict(success=None if invalid else bool(summary['task_success'] and failure is None),
                failure=failure,invalid=invalid,summary=summary,decisions=decisions,
                prefix_length=len(example.trajectory),fallback=False,
                trajectory=[t.to_dict() for t in trajectory.transitions])


def prepare():
    train,slots=build_training(); dev,pairs=build_cases()
    fresh: list[dict[str, Any]] = [dict(case_id=f'transfer-{i}',group='new_initial',variant='transfer',example=Session(t).example())
           for i,t in enumerate(NEW_TASKS)]
    labels=defaultdict(set); training_patterns=set(); manifest=[]
    for c in train+dev+fresh:
        ex=c['example']; replay(ex); independent_rule_check(ex)
        x=features(ex); name=action_type(ex.target)
        if bind_type(name,ex.candidates)!=ex.target or bind_type(name,tuple(reversed(ex.candidates)))!=ex.target:
            raise ValueError('Teacher candidate binding mismatch')
        modified=replace(ex.trajectory,transitions=tuple(replace(t,evaluation=replace(t.evaluation,goal_satisfied=not t.evaluation.goal_satisfied)) for t in ex.trajectory.transitions))
        if x!=features(replace(ex,trajectory=modified,teacher='changed',target_index=(ex.target_index+1)%len(ex.candidates))):
            raise ValueError('Supervision metadata leaks into features')
        labels[tuple(x)].add(name)
        if c.get('split')=='train':training_patterns.add(tuple(x))
        manifest.append(dict(case_id=c['case_id'],features=x,target_type=name,
                             example_sha256=digest(ex.to_dict())))
    if any(len(v)!=1 for v in labels.values()):
        raise ValueError('Feature/target-type collision')
    report=dict(schema_version=SCHEMA,feature_id=FEATURE_ID,binder_id=BINDER_ID,
                goal_contract=CURRENT_GOAL_V1,train_cases=48,dev_cases=128,new_initial_cases=4,
                slot_sha256=digest(slots[ARMS[1]]),manifest=manifest,pairs=pairs,
                train_unique_feature_patterns=len(training_patterns),collisions=0,
                dev_feature_train_overlap=sum(tuple(features(c['example'])) in training_patterns for c in dev),
                new_initial_feature_train_overlap=sum(tuple(features(c['example'])) in training_patterns for c in fresh),
                model_updates=0,model_scored=0,test_constructed=0,test_scored=0,
                independent_holdout=False)
    return train,slots[ARMS[1]],dev,pairs,fresh,report


def rows_metrics(rows):
    losses=[]
    for r in rows:
        v=torch.tensor(r['logits'],dtype=torch.float64)
        losses.append(float(torch.logsumexp(v,0)-v[TYPES.index(r['target_type'])]))
    return dict(count=len(rows),correct=sum(r['correct_type'] for r in rows),
                accuracy=sum(r['correct_type'] for r in rows)/len(rows),nll=sum(losses)/len(rows),
                min_margin=min(r['margin'] for r in rows))


def run_study(root,registration,*,source,debug=False):
    train,slots,dev,pairs,fresh,actual=prepare()
    if actual!=registration:raise ValueError('Action-type registration mismatch')
    epochs,seeds=(1,(42,)) if debug else (25,(42,43,44))
    protocol=dict(schema_version=SCHEMA,source=source,registration_sha256=digest(actual),
                  epochs=epochs,seeds=seeds,debug=debug,lr=0.01,optimizer='AdamW',threads=torch.get_num_threads())
    run=RunLog(Path(root),'decision_action_types',protocol,producer='trainer')
    _write_json(run.path/'protocol.json',protocol); _write_json(run.path/'registration.json',actual)
    _write_json(run.path/'slots.json',slots); _write_json(run.path/'pairs.json',pairs)
    for filename,cases in (('train-cases.jsonl',train),('dev-cases.jsonl',dev),('fresh-cases.jsonl',fresh)):
        _write_jsonl(run.path/filename,[{**{k:v for k,v in c.items() if k!='example'},'example':c['example'].to_dict()} for c in cases])
    by_id={c['case_id']:c['example'] for c in train}
    results=[]; started=time.perf_counter(); previous=signal.getsignal(signal.SIGALRM)
    def expire(signum,frame):raise TimeoutError('Action-type seed deadline exceeded')
    try:
        signal.signal(signal.SIGALRM,expire)
        for seed in seeds:
            signal.setitimer(signal.ITIMER_REAL,300); path=run.path/f'seed-{seed}';path.mkdir()
            torch.manual_seed(seed); model=nn.Linear(3,4)
            initial=_state_digest(model.state_dict()); torch.save(model.state_dict(),path/'initial-weights.pt')
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
            generator=torch.Generator().manual_seed(seed); curves=[]; predictions=[]; orders=[]
            def evaluate(epoch):
                model.eval(); before=_state_digest(model.state_dict()); rng=torch.get_rng_state().clone()
                rows=[dict(case_id=c['case_id'],epoch=epoch,**score(model,c['example'])) for c in train]
                if _state_digest(model.state_dict())!=before or not torch.equal(rng,torch.get_rng_state()):
                    raise ValueError('Train evaluation changed weights/RNG')
                predictions.extend(rows);curves.append(dict(epoch=epoch,updates=epoch*64,**rows_metrics(rows)))
            evaluate(0)
            for epoch in range(1,epochs+1):
                order=torch.randperm(len(slots),generator=generator).tolist();orders.append(dict(epoch=epoch,slot_indices=order))
                model.train()
                for index in order:
                    ex=by_id[slots[index]];x=torch.tensor(features(ex),dtype=torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(model(x).unsqueeze(0),torch.tensor([TYPES.index(action_type(ex.target))]))
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
                    loss.backward();optimizer.step()
                evaluate(epoch)
            _write_jsonl(path/'learning-curve.jsonl',curves);_write_jsonl(path/'train-predictions.jsonl',predictions);_write_json(path/'order.json',orders)
            checkpoint=path/'checkpoint';checkpoint.mkdir();torch.save(model.state_dict(),checkpoint/'weights.pt')
            manifest=dict(feature_id=FEATURE_ID,binder_id=BINDER_ID,types=list(TYPES),goal_contract=CURRENT_GOAL_V1,
                          model_sha256=_state_digest(model.state_dict()),initial_sha256=initial,parameter_count=16,
                          weights_file_sha256=hashlib.sha256((checkpoint/'weights.pt').read_bytes()).hexdigest(),resume_supported=False,
                          registration_sha256=digest(actual),source=source)
            _write_json(checkpoint/'manifest.json',manifest)
            reloaded,_=load_checkpoint(checkpoint)
            if [score(reloaded,c['example']) for c in train]!=[score(model,c['example']) for c in train]:
                raise ValueError('Train reload mismatch')
            before=_state_digest(model.state_dict());rng=torch.get_rng_state().clone()
            rows=[dict(case_id=c['case_id'],**score(model,c['example'])) for c in dev]
            episodes=[dict(case_id=c['case_id'],group=c['group'],variant=c['variant'],**rollout(model,c['example'])) for c in dev]
            new_episodes=[dict(case_id=c['case_id'],**rollout(model,c['example'])) for c in fresh]
            if rows!=[dict(case_id=c['case_id'],**score(reloaded,c['example'])) for c in dev]:
                raise ValueError('Dev reload mismatch')
            if before!=_state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                raise ValueError('Dev changed weights/RNG')
            by_dev={r['case_id']:r for r in rows}
            pair_rows=[{**p,'both_correct':by_dev[p['left']]['correct_action'] and by_dev[p['right']]['correct_action']} for p in pairs]
            summary=dict(seed=seed,stable_fit=len(curves)>=3 and all(c['accuracy']==1 and c['min_margin']>0 for c in curves[-3:]),
                train=curves[-1],dev_type_correct=sum(r['correct_type'] for r in rows),dev_action_correct=sum(r['correct_action'] for r in rows),
                pairs_correct=sum(p['both_correct'] for p in pair_rows),pairs_count=len(pair_rows),
                prefix_success=sum(e['success'] is True for e in episodes),prefix_count=128,
                binding_failures=sum(e['failure'] is not None for e in episodes),invalid=sum(e['invalid'] is not None for e in episodes),
                new_initial_success=sum(e['success'] is True for e in new_episodes),new_initial_count=4,
                initial={v:dict(count=sum(c['group']=='normal' and c['factors']['phase']==0 and c['variant']==v for c in dev),
                          success=sum(c['group']=='normal' and c['factors']['phase']==0 and c['variant']==v and e['success'] is True for c,e in zip(dev,episodes,strict=True))) for v in ('seen','numbers','key','joint')},
                reload_match=True,weights_rng_unchanged=True,model_sha256=before,initial_sha256=initial,order_sha256=digest(orders))
            for fn,value in (('dev-predictions.jsonl',rows),('episodes.jsonl',episodes),('new-episodes.jsonl',new_episodes),('pairs.jsonl',pair_rows)):_write_jsonl(path/fn,value)
            _write_json(path/'summary.json',summary);results.append(summary);_write_json(run.path/'partial-results.json',results)
            print(f'seed={seed} fit={summary["stable_fit"]} dev={summary["dev_action_correct"]}/128 prefix={summary["prefix_success"]}/128 new={summary["new_initial_success"]}/4',flush=True)
            signal.setitimer(signal.ITIMER_REAL,0)
        study=dict(**protocol,run_id=run.id,models=results,preflight=actual,python=platform.python_version(),torch=torch.__version__,
                   wall_seconds=time.perf_counter()-started,independent_holdout=False,test_scored=0)
        _write_json(run.path/'study.json',study);run.finish('completed',study='study.json')
    except BaseException as error:
        _write_json(run.path/'failure.json',dict(error=str(error),completed=results));run.finish('failed',error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return run.path
