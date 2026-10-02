"""Paired F-only versus CMF objective study and independent saved evidence audit."""
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import time
from typing import Any

import torch
from torch.nn import functional as F

from aster.benchmark.history_freshness import audit_freshness, build_freshness
from aster.benchmark.state_representation import digest
from aster.records.runlog import RunLog
from aster.training.history_reader import (HistoryReader, CLASSES, CONFIG, encode,
    target_ids, write_json, write_lines)
from aster.training.decision_fit import _state_digest

PROTOCOL = 'docs/HistoryFreshnessMinimal-v0.md'
EVALUATIONS = (0,10,20,30,40,48,49,50)


def source_identity(repo):
    paths = subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard'],cwd=repo,text=True).splitlines()
    return dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                git_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()),
                python_file_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest()
                                    for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()},
                protocol_sha256=hashlib.sha256((repo/PROTOCOL).read_bytes()).hexdigest())


def score(model, row):
    with torch.no_grad():
        logits = model(encode(row['input']))[0,2].tolist()
    if not all(math.isfinite(v) for v in logits):
        raise ValueError('Nonfinite F scores')
    target = CLASSES.index(row['facts']['F'])
    pred = max(range(3),key=lambda j:logits[j])
    top = max(logits)
    return dict(case_id=row['case_id'], cohort=row['cohort'], family=row['family'],
                input_sha256=digest(row['input']), logits=logits, target=target,
                prediction=pred, correct=pred==target,
                margin=logits[target]-max(v for j,v in enumerate(logits) if j!=target),
                nll=top+math.log(sum(math.exp(v-top) for v in logits))-logits[target])


def metrics(predictions):
    matrix = [[sum(p['target']==i and p['prediction']==j for p in predictions) for j in range(3)] for i in range(3)]
    return dict(count=len(predictions), correct=sum(p['correct'] for p in predictions),
                nll=sum(p['nll'] for p in predictions)/len(predictions),
                min_margin=min(p['margin'] for p in predictions), confusion=matrix)


def pair_scores(predictions, pairs):
    by_id = {p['case_id']:p for p in predictions}
    return [{**p,'both_correct':bool(by_id[p['left']]['correct'] and by_id[p['right']]['correct'])} for p in pairs]


def summarize(predictions,pairs):
    scored = pair_scores(predictions,pairs)
    return dict(cohorts={c:metrics([p for p in predictions if p['cohort']==c]) for c in sorted({p['cohort'] for p in predictions})},
                pairs={c:dict(count=sum(p['cohort']==c for p in scored),
                             both_correct=sum(p['cohort']==c and p['both_correct'] for p in scored)) for c in sorted({p['cohort'] for p in scored})},
                shape_pairs={c:dict(count=sum(p['cohort']+'/'+p['shape']==c for p in scored),
                                   both_correct=sum(p['cohort']+'/'+p['shape']==c and p['both_correct'] for p in scored))
                             for c in sorted({p['cohort']+'/'+p['shape'] for p in scored})})


def baselines(rows,pairs):
    train = [r for r in rows if r['cohort']=='train']
    def label(r):
        return CLASSES.index(r['facts']['F'])
    def tail(r):
        return json.loads(r['input'])['events'][-1]['action']['name']
    majority = max(range(3),key=lambda c:sum(label(r)==c for r in train))
    lookup = {name:max(range(3),key=lambda c:sum(tail(r)==name and label(r)==c for r in train)) for name in {tail(r) for r in train}}
    out = {}
    for name in ('majority','last_tool'):
        predictions = [dict(case_id=r['case_id'],cohort=r['cohort'],family=r['family'],target=label(r),
                            prediction=majority if name=='majority' else lookup.get(tail(r),majority),
                            correct=label(r)==(majority if name=='majority' else lookup.get(tail(r),majority))) for r in rows]
        ps = pair_scores(predictions,pairs)
        out[name] = dict(cohorts={c:dict(count=sum(p['cohort']==c for p in predictions),correct=sum(p['cohort']==c and p['correct'] for p in predictions)) for c in sorted({p['cohort'] for p in predictions})},
                         pairs={c:dict(count=sum(p['cohort']==c for p in ps),both_correct=sum(p['cohort']==c and p['both_correct'] for p in ps)) for c in sorted({p['cohort'] for p in ps})})
    return out


def run_study(repo: Path, root: Path, registration, debug=False):
    rows,pairs = build_freshness()
    if audit_freshness(rows,pairs)!=registration:
        raise ValueError('Freshness registration mismatch')
    source = source_identity(repo)
    if source['dirty'] and not debug:
        raise ValueError('Formal measurement needs clean source')
    epochs = 1 if debug else 50
    seeds = (42,) if debug else (42,43,44)
    protocol = dict(schema_version='aster-history-freshness-study-0', source=source, debug=debug,
                    seeds=seeds,epochs=epochs,lr=0.001,config=CONFIG,threads=torch.get_num_threads(),
                    objectives=['cmf','f_only'],registration_sha256=digest(registration))
    run = RunLog(root,'history_freshness_study',protocol,producer='trainer')
    write_json(run.path/'protocol.json',protocol)
    write_json(run.path/'registration.json',registration)
    write_lines(run.path/'cases.jsonl',rows)
    write_json(run.path/'pairs.json',pairs)
    baseline = baselines(rows,pairs)
    write_json(run.path/'baselines.json',baseline)
    train = [r for r in rows if r['cohort']=='train']
    prepared = [(encode(r['input']),target_ids(r['facts'])) for r in train]
    previous = signal.getsignal(signal.SIGALRM)
    results: list[dict[str,Any]] = []
    started = time.perf_counter()
    try:
        def expired(signum,frame):
            raise TimeoutError('Freshness arm exceeded 900 seconds')
        signal.signal(signal.SIGALRM,expired)
        for seed in seeds:
            torch.manual_seed(seed)
            parent = HistoryReader()
            initial = {k:v.clone() for k,v in parent.state_dict().items()}
            initial_hash = _state_digest(initial)
            torch.save(initial,run.path/f'initial-{seed}.pt')
            generator = torch.Generator().manual_seed(seed)
            orders = [torch.randperm(len(train),generator=generator).tolist() for _ in range(epochs)]
            write_json(run.path/f'order-{seed}.json',orders)
            for arm in ('cmf','f_only'):
                signal.setitimer(signal.ITIMER_REAL,900)
                arm_start = time.perf_counter()
                path = run.path/f'{arm}-{seed}'
                path.mkdir()
                model = HistoryReader()
                model.load_state_dict(initial)
                if _state_digest(model.state_dict())!=initial_hash:
                    raise ValueError('Paired initialization mismatch')
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001)
                curves,train_predictions = [],[]
                def evaluate(epoch):
                    model.eval()
                    before,rng = _state_digest(model.state_dict()),torch.get_rng_state().clone()
                    predictions = [dict(epoch=epoch,**score(model,r)) for r in train]
                    if before!=_state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                        raise ValueError('Train evaluation mutation')
                    curves.append(dict(epoch=epoch,updates=epoch*len(train),**metrics(predictions)))
                    train_predictions.extend(predictions)
                    write_lines(path/'curve.jsonl',curves)
                    write_lines(path/'train-predictions.jsonl',train_predictions)
                    print(f'{arm} seed={seed} epoch={epoch} F={curves[-1]["correct"]}/48',flush=True)
                evaluate(0)
                for epoch,order in enumerate(orders,1):
                    model.train()
                    for index in order:
                        ids,targets = prepared[index]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(ids)[0]
                        loss = F.cross_entropy(logits,targets) if arm=='cmf' else F.cross_entropy(logits[2:3],targets[2:3])
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite training loss')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        optimizer.step()
                    if epoch in EVALUATIONS or epoch==epochs:
                        evaluate(epoch)
                model.eval()
                before,rng = _state_digest(model.state_dict()),torch.get_rng_state().clone()
                final = [score(model,r) for r in rows]
                if before!=_state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                    raise ValueError('Final evaluation mutation')
                torch.save(model.state_dict(),path/'weights.pt')
                torch.save(optimizer.state_dict(),path/'optimizer.pt')
                write_json(path/'checkpoint.json',dict(config=CONFIG,classes=list(CLASSES),source=source,
                    registration_sha256=digest(registration),objective=arm,model_sha256=before,
                    initial_sha256=initial_hash,weights_sha256=hashlib.sha256((path/'weights.pt').read_bytes()).hexdigest(),resume_api=False))
                write_lines(path/'predictions.jsonl',final)
                write_json(path/'pairs.json',pair_scores(final,pairs))
                summary = dict(seed=seed,arm=arm,**summarize(final,pairs),
                    stable_fit=not debug and all(c['epoch'] in (48,49,50) and c['correct']==48 and c['min_margin']>0 for c in curves[-3:]),
                    initial_sha256=initial_hash,model_sha256=before,order_sha256=digest(orders),updates=epochs*48,
                    input_tokens=sum(ids.numel() for ids,_ in prepared)*epochs,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    weights_rng_unchanged=True,wall_seconds=time.perf_counter()-arm_start)
                write_json(path/'summary.json',summary)
                results.append(summary)
                signal.setitimer(signal.ITIMER_REAL,0)
                write_json(run.path/'partial-results.json',results)
        if source_identity(repo)!=source:
            raise ValueError('Source changed during measurement')
        study = dict(**protocol,run_id=run.id,models=results,baselines=baseline,
                     python=platform.python_version(),torch=torch.__version__,wall_seconds=time.perf_counter()-started,
                     test_constructed=0,test_scored=0,episode_scored=0,independent_holdout=False)
        write_json(run.path/'study.json',study)
        run.finish('completed',study='study.json')
    except BaseException as error:
        run.finish('failed',error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return run.path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_saved(path: Path):
    study = json.loads((path/'study.json').read_text())
    registration = json.loads((path/'registration.json').read_text())
    rows,pairs = read_lines(path/'cases.jsonl'),json.loads((path/'pairs.json').read_text())
    if audit_freshness(rows,pairs)!=registration or study['registration_sha256']!=digest(registration):
        raise ValueError('Saved registration mismatch')
    if baselines(rows,pairs)!=study['baselines']:
        raise ValueError('Baseline mismatch')
    reload_count = curve_count = 0
    lookup = {r['case_id']:r for r in rows}
    train = [r for r in rows if r['cohort']=='train']
    for summary in study['models']:
        arm,seed = summary['arm'],summary['seed']
        folder = path/f'{arm}-{seed}'
        meta = json.loads((folder/'checkpoint.json').read_text())
        if (meta['config']!=CONFIG or meta['classes']!=list(CLASSES) or meta['objective']!=arm
                or meta['source']!=study['source'] or meta['registration_sha256']!=digest(registration)
                or meta['weights_sha256']!=hashlib.sha256((folder/'weights.pt').read_bytes()).hexdigest()):
            raise ValueError('Checkpoint contract/hash mismatch')
        rng = torch.get_rng_state().clone()
        model = HistoryReader()
        model.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True))
        torch.set_rng_state(rng)
        model.eval()
        initial = torch.load(path/f'initial-{seed}.pt',map_location='cpu',weights_only=True)
        if _state_digest(initial)!=summary['initial_sha256'] or _state_digest(model.state_dict())!=meta['model_sha256'] or meta['model_sha256']!=summary['model_sha256']:
            raise ValueError('Model hash mismatch')
        cached = read_lines(folder/'predictions.jsonl')
        before = _state_digest(model.state_dict())
        if cached!=[score(model,r) for r in rows]:
            raise ValueError('Reload score mismatch')
        if before!=_state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
            raise ValueError('Audit evaluation mutation')
        if summarize(cached,pairs)!={k:summary[k] for k in ('cohorts','pairs','shape_pairs')} or pair_scores(cached,pairs)!=json.loads((folder/'pairs.json').read_text()):
            raise ValueError('Summary/pair mismatch')
        orders = json.loads((path/f'order-{seed}.json').read_text())
        if (len(orders)!=study['epochs'] or any(sorted(o)!=list(range(48)) for o in orders)
                or digest(orders)!=summary['order_sha256'] or summary['updates']!=48*study['epochs']
                or summary['input_tokens']!=sum(len(r['input'].encode())+2 for r in train)*study['epochs']):
            raise ValueError('Order/update/token mismatch')
        curves = read_lines(folder/'curve.jsonl')
        train_scores = read_lines(folder/'train-predictions.jsonl')
        for p in train_scores:
            if p['cohort']!='train' or p['target']!=CLASSES.index(lookup[p['case_id']]['facts']['F']) or p['input_sha256']!=digest(lookup[p['case_id']]['input']):
                raise ValueError('Train target/input mismatch')
            logits,t = p['logits'],p['target']
            prob = [math.exp(v-max(logits)) for v in logits]
            nll = -math.log(prob[t]/sum(prob))
            margin = min(logits[t]-v for j,v in enumerate(logits) if j!=t)
            predicted = max(range(3),key=lambda j:logits[j])
            if abs(nll-p['nll'])>1e-10 or abs(margin-p['margin'])>1e-10 or predicted!=p['prediction'] or p['correct']!=(predicted==t):
                raise ValueError('Train independent score mismatch')
        for c in curves:
            selected = [p for p in train_scores if p['epoch']==c['epoch']]
            if len(selected)!=48 or {k:c[k] for k in metrics(selected)}!=metrics(selected):
                raise ValueError('Curve mismatch')
        wanted_epochs = [e for e in EVALUATIONS if e<=study['epochs']]
        if study['epochs'] not in wanted_epochs:
            wanted_epochs.append(study['epochs'])
        if [c['epoch'] for c in curves]!=wanted_epochs:
            raise ValueError('Evaluation epoch mismatch')
        stable = not study['debug'] and all(c['epoch'] in (48,49,50) and c['correct']==48 and c['min_margin']>0 for c in curves[-3:])
        if stable!=summary['stable_fit']:
            raise ValueError('Stable fit mismatch')
        if [{k:v for k,v in p.items() if k!='epoch'} for p in train_scores[-48:]]!=[p for p in cached if p['cohort']=='train']:
            raise ValueError('Final curve/checkpoint mismatch')
        reload_count += len(cached)
        curve_count += len(train_scores)
    for seed in study['seeds']:
        a,b = [m for m in study['models'] if m['seed']==seed]
        if any(a[k]!=b[k] for k in ('initial_sha256','order_sha256','updates','input_tokens','parameters')):
            raise ValueError('Paired control mismatch')
    return dict(reloaded_predictions=reload_count,independently_scored_train_predictions=curve_count,
                executed_prefixes=len(rows),paired_models=len(study['models']),passed=True)
