"""Checkpoint-start matched-budget order intervention, with immutable evaluation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import signal
import time
from typing import Any

import torch
from torch.nn import functional as F

from aster.benchmark.history_order import EPOCHS, SCHEMA, audit_order, build_order, training_slots
from aster.benchmark.history_reader import build_cases
from aster.benchmark.state_representation import digest
from aster.records.runlog import RunLog
from aster.training.decision_fit import _state_digest
from aster.training.history_contrasts import (MODEL_HASHES, PREVIOUS_RUN, checkpoint_files,
    compare_previous_score, lines, score_case)
from aster.training.history_reader import (CLASSIFIER_HASH, annotate, encode, episode_metrics,
    load_classifier, load_reader, metrics, predict, rollout, target_ids, write_json, write_lines)


def summarize(rows, pairs, predictions, episodes, registration):
    by_id = {p['case_id']:p for p in predictions}
    overlap = {m['case_id']:m['train_union_overlap'] for m in registration['manifest']}
    cohorts = sorted({r['cohort'] for r in rows})
    fact = {g:metrics([by_id[r['case_id']] for r in rows if r['cohort']==g]) for g in cohorts}
    action = {g:dict(count=fact[g]['count'], correct=sum(by_id[r['case_id']]['action_correct'] for r in rows if r['cohort']==g)) for g in cohorts}
    ep = {g:episode_metrics([e for e in episodes if next(r['cohort'] for r in rows if r['case_id']==e['case_id'])==g]) for g in cohorts}
    scored = []
    for pair in pairs:
        a,b = (by_id[pair[k]] for k in ('left','right'))
        scored.append({**pair, 'both_facts_correct':a['correct'] and b['correct'],
                       'both_actions_correct':a['action_correct'] and b['action_correct'],
                       'both_nontrain':not overlap[pair['left']] and not overlap[pair['right']],
                       'prediction_same':a['predicted']==b['predicted']})
    grouped = {}
    for p in scored:
        key = p['cohort']+'/'+p['shape']
        group = [q for q in scored if q['cohort']+'/'+q['shape']==key]
        grouped[key] = dict(count=len(group), both_facts_correct=sum(q['both_facts_correct'] for q in group),
                            both_actions_correct=sum(q['both_actions_correct'] for q in group),
                            both_nontrain_count=sum(q['both_nontrain'] for q in group),
                            both_nontrain_facts_correct=sum(q['both_nontrain'] and q['both_facts_correct'] for q in group))
    normal = [r for r in rows if r['cohort']=='old_train' and r['family']=='normal0']
    summary: dict[str, Any] = dict(facts=fact, actions=action, episodes=ep, pairs=grouped,
                normal_initial=dict(count=8,success=sum(e['success'] is True for e in episodes if e['case_id'] in {r['case_id'] for r in normal})))
    return summary, scored


def run_intervention(root: Path, previous: Path, registration: dict, source: dict):
    rows,pairs = build_order()
    if audit_order(rows,pairs) != registration:
        raise ValueError('Order registration mismatch')
    old_meta = json.loads((previous/'run.json').read_text())
    if old_meta['run_id'] != PREVIOUS_RUN or old_meta['status'] != 'completed':
        raise ValueError('Parent formal run mismatch')
    original_files = checkpoint_files(previous)
    classifier, meta = load_classifier(previous/'fixed-classifier')
    if meta['model_sha256'] != CLASSIFIER_HASH:
        raise ValueError('Classifier identity mismatch')
    classifier.requires_grad_(False)
    classifier.eval()
    classifier_before = _state_digest(classifier.state_dict())
    run = RunLog(root,'history_reader_order_intervention',source,producer='trainer')
    write_json(run.path/'registration.json',registration)
    write_json(run.path/'source.json',source)
    write_json(run.path/'source-checkpoint-files.json',original_files)
    write_lines(run.path/'cases.jsonl',rows)
    write_lines(run.path/'pairs.jsonl',pairs)
    shutil.copytree(previous/'fixed-classifier',run.path/'fixed-classifier')
    started = time.perf_counter()
    previous_signal = signal.getsignal(signal.SIGALRM)
    results = []
    try:
        def expired(signum, frame):
            raise TimeoutError('order_arm_900_second_deadline')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,900)
        oracle = [rollout(r,None,classifier) for r in rows]
        if not all(e['success'] is True for e in oracle):
            raise ValueError('Order oracle continuation failed')
        write_lines(run.path/'oracle-episodes.jsonl',oracle)
        old_rows = build_cases()
        if old_rows != lines(previous/'cases.jsonl'):
            raise ValueError('Parent input suite mismatch')
        for seed in MODEL_HASHES:
            parent = previous/f'seed-{seed}/checkpoint'
            model = load_reader(parent,CLASSIFIER_HASH)
            if _state_digest(model.state_dict()) != MODEL_HASHES[seed]:
                raise ValueError('Parent reader identity mismatch')
            seed_path = run.path/f'seed-{seed}'
            seed_path.mkdir()
            shutil.copytree(parent,seed_path/'parent-checkpoint')
            cached = lines(previous/f'seed-{seed}/final-predictions.jsonl')
            old_by_id = {r['case_id']:r for r in old_rows}
            baseline = []
            for p in cached:
                q = predict(model,old_by_id[p['case_id']]['input'])
                maximum = compare_previous_score(q,p)
                baseline.append(dict(case_id=p['case_id'],**q,previous_logits=p['logits'],max_absolute_difference=maximum))
            write_lines(seed_path/'parent-reload.jsonl',baseline)
            for name in ('frozen','repeat','order'):
                signal.setitimer(signal.ITIMER_REAL,900)
                arm_started = time.perf_counter()
                arm = seed_path/name
                arm.mkdir()
                model = load_reader(parent,CLASSIFIER_HASH)
                initial_hash = _state_digest(model.state_dict())
                torch.manual_seed(seed)
                curves, orders, curve_predictions = [], [], []
                total_tokens = 0
                if name != 'frozen':
                    slots = training_slots(rows,name)
                    prepared = [(encode(r['input']),target_ids(r['facts'])) for r in slots]
                    write_json(arm/'slots.json',[r['case_id'] for r in slots])
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001)
                    generator = torch.Generator().manual_seed(seed)
                    def evaluate(epoch):
                        model.eval()
                        predictions = [dict(case_id=r['case_id'],epoch=epoch,**annotate(predict(model,r['input']),r['facts'])) for r in slots]
                        curve_predictions.extend(predictions)
                        curves.append(dict(epoch=epoch,updates=epoch*112,**metrics(predictions)))
                        write_lines(arm/'learning-curve.jsonl',curves)
                        write_lines(arm/'train-predictions.jsonl',curve_predictions)
                    evaluate(0)
                    for epoch in range(1,EPOCHS+1):
                        model.train()
                        order = torch.randperm(112,generator=generator).tolist()
                        orders.append(dict(epoch=epoch,indices=order))
                        for index in order:
                            ids,targets = prepared[index]
                            optimizer.zero_grad(set_to_none=True)
                            loss = F.cross_entropy(model(ids)[0],targets)
                            if not bool(torch.isfinite(loss)):
                                raise ValueError('Nonfinite order loss')
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                            optimizer.step()
                            total_tokens += ids.numel()
                        if epoch % 10 == 0:
                            evaluate(epoch)
                            print(f'seed={seed} arm={name} epoch={epoch} slots={curves[-1]["correct"]}/112',flush=True)
                    write_json(arm/'order.json',orders)
                    torch.save(optimizer.state_dict(),arm/'optimizer.pt')
                model.eval()
                checkpoint = arm/'checkpoint'
                checkpoint.mkdir()
                torch.save(model.state_dict(),checkpoint/'weights.pt')
                manifest = json.loads((parent/'manifest.json').read_text())
                manifest.update(model_sha256=_state_digest(model.state_dict()),
                                weights_sha256=hashlib.sha256((checkpoint/'weights.pt').read_bytes()).hexdigest(),
                                initial_sha256=initial_hash,source=source,suite_sha256=registration['suite_sha256'],
                                intervention=SCHEMA,arm=name,parent_model_sha256=initial_hash)
                write_json(checkpoint/'manifest.json',manifest)
                before,rng = _state_digest(model.state_dict()),torch.get_rng_state().clone()
                predictions = [score_case(model,classifier,r) for r in rows]
                reloaded = load_reader(checkpoint,CLASSIFIER_HASH)
                if [predict(reloaded,r['input']) for r in rows] != [{k:p[k] for k in ('logits','predicted')} for p in predictions]:
                    raise ValueError('Order checkpoint reload mismatch')
                episodes = [rollout(r,model,classifier) for r in rows]
                if before != _state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                    raise ValueError('Order evaluation changed weights/RNG')
                summary,paired = summarize(rows,pairs,predictions,episodes,registration)
                summary.update(seed=seed,arm=name,parent_model_sha256=initial_hash,model_sha256=before,
                               updates=0 if name=='frozen' else EPOCHS*112,train_tokens=total_tokens,
                               train_class_counts=dict(Counter(canonical_facts(r['facts']) for r in training_slots(rows,name))) if name!='frozen' else {},
                               parent_max_logit_difference=max(p['max_absolute_difference'] for p in baseline),
                               wall_seconds=time.perf_counter()-arm_started,reload_matched=len(rows),evaluation_immutable=True)
                write_lines(arm/'predictions.jsonl',predictions)
                write_lines(arm/'episodes.jsonl',episodes)
                write_lines(arm/'paired-predictions.jsonl',paired)
                write_json(arm/'summary.json',summary)
                results.append(summary)
                write_json(run.path/'partial-results.json',results)
                print(f'seed={seed} arm={name} old={summary["facts"]["old_train"]["correct"]}/80 '
                      f'added={summary["facts"]["added_train"]["correct"]}/32 shape-dev={summary["facts"]["shape_dev"]["correct"]}/24',flush=True)
        if original_files != checkpoint_files(previous) or classifier_before != _state_digest(classifier.state_dict()):
            raise ValueError('Parent/classifier changed')
        study = dict(schema_version=SCHEMA,run_id=run.id,previous_run_id=PREVIOUS_RUN,source=source,
                     registration=registration,models=results,oracle=episode_metrics(oracle),
                     parent_checkpoint_files_unchanged=True,classifier_unchanged=True,
                     test_constructed=0,test_scored=0,independent_holdout=False,
                     python=platform.python_version(),torch=torch.__version__,threads=torch.get_num_threads(),
                     wall_seconds=time.perf_counter()-started)
        write_json(run.path/'study.json',study)
        run.finish('completed',study='study.json')
    except BaseException as error:
        run.finish('failed',error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_signal)
    return run.path


def canonical_facts(facts):
    return json.dumps(facts,sort_keys=True)
