"""Read-only evaluation of saved history readers on preregistered contrasts."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import signal
import time
from typing import Any

import torch

from aster.benchmark.history_contrasts import SCHEMA, audit_contrasts, build_contrasts
from aster.benchmark.history_reader import build_cases, replay
from aster.benchmark.state_representation import digest
from aster.records.runlog import RunLog
from aster.training.decision_fit import _state_digest
from aster.training.history_reader import (CLASSIFIER_HASH, annotate, episode_metrics,
    load_classifier, load_reader, metrics, predict, rollout, select_action, write_json, write_lines)

PREVIOUS_RUN = '1cc57fcd7b6e41819bdd95d35c272003'
MODEL_HASHES = {
    42: '73757f4e5391e2e7ddaf6a28d997365151698137c70012368bd6b22273dff9e8',
    43: '5045b6db50f260a2b26fbfe23cef96ad230ab8c358980687cc1d4b67f3a41461',
    44: '49a1353b83c95079bbdd5dd7380280663945fde9108a5eec1ddcef8ce3e646b4',
}
PREVIOUS_LOGIT_ATOL = 1e-5


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def verify_registration(actual, registered):
    if actual != registered:
        raise ValueError('Frozen contrast registration mismatch')


def compare_previous_score(current, previous):
    if (len(current['logits']) != 3 or len(previous['logits']) != 3
            or any(len(r) != 3 for r in current['logits']+previous['logits'])
            or not all(math.isfinite(x) for r in current['logits']+previous['logits'] for x in r)):
        raise ValueError('Frozen reader previous nonfinite/shape mismatch')
    differences = [abs(x-y) for a, b in zip(current['logits'], previous['logits'], strict=True)
                   for x, y in zip(a, b, strict=True)]
    maximum = max(differences)
    if maximum > PREVIOUS_LOGIT_ATOL or current['predicted'] != previous['predicted']:
        raise ValueError('Frozen reader previous logits/class mismatch')
    return maximum


def checkpoint_files(path):
    names = ['fixed-classifier/manifest.json', 'fixed-classifier/weights.pt']
    for seed in MODEL_HASHES:
        names += [f'seed-{seed}/checkpoint/{name}' for name in ('manifest.json', 'weights.pt')]
    return {name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in names}


def summarize_pairs(pairs, predictions, manifest):
    lookup = {p['case_id']: p for p in predictions}
    overlap = {m['case_id']: m['old_train_overlap'] for m in manifest}
    scored = []
    for pair in pairs:
        a, b = (lookup[pair[k]] for k in ('left', 'right'))
        scored.append({**pair, 'both_facts_correct': a['correct'] and b['correct'],
                       'both_actions_correct': a['action_correct'] and b['action_correct'],
                       'prediction_same': a['predicted'] == b['predicted'],
                       'truth_same': a['truth'] == b['truth'],
                       'both_nontrain': not overlap[pair['left']] and not overlap[pair['right']]})
    summaries = {}
    for axis in ('rename', 'value', 'order'):
        summaries[axis] = {}
        for split in ('train', 'dev'):
            chosen = [p for p in scored if p['axis'] == axis and p['split'] == split]
            summaries[axis][split] = dict(count=len(chosen),
                both_facts_correct=sum(p['both_facts_correct'] for p in chosen),
                both_actions_correct=sum(p['both_actions_correct'] for p in chosen),
                prediction_same=sum(p['prediction_same'] for p in chosen),
                both_nontrain_count=sum(p['both_nontrain'] for p in chosen),
                both_nontrain_facts_correct=sum(p['both_nontrain'] and p['both_facts_correct'] for p in chosen))
    return scored, summaries


def score_case(model, classifier, row) -> dict[str, Any]:
    scored = annotate(predict(model, row['input']), row['facts'])
    session = replay(row)
    teacher = session.next_action()
    action, failure = select_action(classifier, scored['predicted'], session)
    return dict(case_id=row['case_id'], split=row['split'], family=row['family'],
                input_sha256=digest(row['input']), **scored,
                teacher=teacher.to_dict(), selected=None if action is None else action.to_dict(),
                action_correct=action == teacher, failure=failure)


def run_diagnostic(root: Path, previous: Path, registration: dict, source: dict):
    rows, pairs = build_contrasts()
    actual = audit_contrasts(rows, pairs)
    verify_registration(actual, registration)
    old_metadata = json.loads((previous/'run.json').read_text())
    if old_metadata['run_id'] != PREVIOUS_RUN or old_metadata['status'] != 'completed':
        raise ValueError('Previous formal Run mismatch')
    file_hashes = checkpoint_files(previous)
    rng_start = torch.get_rng_state().clone()
    classifier, meta = load_classifier(previous/'fixed-classifier')
    torch.set_rng_state(rng_start)
    if meta['model_sha256'] != CLASSIFIER_HASH:
        raise ValueError('Frozen classifier identity mismatch')
    classifier.requires_grad_(False)
    classifier.eval()
    classifier_before = _state_digest(classifier.state_dict())
    run = RunLog(root, 'history_reader_frozen_contrasts', source, producer='evaluator')
    write_json(run.path/'registration.json', actual)
    write_json(run.path/'source.json', source)
    write_lines(run.path/'cases.jsonl', rows)
    write_lines(run.path/'pairs.jsonl', pairs)
    shutil.copyfile(previous/'cases.jsonl', run.path/'previous-cases.jsonl')
    write_json(run.path/'source-checkpoint-files.json', file_hashes)
    shutil.copytree(previous/'fixed-classifier', run.path/'fixed-classifier')
    started, previous_signal = time.perf_counter(), signal.getsignal(signal.SIGALRM)
    results = []
    try:
        def expired(signum, frame):
            raise TimeoutError('frozen_reader_300_second_deadline')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 300)
        oracle = [rollout(r, None, classifier) for r in rows]
        if not all(e['success'] is True for e in oracle):
            raise ValueError('Frozen contrast oracle continuation failed')
        write_lines(run.path/'oracle-episodes.jsonl', oracle)
        old_rows = build_cases()
        if old_rows != lines(previous/'cases.jsonl'):
            raise ValueError('Previous saved suite mismatch')
        for seed, wanted_hash in MODEL_HASHES.items():
            signal.setitimer(signal.ITIMER_REAL, 300)
            seed_started = time.perf_counter()
            arm = run.path/f'seed-{seed}'
            arm.mkdir()
            checkpoint = previous/f'seed-{seed}/checkpoint'
            manifest = json.loads((checkpoint/'manifest.json').read_text())
            if manifest['model_sha256'] != wanted_hash:
                raise ValueError('Frozen reader identity mismatch')
            model = load_reader(checkpoint, CLASSIFIER_HASH)
            model.requires_grad_(False)
            before, rng = _state_digest(model.state_dict()), torch.get_rng_state().clone()
            cached = lines(previous/f'seed-{seed}/final-predictions.jsonl')
            if len(cached) != 120 or {p['case_id'] for p in cached} != {r['case_id'] for r in old_rows}:
                raise ValueError('Previous prediction coverage mismatch')
            old_by_id = {r['case_id']: r for r in old_rows}
            baseline_reload = []
            for p in cached:
                q = predict(model, old_by_id[p['case_id']]['input'])
                maximum = compare_previous_score(q, p)
                baseline_reload.append(dict(case_id=p['case_id'], **q, previous_logits=p['logits'],
                    previous_predicted=p['predicted'], max_absolute_difference=maximum))
            write_lines(arm/'baseline-reload-predictions.jsonl', baseline_reload)
            shutil.copyfile(previous/f'seed-{seed}/final-predictions.jsonl', arm/'previous-final-predictions.jsonl')
            predictions = [score_case(model, classifier, r) for r in rows]
            reloaded = load_reader(checkpoint, CLASSIFIER_HASH)
            if [predict(reloaded, r['input']) for r in rows] != [
                    {k: p[k] for k in ('logits', 'predicted')} for p in predictions]:
                raise ValueError('Frozen reader reload mismatch')
            episodes = [rollout(r, model, classifier) for r in rows]
            if (before != _state_digest(model.state_dict())
                    or classifier_before != _state_digest(classifier.state_dict())
                    or not torch.equal(rng, torch.get_rng_state())):
                raise ValueError('Frozen evaluation changed weights/RNG')
            paired, summaries = summarize_pairs(pairs, predictions, actual['manifest'])
            summary: dict[str, Any] = dict(seed=seed, model_sha256=before, previous_logits_within_tolerance=120,
                previous_logits_exact=sum(p['logits'] == p['previous_logits'] for p in baseline_reload),
                previous_logits_max_absolute_difference=max(p['max_absolute_difference'] for p in baseline_reload),
                previous_logits_absolute_tolerance=PREVIOUS_LOGIT_ATOL,
                previous_fact_classes_matched=120,
                reload_matched=40, weights_rng_unchanged=True, pairs=summaries,
                facts={s: metrics([p for p in predictions if p['split'] == s]) for s in ('train', 'dev')},
                actions={s: dict(count=sum(p['split'] == s for p in predictions),
                                correct=sum(p['split'] == s and p['action_correct'] for p in predictions),
                                failure_counts=dict(Counter(p['failure'] for p in predictions
                                                            if p['split'] == s and p['failure'])))
                         for s in ('train', 'dev')},
                episodes={s: episode_metrics([e for e in episodes if e['split'] == s]) for s in ('train', 'dev')},
                wall_seconds=time.perf_counter()-seed_started)
            shutil.copytree(checkpoint, arm/'checkpoint')
            write_lines(arm/'predictions.jsonl', predictions)
            write_lines(arm/'paired-predictions.jsonl', paired)
            write_lines(arm/'episodes.jsonl', episodes)
            write_json(arm/'summary.json', summary)
            results.append(summary)
            write_json(run.path/'partial-results.json', results)
            print(f'seed={seed} facts={summary["facts"]["dev"]["correct"]}/20 '
                  f'prefix={summary["episodes"]["dev"]["success"]}/20', flush=True)
        if file_hashes != checkpoint_files(previous) or not torch.equal(rng_start, torch.get_rng_state()):
            raise ValueError('Source checkpoints/RNG changed')
        result = dict(schema_version=SCHEMA, run_id=run.id, previous_run_id=PREVIOUS_RUN,
            source=source, registration=actual, source_checkpoint_files=file_hashes,
            models=results, oracle=episode_metrics(oracle), model_updates=0,
            calibration_updates=0, test_constructed=0, test_scored=0, fallback=False,
            structural_holdout=False, python=platform.python_version(), torch=torch.__version__,
            threads=torch.get_num_threads(), wall_seconds=time.perf_counter()-started)
        write_json(run.path/'study.json', result)
        run.finish('completed', study='study.json')
    except BaseException as error:
        run.finish('failed', error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_signal)
    return run.path
