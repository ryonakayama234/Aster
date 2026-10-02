"""Train a byte-history fact reader; keep the saved action classifier frozen."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import signal
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.benchmark.history_reader import (CONTEXT, INPUT_ID, audit, build_cases,
    feature_vector, read_facts, replay)
from aster.benchmark.state_representation import canonical, digest
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.runlog import RunLog
from aster.records.transition import Action, Observation, Transition
from aster.training.decision_action_types import TYPES, bind_type, load_checkpoint as load_classifier
from aster.training.decision_fit import _state_digest

SCHEMA = 'aster-history-reader-study-0'
CLASSES = (False, True, None)
FACTS = ('C', 'M', 'F')
CLASSIFIER_HASH = 'de12ec2dedfac982d3085ccb24540b26fcf20eecb1d0cf06f99c31aad9efea32'
LEGAL = {(False, None, None), (True, False, None), (True, True, None),
         (True, True, True), (True, False, False), (True, True, False)}
CONFIG = dict(vocab_size=258, context_length=CONTEXT, width=16, heads=2, layers=1)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')


def write_lines(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(canonical(row)+'\n')


class HistoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyLM(ModelConfig(**CONFIG))
        self.backbone.lm_head.requires_grad_(False)
        self.head = nn.Linear(CONFIG['width'], 9)
        TinyLM._initialize(self.head)

    def forward(self, ids):
        return self.head(self.backbone.encode(ids)[:, -1]).reshape(-1, 3, 3)


def encode(text):
    raw = list(text.encode('utf-8'))
    if len(raw)+2 > CONTEXT:
        raise ValueError('reader_context_exceeded')
    if bytes(raw).decode('utf-8') != text:
        raise ValueError('reader_byte_roundtrip_failed')
    return torch.tensor([[256, *raw, 257]], dtype=torch.long)


def target_ids(facts):
    return torch.tensor([CLASSES.index(facts[k]) for k in FACTS], dtype=torch.long)


def predict(model, text):
    with torch.no_grad():
        logits = model(encode(text))[0]
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite_reader_logits')
    return dict(logits=logits.tolist(), predicted={k: CLASSES[int(logits[i].argmax())] for i, k in enumerate(FACTS)})


def annotate(prediction, truth):
    logits = torch.tensor(prediction['logits'], dtype=torch.float64)
    targets = target_ids(truth)
    margin = [float(logits[i, t]-max(logits[i, j] for j in range(3) if j != int(t))) for i, t in enumerate(targets)]
    return {**prediction, 'truth': truth, 'correct': prediction['predicted'] == truth,
            'nll': float(F.cross_entropy(logits, targets)), 'margins': margin}


def metrics(rows):
    fact_metrics = {}
    for k in FACTS:
        confusion = [[sum(r['truth'][k] is CLASSES[t] and r['predicted'][k] is CLASSES[p] for r in rows)
                      for p in range(3)] for t in range(3)]
        tp = confusion[1][1]
        pp, actual = sum(v[1] for v in confusion), sum(confusion[1])
        fact_metrics[k] = dict(confusion=confusion, correct=sum(confusion[i][i] for i in range(3)),
                              positive_precision=tp/pp if pp else None, positive_recall=tp/actual if actual else None)
    return dict(count=len(rows), correct=sum(r['correct'] for r in rows),
                nll=sum(r['nll'] for r in rows)/len(rows), min_margin=min(min(r['margins']) for r in rows),
                facts=fact_metrics)


def select_action(classifier, facts, session):
    if tuple(facts[k] for k in FACTS) not in LEGAL:
        return None, 'inconsistent_facts_abstain'
    with torch.no_grad():
        scores = classifier(torch.tensor(feature_vector(facts), dtype=torch.float32))
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('nonfinite_classifier_logits')
    name = TYPES[int(scores.argmax())]
    history = session.recorder.trajectory()
    state = session.context.snapshot(len(history))
    candidates = CalculateAndStoreCandidates(goal_contract=CURRENT_GOAL_V1).build(state, history, session.available)
    try:
        return bind_type(name, candidates), None
    except ValueError:
        return None, 'binding_failure'


def terminal_check(session):
    """Independent stop-time state and order check, no Evaluator flags used."""
    task = session.context.task
    expected = task['left']+task['right'] if task['operation']=='add' else task['left']-task['right']
    last_put, last_get = -1, -1
    for i, t in enumerate(session.recorder.trajectory().transitions):
        if t.action.arguments.get('key') != task['store_as'] or not t.observation.ok:
            continue
        if t.action.name == 'memory.put':
            last_put = i
        elif t.action.name == 'memory.get' and t.observation.output == {'key':task['store_as'], 'value':expected}:
            last_get = i
    return bool(session.recorder.trajectory().stopped and session.context.memory.get(task['store_as']) == expected
                and 0 <= last_put < last_get)


def execute_proposal(session, action):
    history = session.recorder.trajectory()
    before = session.context.snapshot(len(history))
    obs = (Observation(True, True, {'stopped': True, 'reason': action.arguments.get('reason')})
           if action.kind == 'stop' else session.tools.execute(action, session.context))
    after = session.context.snapshot(len(history)+1)
    transition = Transition(len(history), before.to_dict(), session.available, action, obs, after.to_dict(),
                            session.evaluator.evaluate(session.context.task, after, action, obs, history))
    session.recorder.record(transition)
    return transition


def rollout(row, model, classifier):
    session = replay(row)
    visited, failure, invalid = [], None, None
    for _ in range(8):
        text = session.input_text()
        # Truth is annotation only. Action selection receives prediction (or oracle arm) explicitly.
        truth = read_facts(text)
        try:
            prediction = predict(model, text) if model is not None else {'predicted':truth, 'logits':None}
        except ValueError as error:
            invalid = str(error)
            break
        action, failure = select_action(classifier, prediction['predicted'], session)
        gold_action = session.next_action()
        visited.append(dict(input=text, input_sha256=digest(text), truth=truth, **prediction,
                            selected=None if action is None else action.to_dict(),
                            teacher=gold_action.to_dict(), fact_correct=prediction['predicted']==truth,
                            action_correct=action==gold_action, failure=failure))
        if action is None:
            break
        transition = execute_proposal(session, action)
        if not transition.observation.ok:
            invalid = 'failed_tool_outside_reader_scope'
            break
        if transition.evaluation.terminal:
            break
    independent = terminal_check(session)
    summary = evaluate_episode(session.recorder.trajectory()).to_dict()
    if independent != summary['task_success']:
        raise ValueError('Independent/evaluator terminal mismatch')
    first = next((i for i, v in enumerate(visited) if not v['fact_correct']), None)
    return dict(case_id=row['case_id'], split=row['split'], family=row['family'],
                success=None if invalid else independent, failure=failure, invalid=invalid,
                first_fact_error=first, visited=visited, summary=summary,
                trajectory=[t.to_dict() for t in session.recorder.trajectory().transitions],
                prefix_length=len(row['trajectory']), fallback=False)


def load_reader(path, classifier_hash):
    meta = json.loads((path/'manifest.json').read_text())
    expected = dict(input_id=INPUT_ID, classes=list(CLASSES), facts=list(FACTS), config=CONFIG,
                    goal_contract=CURRENT_GOAL_V1, classifier_model_sha256=classifier_hash)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Reader checkpoint contract mismatch')
    if hashlib.sha256((path/'weights.pt').read_bytes()).hexdigest() != meta['weights_sha256']:
        raise ValueError('Reader checkpoint weights mismatch')
    rng = torch.get_rng_state().clone()
    model = HistoryReader()
    torch.set_rng_state(rng)
    model.load_state_dict(torch.load(path/'weights.pt', map_location='cpu', weights_only=True))
    if _state_digest(model.state_dict()) != meta['model_sha256']:
        raise ValueError('Reader model identity mismatch')
    model.eval()
    return model


def episode_metrics(episodes):
    return dict(count=len(episodes), success=sum(e['success'] is True for e in episodes),
                invalid=sum(e['invalid'] is not None for e in episodes),
                abstain=sum(e['failure']=='inconsistent_facts_abstain' for e in episodes),
                binding_failure=sum(e['failure']=='binding_failure' for e in episodes),
                wrong_stop=sum(any(v['selected'] and v['selected']['kind']=='stop' and not v['action_correct'] for v in e['visited']) for e in episodes),
                wrong_put=sum(any(v['selected'] and v['selected']['name']=='memory.put' and not v['action_correct'] for v in e['visited']) for e in episodes))


def run_study(root, classifier_path, registration, source, *, debug=False):
    rows = build_cases()
    actual: dict[str, Any] = audit(rows)
    for key in ('suite_sha256', 'manifest_sha256', 'counts', 'input_id', 'goal_contract', 'context_length'):
        if actual[key] != registration[key]:
            raise ValueError('Reader registration mismatch: '+key)
    classifier, classifier_meta = load_classifier(classifier_path)
    if classifier_meta['model_sha256'] != CLASSIFIER_HASH:
        raise ValueError('Frozen classifier identity mismatch')
    classifier.requires_grad_(False)
    classifier_before = _state_digest(classifier.state_dict())
    seeds, epochs = ((42,), 1) if debug else ((42,43,44), 50)
    protocol = dict(schema_version=SCHEMA, seeds=seeds, epochs=epochs, debug=debug, lr=0.001,
                    config=CONFIG, threads=torch.get_num_threads(), source=source,
                    suite_sha256=actual['suite_sha256'], classifier_model_sha256=CLASSIFIER_HASH,
                    optimizer='AdamW defaults', clip_norm=1.0, classes=list(CLASSES), input_id=INPUT_ID)
    run = RunLog(root, 'history_reader_study', protocol, producer='trainer')
    write_json(run.path/'protocol.json', protocol)
    write_lines(run.path/'cases.jsonl', rows)
    write_json(run.path/'registration.json', actual)
    shutil.copytree(classifier_path, run.path/'fixed-classifier')
    train = [r for r in rows if r['split']=='train']
    prepared = [(encode(r['input']), target_ids(r['facts'])) for r in train]
    started = time.perf_counter()
    previous = signal.getsignal(signal.SIGALRM)
    results = []
    try:
        def expired(signum, frame):
            raise TimeoutError('reader_seed_900_second_deadline')
        signal.signal(signal.SIGALRM, expired)
        oracle = [rollout(r, None, classifier) for r in rows]
        if not all(e['success'] is True for e in oracle):
            raise ValueError('Frozen classifier oracle control failed')
        write_lines(run.path/'oracle-episodes.jsonl', oracle)
        majority = {k: max(CLASSES, key=lambda c: sum(r['facts'][k] is c for r in train)) for k in FACTS}
        baseline = {split: dict(count=sum(r['split']==split for r in rows),
                               correct=sum(r['split']==split and r['facts']==majority for r in rows)) for split in ('train','dev')}
        write_json(run.path/'majority-baseline.json', dict(predicted=majority, metrics=baseline))
        for seed in seeds:
            signal.setitimer(signal.ITIMER_REAL,900)
            arm_started = time.perf_counter()
            path = run.path/f'seed-{seed}'
            path.mkdir()
            torch.manual_seed(seed)
            model = HistoryReader()
            initial_hash = _state_digest(model.state_dict())
            torch.save(model.state_dict(), path/'initial-weights.pt')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
            generator = torch.Generator().manual_seed(seed)
            curves: list[dict[str, Any]] = []
            predictions: list[dict[str, Any]] = []
            orders: list[dict[str, Any]] = []
            def evaluate(epoch):
                model.eval()
                before, rng = _state_digest(model.state_dict()), torch.get_rng_state().clone()
                scored = [dict(case_id=r['case_id'], epoch=epoch, **annotate(predict(model, r['input']), r['facts'])) for r in train]
                if before != _state_digest(model.state_dict()) or not torch.equal(rng, torch.get_rng_state()):
                    raise ValueError('Reader train evaluation changed weights/RNG')
                predictions.extend(scored)
                curves.append(dict(epoch=epoch, updates=epoch*80, **metrics(scored)))
                write_lines(path/'learning-curve.jsonl', curves)
                write_lines(path/'train-predictions.jsonl', predictions)
                print(f'seed={seed} epoch={epoch} train={curves[-1]["correct"]}/80', flush=True) if epoch % 10 == 0 else None
            evaluate(0)
            for epoch in range(1, epochs+1):
                model.train()
                order = torch.randperm(80, generator=generator).tolist()
                orders.append(dict(epoch=epoch, indices=order))
                write_json(path/'order.json', orders)
                for index in order:
                    ids, targets = prepared[index]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(ids)[0], targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite_reader_loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()
                evaluate(epoch)
            checkpoint = path/'checkpoint'
            checkpoint.mkdir()
            torch.save(model.state_dict(), checkpoint/'weights.pt')
            manifest = dict(input_id=INPUT_ID, classes=list(CLASSES), facts=list(FACTS), config=CONFIG,
                            goal_contract=CURRENT_GOAL_V1, classifier_model_sha256=CLASSIFIER_HASH,
                            model_sha256=_state_digest(model.state_dict()), initial_sha256=initial_hash,
                            weights_sha256=hashlib.sha256((checkpoint/'weights.pt').read_bytes()).hexdigest(),
                            source=source, suite_sha256=actual['suite_sha256'], resume_supported=False)
            write_json(checkpoint/'manifest.json', manifest)
            model.eval()
            before, rng = _state_digest(model.state_dict()), torch.get_rng_state().clone()
            reloaded = load_reader(checkpoint, CLASSIFIER_HASH)
            final = [dict(case_id=r['case_id'], split=r['split'], family=r['family'],
                          **annotate(predict(model,r['input']),r['facts'])) for r in rows]
            if [predict(model,r['input']) for r in rows] != [predict(reloaded,r['input']) for r in rows]:
                raise ValueError('Reader reload logits mismatch')
            if [predict(model,r['input']) for r in train] != [{k:v[k] for k in ('logits','predicted')} for v in predictions[-80:]]:
                raise ValueError('Final train cached score mismatch')
            episodes = [rollout(r, model, classifier) for r in rows]
            if before != _state_digest(model.state_dict()) or classifier_before != _state_digest(classifier.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                raise ValueError('Reader/classifier evaluation changed weights/RNG')
            by_id = {r['case_id']:r for r in final}
            pairs = [{**p, 'both_correct':by_id[p['left']]['correct'] and by_id[p['right']]['correct']}
                     for p in actual['pairs'] if p['left'].startswith('dev/')]
            summary: dict[str, Any] = dict(seed=seed, stable_fit=len(curves)>=3 and all(c['correct']==80 and c['min_margin']>0 for c in curves[-3:]),
                           train=curves[-1], dev=metrics([r for r in final if r['split']=='dev']),
                           pairs_correct=sum(p['both_correct'] for p in pairs), pairs_count=len(pairs),
                           episodes={split:episode_metrics([e for e in episodes if e['split']==split]) for split in ('train','dev')},
                           initial={split:episode_metrics([e for e in episodes if e['split']==split and e['family']=='normal0']) for split in ('train','dev')},
                           common_valid_dev=sum(e['split']=='dev' and e['invalid'] is None for e in episodes),
                           dev_by_family={family:dict(count=sum(r['split']=='dev' and r['family']==family for r in final),
                                                     correct=sum(r['split']=='dev' and r['family']==family and r['correct'] for r in final))
                                          for family in sorted({r['family'] for r in rows})},
                           parameter_count=sum(p.numel() for p in model.parameters()),
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           model_sha256=before, initial_sha256=initial_hash, order_sha256=digest(orders),
                           reload_match=True, weights_rng_unchanged=True, wall_seconds=time.perf_counter()-arm_started)
            write_lines(path/'final-predictions.jsonl', final)
            write_lines(path/'episodes.jsonl', episodes)
            write_lines(path/'pairs.jsonl', pairs)
            write_json(path/'summary.json', summary)
            results.append(summary)
            write_json(run.path/'partial-results.json', results)
            signal.setitimer(signal.ITIMER_REAL,0)
            print(f'seed={seed} fit={summary["stable_fit"]} dev={summary["dev"]["correct"]}/40 prefix={summary["episodes"]["dev"]["success"]}/40', flush=True)
        study = dict(**protocol, run_id=run.id, models=results, oracle=episode_metrics(oracle),
                     majority=baseline, python=platform.python_version(), torch=torch.__version__,
                     wall_seconds=time.perf_counter()-started, test_constructed=0, test_scored=0,
                     structural_holdout=False)
        write_json(run.path/'study.json', study)
        run.finish('completed', study='study.json')
    except BaseException as error:
        run.finish('failed', error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return run.path
