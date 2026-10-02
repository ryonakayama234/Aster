"""Independently recompute cached metrics and replay all frozen-reader continuations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from audit_history_reader_training import validate_prediction
from aster.benchmark.history_contrasts import audit_contrasts, build_contrasts
from aster.benchmark.history_reader import read_facts, replay
from aster.benchmark.state_representation import digest
from aster.evaluator.episode import evaluate_episode
from aster.records.transition import Action
from aster.training.history_reader import (CLASSES, FACTS, LEGAL, execute_proposal,
    load_classifier, select_action)


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def independent_terminal(session):
    task, transitions = session.context.task, session.recorder.trajectory().transitions
    key = task['store_as']
    expected = task['left']+task['right'] if task['operation'] == 'add' else task['left']-task['right']
    puts = [i for i, t in enumerate(transitions) if t.observation.ok and t.action.name == 'memory.put'
            and t.action.arguments.get('key') == key]
    gets = [i for i, t in enumerate(transitions) if t.observation.ok and t.action.name == 'memory.get'
            and t.observation.output == {'key': key, 'value': expected}]
    return bool(transitions and transitions[-1].action.kind == 'stop' and session.context.memory.get(key) == expected
                and puts and gets and max(gets) > max(puts))


def scalar_selected_type(classifier, facts):
    if tuple(facts[k] for k in FACTS) not in LEGAL:
        return None
    weight, bias = classifier.weight.tolist(), classifier.bias.tolist()
    feature = [int(facts[k] is True) for k in FACTS]
    scores = [b+sum(w*x for w, x in zip(row, feature, strict=True)) for row, b in zip(weight, bias, strict=True)]
    return ('calculator', 'memory.put', 'memory.get', 'stop')[max(range(4), key=lambda j: scores[j])]


def audit_run(path):
    rows, pairs = lines(path/'cases.jsonl'), lines(path/'pairs.jsonl')
    if (rows, pairs) != build_contrasts():
        raise ValueError('Saved contrast cases/pairs changed')
    registration = audit_contrasts(rows, pairs)
    if registration != json.loads((path/'registration.json').read_text()):
        raise ValueError('Saved contrast registration changed')
    study = json.loads((path/'study.json').read_text())
    classifier, _ = load_classifier(path/'fixed-classifier')
    by_id = {r['case_id']: r for r in rows}
    count = dict(predictions=0, baseline_predictions=0, pairs=0, episodes=0, transitions=0, visited=0, oracle_episodes=0)
    for seed in (42, 43, 44):
        arm = path/f'seed-{seed}'
        summary = json.loads((arm/'summary.json').read_text())
        if summary != next(m for m in study['models'] if m['seed'] == seed):
            raise ValueError('Canonical model summary mismatch')
        baseline = lines(arm/'baseline-reload-predictions.jsonl')
        previous_predictions = lines(arm/'previous-final-predictions.jsonl')
        old_inputs = lines(path/'previous-cases.jsonl')
        if (len(baseline) != 120 or len(previous_predictions) != 120
                or [b['case_id'] for b in baseline] != [b['case_id'] for b in previous_predictions]
                or set(b['case_id'] for b in baseline) != set(r['case_id'] for r in old_inputs)):
            raise ValueError('Baseline coverage mismatch')
        for b, p in zip(baseline, previous_predictions, strict=True):
            validate_prediction(p)
            if b['previous_logits'] != p['logits'] or b['previous_predicted'] != p['predicted']:
                raise ValueError('Original baseline score changed')
            if not all(math.isfinite(x) for row in b['logits'] for x in row):
                raise ValueError('Nonfinite reloaded baseline')
            maximum = max(abs(x-y) for xs, ys in zip(b['logits'], p['logits'], strict=True)
                          for x, y in zip(xs, ys, strict=True))
            predicted = {k: CLASSES[max(range(3), key=lambda j: b['logits'][i][j])] for i, k in enumerate(FACTS)}
            if (maximum != b['max_absolute_difference'] or maximum > 1e-5
                    or predicted != b['predicted'] or predicted != p['predicted']):
                raise ValueError('Baseline tolerance/class mismatch')
        if (summary['previous_logits_absolute_tolerance'] != 1e-5
                or summary['previous_logits_within_tolerance'] != 120
                or summary['previous_fact_classes_matched'] != 120
                or summary['previous_logits_exact'] != sum(b['logits'] == b['previous_logits'] for b in baseline)
                or summary['previous_logits_max_absolute_difference'] != max(b['max_absolute_difference'] for b in baseline)):
            raise ValueError('Baseline tolerance summary mismatch')
        count['baseline_predictions'] += len(baseline)
        predictions = lines(arm/'predictions.jsonl')
        if len(predictions) != 40 or {p['case_id'] for p in predictions} != set(by_id):
            raise ValueError('Prediction coverage mismatch')
        for p in predictions:
            validate_prediction(p)
            row, session = by_id[p['case_id']], replay(by_id[p['case_id']])
            if p['truth'] != row['facts'] or p['input_sha256'] != digest(row['input']):
                raise ValueError('Prediction truth/input mismatch')
            teacher = session.next_action().to_dict()
            action, failure = select_action(classifier, p['predicted'], session)
            selected = None if action is None else action.to_dict()
            if (p['teacher'] != teacher or p['selected'] != selected or p['failure'] != failure
                    or p['action_correct'] != (selected == teacher)):
                raise ValueError('Cached action correctness mismatch')
            manual_type = scalar_selected_type(classifier, p['predicted'])
            if action is not None and (action.name if action.kind == 'tool' else 'stop') != manual_type:
                raise ValueError('Scalar classifier mismatch')
        pp = {p['case_id']: p for p in predictions}
        paired = lines(arm/'paired-predictions.jsonl')
        if len(paired) != 24 or {p['pair_id'] for p in paired} != {p['pair_id'] for p in pairs}:
            raise ValueError('Pair coverage mismatch')
        for p in paired:
            original = next(pair for pair in pairs if pair['pair_id'] == p['pair_id'])
            if any(p[k] != v for k, v in original.items()):
                raise ValueError('Pair identity mismatch')
            a, b = pp[p['left']], pp[p['right']]
            overlap = {m['case_id']: m['old_train_overlap'] for m in registration['manifest']}
            expected = dict(both_facts_correct=a['predicted'] == a['truth'] and b['predicted'] == b['truth'],
                            both_actions_correct=a['selected'] == a['teacher'] and b['selected'] == b['teacher'],
                            prediction_same=a['predicted'] == b['predicted'], truth_same=a['truth'] == b['truth'],
                            both_nontrain=not overlap[p['left']] and not overlap[p['right']])
            if any(p[k] != v for k, v in expected.items()):
                raise ValueError('Cached pair metric mismatch')
        for axis in ('rename', 'value', 'order'):
            for split in ('train', 'dev'):
                group = [p for p in paired if p['axis'] == axis and p['split'] == split]
                s = summary['pairs'][axis][split]
                if s['count'] != len(group) or any(s[k] != sum(p[k] for p in group)
                        for k in ('both_facts_correct', 'both_actions_correct', 'prediction_same')):
                    raise ValueError('Pair summary mismatch')
                if (s['both_nontrain_count'] != sum(p['both_nontrain'] for p in group)
                        or s['both_nontrain_facts_correct'] != sum(p['both_nontrain'] and p['both_facts_correct'] for p in group)):
                    raise ValueError('Pair nontrain summary mismatch')
        episodes = lines(arm/'episodes.jsonl')
        if len(episodes) != 40 or {e['case_id'] for e in episodes} != set(by_id):
            raise ValueError('Episode coverage mismatch')
        for ep in episodes:
            session = replay(by_id[ep['case_id']])
            failed_tool = False
            for v in ep['visited']:
                text = session.input_text()
                truth = read_facts(text)
                inferred = {k: CLASSES[max(range(3), key=lambda j: v['logits'][i][j])] for i, k in enumerate(FACTS)}
                if (v['input'] != text or v['input_sha256'] != digest(text) or v['truth'] != truth
                        or v['predicted'] != inferred or v['fact_correct'] != (inferred == truth)):
                    raise ValueError('Visited history/label/logits mismatch')
                action, failure = select_action(classifier, inferred, session)
                selected = None if action is None else action.to_dict()
                teacher = session.next_action().to_dict()
                if (v['selected'] != selected or v['teacher'] != teacher or v['failure'] != failure
                        or v['action_correct'] != (selected == teacher)):
                    raise ValueError('Visited selected action mismatch')
                if action is not None:
                    transition = execute_proposal(session, action)
                    failed_tool |= not transition.observation.ok
                count['visited'] += 1
            tr = [t.to_dict() for t in session.recorder.trajectory().transitions]
            if tr != ep['trajectory'] or evaluate_episode(session.recorder.trajectory()).to_dict() != ep['summary']:
                raise ValueError('Real-tool episode replay mismatch')
            first = next((i for i, v in enumerate(ep['visited']) if not v['fact_correct']), None)
            if first != ep['first_fact_error']:
                raise ValueError('First error mismatch')
            if ep['invalid'] == 'failed_tool_outside_reader_scope' and not failed_tool:
                raise ValueError('Missing real failed Tool observation')
            if ep['invalid'] == 'reader_context_exceeded' and len(session.input_text().encode())+2 <= 2048:
                raise ValueError('False context invalid')
            if ep['success'] != (None if ep['invalid'] else independent_terminal(session)):
                raise ValueError('Independent terminal mismatch')
            count['transitions'] += len(tr)
            count['episodes'] += 1
        for split in ('train', 'dev'):
            group = [p for p in predictions if p['split'] == split]
            s = summary['facts'][split]
            if s['count'] != len(group) or s['correct'] != sum(p['correct'] for p in group):
                raise ValueError('Fact summary mismatch')
            if abs(s['nll']-sum(p['nll'] for p in group)/len(group)) > 1e-10 or s['min_margin'] != min(min(p['margins']) for p in group):
                raise ValueError('Fact summary loss/margin mismatch')
            for fact in FACTS:
                confusion = [[sum(p['truth'][fact] is a and p['predicted'][fact] is b for p in group) for b in CLASSES] for a in CLASSES]
                actual = s['facts'][fact]
                tp, predicted_positive, actual_positive = confusion[1][1], sum(r[1] for r in confusion), sum(confusion[1])
                expected_precision = tp/predicted_positive if predicted_positive else None
                expected_recall = tp/actual_positive if actual_positive else None
                if (actual['confusion'] != confusion or actual['correct'] != sum(confusion[i][i] for i in range(3))
                        or actual['positive_precision'] != expected_precision or actual['positive_recall'] != expected_recall):
                    raise ValueError('Fact confusion/precision/recall mismatch')
            s = summary['actions'][split]
            if (s['count'] != len(group) or s['correct'] != sum(p['action_correct'] for p in group)
                    or s['failure_counts'] != dict(Counter(p['failure'] for p in group if p['failure']))):
                raise ValueError('Action summary mismatch')
            es = [e for e in episodes if e['split'] == split]
            expected = dict(count=len(es), success=sum(e['success'] is True for e in es),
                invalid=sum(e['invalid'] is not None for e in es),
                abstain=sum(e['failure'] == 'inconsistent_facts_abstain' for e in es),
                binding_failure=sum(e['failure'] == 'binding_failure' for e in es),
                wrong_stop=sum(any(v['selected'] and v['selected']['kind'] == 'stop' and not v['action_correct'] for v in e['visited']) for e in es),
                wrong_put=sum(any(v['selected'] and v['selected']['name'] == 'memory.put' and not v['action_correct'] for v in e['visited']) for e in es))
            if summary['episodes'][split] != expected:
                raise ValueError('Episode summary mismatch')
        count['predictions'] += len(predictions)
        count['pairs'] += len(paired)
    oracle = lines(path/'oracle-episodes.jsonl')
    if len(oracle) != 40 or {e['case_id'] for e in oracle} != set(by_id):
        raise ValueError('Oracle coverage mismatch')
    for ep in oracle:
        session = replay(by_id[ep['case_id']])
        for t in ep['trajectory'][len(by_id[ep['case_id']]['trajectory']):]:
            action = Action.from_dict(t['action'])
            if action != session.next_action() or execute_proposal(session, action).to_dict() != t:
                raise ValueError('Oracle real-tool replay mismatch')
        if not independent_terminal(session) or ep['success'] is not True:
            raise ValueError('Oracle independent terminal failed')
        count['oracle_episodes'] += 1
    if study['oracle'] != dict(count=40, success=40, invalid=0, abstain=0, binding_failure=0, wrong_stop=0, wrong_put=0):
        raise ValueError('Oracle summary mismatch')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), sort_keys=True))
