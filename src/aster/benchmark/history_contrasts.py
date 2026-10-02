"""Equal-length, executed diagnostic contrasts for frozen history readers (no torch)."""
from collections import Counter
from copy import deepcopy
import json

from aster.benchmark.history_reader import (
    CONTEXT, INPUT_ID, TASKS, ExecutedHistory, build_cases, independent_completion,
    make_prefix, read_facts, replay,
)
from aster.benchmark.state_representation import canonical, digest
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.records.decision_relations import observable_relations
from aster.records.transition import Action

SCHEMA = 'aster-history-frozen-contrasts-0'
AXES = ('rename', 'value', 'order')


def rename_key(key):
    return ''.join(chr((ord(c)-97+1) % 26+97) if 'a' <= c <= 'z' else c for c in key)


def wrong_value(value):
    for candidate in (value+1, value-1):
        if candidate != value and len(str(candidate)) == len(str(value)) and (candidate < 0) == (value < 0):
            return candidate
    raise ValueError('No same-width/same-sign wrong value')


def as_row(session, case_id, split, group, family):
    text = session.input_text()
    return dict(case_id=case_id, split=split, leakage_group=group, family=family,
                task=deepcopy(session.context.task), input=text, input_sha256=digest(text),
                facts=read_facts(text),
                trajectory=[t.to_dict() for t in session.recorder.trajectory().transitions])


def build_contrasts():
    rows, pairs = {}, []
    selections = dict(train=(0, 1, 4, 5), dev=(0, 1, 2, 3))
    for split, indices in selections.items():
        for index in indices:
            op, left, right, key = TASKS[split][index]
            task = dict(kind='calculate_and_store', operation=op, left=left, right=right, store_as=key)
            group = f'{split}/task{index}'
            normal = make_prefix(task, 'normal3')
            rewritten = deepcopy(task)
            rewritten['store_as'] = rename_key(key)
            renamed = make_prefix(rewritten, 'normal3')
            same = make_prefix(task, 'same_write')
            wrong = make_prefix(task, 'normal3')
            wrong.execute(Action.tool('memory.put', key=key, value=wrong_value(wrong.context.memory[key])))
            fresh = make_prefix(task, 'normal2')
            fresh.execute(Action.tool('memory.put', key=key, value=fresh.context.memory[key]))
            fresh.execute(Action.tool('memory.get', key=key))
            for family, session in [('normal', normal), ('renamed', renamed), ('same', same),
                                    ('wrong', wrong), ('fresh', fresh)]:
                row = as_row(session, f'{group}/{family}', split, group, family)
                rows[row['case_id']] = row
            for axis, a, b in [('rename', 'normal', 'renamed'), ('value', 'same', 'wrong'),
                               ('order', 'same', 'fresh')]:
                pairs.append(dict(pair_id=f'{group}/{axis}', axis=axis, split=split,
                                  leakage_group=group, left=f'{group}/{a}', right=f'{group}/{b}'))
    return list(rows.values()), pairs


def expected_facts(family):
    return {'C': True, 'M': family != 'wrong', 'F': family in ('normal', 'renamed', 'fresh')}


def action_type(action):
    return action.name if action.kind == 'tool' else 'stop'


def audit_contrasts(rows, pairs):
    if (len(rows), len(pairs)) != (40, 24):
        raise ValueError('Contrast counts mismatch')
    expected_rows, expected_pairs = build_contrasts()
    if rows != expected_rows or pairs != expected_pairs:
        raise ValueError('Contrast registration generation mismatch')
    by_id = {r['case_id']: r for r in rows}
    if len(by_id) != 40 or len({r['input_sha256'] for r in rows}) != 40:
        raise ValueError('Duplicate contrast ID/input')
    old = build_cases()
    old_train = {digest(r['input']) for r in old if r['split'] == 'train'}
    old_dev = {digest(r['input']) for r in old if r['split'] == 'dev'}
    labels, lengths, manifest = {}, [], []
    for row in rows:
        session = replay(row)
        truth = read_facts(row['input'])
        if truth != row['facts'] or truth != expected_facts(row['family']):
            raise ValueError('Contrast label mismatch')
        rel = observable_relations(session.context.snapshot(len(session.recorder.trajectory())),
                                   session.recorder.trajectory())
        alternate = dict(C=rel['calculation_observed'], M=rel['memory_matches_calculation'],
                         F=rel['get_after_last_target_put'] is True and rel['get_matches_calculation'] is True)
        if alternate != truth:
            raise ValueError('Independent relation scan mismatch')
        teacher_type = action_type(session.next_action())
        wanted = 'memory.put' if not truth['M'] else 'stop' if truth['F'] else 'memory.get'
        if teacher_type != wanted:
            raise ValueError('Contrast teacher mismatch')
        independent_completion(session)
        raw = row['input'].encode('utf-8')
        if len(raw)+2 > CONTEXT or bytes(raw).decode('utf-8') != row['input']:
            raise ValueError('Contrast context/roundtrip failed')
        sig = digest(row['input'])
        if row['input_sha256'] != sig or (sig in labels and labels[sig] != truth):
            raise ValueError('Contrast digest/label collision')
        labels[sig] = truth
        lengths.append(len(raw)+2)
        manifest.append(dict(case_id=row['case_id'], input_sha256=sig, byte_tokens=len(raw)+2,
                             old_train_overlap=sig in old_train, old_dev_overlap=sig in old_dev,
                             teacher_type=teacher_type, facts=truth))
    for pair in pairs:
        a, b = (by_id[pair[k]] for k in ('left', 'right'))
        da, db = (json.loads(r['input']) for r in (a, b))
        if len(a['input'].encode()) != len(b['input'].encode()) or len(da['events']) != len(db['events']):
            raise ValueError('Pair length/event-count mismatch')
        axis = pair['axis']
        if axis == 'rename':
            key, alias = da['task']['store_as'], db['task']['store_as']
            if key == alias or alias != rename_key(key) or len(key.encode()) != len(alias.encode()):
                raise ValueError('Rename identity mismatch')
            def mapped(value):
                if isinstance(value, dict):
                    return {alias if k == key else k: mapped(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [mapped(v) for v in value]
                return alias if value == key else value
            if mapped(da) != db or a['facts'] != b['facts']:
                raise ValueError('Rename consistency mismatch')
        elif axis == 'value':
            if da['task'] != db['task'] or da['events'][:-1] != db['events'][:-1]:
                raise ValueError('Value context changed')
            fixed = deepcopy(db)
            key = da['task']['store_as']
            fixed['memory'][key] = da['memory'][key]
            fixed['events'][-1]['action']['arguments']['value'] = da['events'][-1]['action']['arguments']['value']
            if fixed != da or a['facts']['M'] is not True or b['facts']['M'] is not False:
                raise ValueError('Value intervention mismatch')
        elif axis == 'order':
            if (da['task'] != db['task'] or da['memory'] != db['memory']
                    or Counter(canonical(e) for e in da['events']) != Counter(canonical(e) for e in db['events'])
                    or da['events'] == db['events'] or a['facts']['F'] is not False or b['facts']['F'] is not True):
                raise ValueError('Order intervention mismatch')
        else:
            raise ValueError('Unknown contrast axis')
    return dict(schema_version=SCHEMA, suite_sha256=digest(rows), pairs_sha256=digest(pairs),
                manifest_sha256=digest(manifest), input_id=INPUT_ID, goal_contract=CURRENT_GOAL_V1,
                unique_inputs=40, pair_count=24, reference_slots=48, task_groups=8,
                counts=dict(Counter(r['split'] for r in rows)),
                max_byte_tokens=max(lengths), context_length=CONTEXT, manifest=manifest,
                old_train_overlap=sum(m['old_train_overlap'] for m in manifest),
                old_dev_overlap=sum(m['old_dev_overlap'] for m in manifest),
                collisions=0, all_pair_lengths_equal=True, all_prefixes_replayed=True,
                oracle_completions=40, structural_holdout=False, model_scored=0,
                model_updates=0, test_constructed=0, test_scored=0)
