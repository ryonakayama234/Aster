"""Executed order intervention; labels never enter model-visible history."""
from collections import Counter
import json

from aster.benchmark.history_contrasts import as_row, build_contrasts
from aster.benchmark.history_reader import (CONTEXT, TASKS, ExecutedHistory,
    build_cases, independent_completion, read_facts, replay)
from aster.benchmark.state_representation import canonical, digest
from aster.records.decision_relations import observable_relations
from aster.records.transition import Action

SCHEMA = 'aster-history-order-intervention-0'
TRAIN_SHAPES = ('plain', 'tail_put')
EVAL_SHAPES = ('middle_put', 'tail_get', 'repeat_get')
EPOCHS = 40


def order_session(task, shape, fresh):
    session = ExecutedHistory(task)
    session.execute(session.next_action())  # observed calculator, no invented result
    key = task['store_as']
    value = session.recorder.trajectory().transitions[-1].observation.output
    noise = 'reader_noise'
    if key == noise:
        raise ValueError('Noise key collides with target')
    if shape == 'tail_get':
        session.execute(Action.tool('memory.put', key=noise, value=101))
    session.execute(Action.tool('memory.put', key=key, value=value))
    if shape == 'middle_put':
        session.execute(Action.tool('memory.put', key=noise, value=101))
    put = Action.tool('memory.put', key=key, value=value)
    get = Action.tool('memory.get', key=key)
    gets = [get, get] if shape == 'repeat_get' else [get]
    for action in ([put, *gets] if fresh else [*gets, put]):
        session.execute(action)
    if shape == 'tail_put':
        session.execute(Action.tool('memory.put', key=noise, value=101))
    elif shape == 'tail_get':
        session.execute(Action.tool('memory.get', key=noise))
    elif shape not in (*TRAIN_SHAPES, *EVAL_SHAPES):
        raise ValueError('Unknown order shape')
    return session


def build_order():
    old = build_cases()
    rows = [{**r, 'case_id': 'old/'+r['case_id'], 'input_sha256': digest(r['input']),
             'cohort': 'old_'+r['split']} for r in old]
    pairs = []
    for split in ('train', 'dev'):
        for index, (op, left, right, key) in enumerate(TASKS[split]):
            task = dict(kind='calculate_and_store', operation=op, left=left, right=right, store_as=key)
            group = f'{split}/task{index}'
            shapes = (*TRAIN_SHAPES, *EVAL_SHAPES) if split == 'train' else EVAL_SHAPES
            for shape in shapes:
                cohort = 'added_train' if shape in TRAIN_SHAPES else 'shape_'+split
                ids = []
                for fresh in (False, True):
                    cid = f'order/{group}/{shape}/{int(fresh)}'
                    row = as_row(order_session(task, shape, fresh), cid, split, group, shape)
                    rows.append({**row, 'cohort': cohort})
                    ids.append(cid)
                pairs.append(dict(pair_id=f'order/{group}/{shape}', left=ids[0], right=ids[1],
                                  shape=shape, split=split, cohort=cohort))
    contrast, prior_pairs = build_contrasts()
    rows += [{**r, 'case_id': 'contrast/'+r['case_id'], 'cohort': 'contrast_'+r['split']} for r in contrast]
    pairs += [{**p, 'pair_id': 'contrast/'+p['pair_id'], 'left': 'contrast/'+p['left'],
               'right': 'contrast/'+p['right'], 'shape': p['axis'], 'cohort': 'contrast_'+p['split']}
              for p in prior_pairs]
    return rows, pairs


def training_slots(rows, arm):
    base = [r for r in rows if r['cohort'] == 'old_train']
    added = [r for r in rows if r['cohort'] == 'added_train']
    if arm == 'order':
        return base+added
    if arm != 'repeat':
        raise ValueError('Unknown training arm')
    counts = Counter()
    extra = []
    for row in added:
        matching = [r for r in base if r['task'] == row['task'] and r['facts'] == row['facts']]
        if not matching:
            raise ValueError('No label/task-matched repeat control')
        signature = canonical((row['task'], row['facts']))
        extra.append(matching[counts[signature] % len(matching)])
        counts[signature] += 1
    return base+extra


def audit_order(rows, pairs):
    if (rows, pairs) != build_order():
        raise ValueError('Order suite generation changed')
    if len(rows) != 264 or len(pairs) != 76 or len({r['case_id'] for r in rows}) != 264:
        raise ValueError('Order suite count mismatch')
    b, c = (training_slots(rows, a) for a in ('repeat', 'order'))
    if len(b) != 112 or len(c) != 112 or b[:80] != c[:80]:
        raise ValueError('Base exposure/slot mismatch')
    if any(x['facts'] != y['facts'] or x['task'] != y['task'] for x,y in zip(b,c,strict=True)):
        raise ValueError('Matched slot labels/task mismatch')
    train_union = {digest(r['input']) for r in b+c}
    manifest = []
    for r in rows:
        s = replay(r)
        truth = read_facts(r['input'])
        relations = observable_relations(s.context.snapshot(len(s.recorder.trajectory())), s.recorder.trajectory())
        independent = dict(C=relations['calculation_observed'], M=relations['memory_matches_calculation'],
                           F=relations['get_after_last_target_put'] is True and relations['get_matches_calculation'] is True)
        # Non-order rows can have unknown; prior preflight already distinguishes these.
        if truth != r['facts'] or (r['case_id'].startswith('order/') and independent != truth):
            raise ValueError('Order label/independent scan mismatch')
        tokens = len(r['input'].encode('utf-8'))+2
        if tokens > CONTEXT or r['input_sha256'] != digest(r['input']):
            raise ValueError('Order context/hash mismatch')
        independent_completion(s)
        manifest.append(dict(case_id=r['case_id'], input_sha256=digest(r['input']), byte_tokens=tokens,
                             cohort=r['cohort'], train_union_overlap=digest(r['input']) in train_union))
    lookup = {r['case_id']: r for r in rows}
    for p in pairs:
        if not p['pair_id'].startswith('order/'):
            continue
        a,b = (lookup[p[k]] for k in ('left','right'))
        da,db = (json.loads(r['input']) for r in (a,b))
        if (da['task'] != db['task'] or da['memory'] != db['memory']
                or Counter(canonical(e) for e in da['events']) != Counter(canonical(e) for e in db['events'])
                or len(a['input'].encode()) != len(b['input'].encode())
                or a['facts'] != dict(C=True,M=True,F=False) or b['facts'] != dict(C=True,M=True,F=True)):
            raise ValueError('Order pair intervention mismatch')
        if p['shape'] in ('tail_put','tail_get') and da['events'][-1] != db['events'][-1]:
            raise ValueError('Shared suffix mismatch')
    if any(m['train_union_overlap'] for m in manifest if m['cohort'].startswith('shape_')):
        raise ValueError('Unseen shape leaks into training union')
    return dict(schema_version=SCHEMA, suite_sha256=digest(rows), pairs_sha256=digest(pairs),
                manifest=manifest, manifest_sha256=digest(manifest), rows=len(rows), pairs=len(pairs),
                unique_inputs=len({r['input_sha256'] for r in rows}), counts=dict(Counter(r['cohort'] for r in rows)),
                max_byte_tokens=max(m['byte_tokens'] for m in manifest), epochs=EPOCHS,
                slots_sha256={a:digest([r['case_id'] for r in training_slots(rows,a)]) for a in ('repeat','order')},
                slots_per_epoch=112, updates_per_arm=EPOCHS*112, oracle_completions=len(rows),
                shape_overlap=0, test_constructed=0, test_scored=0, independent_holdout=False)
