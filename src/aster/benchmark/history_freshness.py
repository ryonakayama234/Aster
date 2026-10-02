"""Small executed freshness task, preserving unknown and equal-multiset order pairs."""
from collections import Counter
import json

from aster.benchmark.history_order import order_session, TRAIN_SHAPES, EVAL_SHAPES
from aster.benchmark.history_contrasts import as_row
from aster.benchmark.history_reader import TASKS, build_cases, read_facts, replay, CONTEXT
from aster.benchmark.state_representation import canonical, digest
from aster.records.decision_relations import observable_relations

SCHEMA = 'aster-history-freshness-0'


def build_freshness():
    rows, pairs = [], []
    for split, tasks in TASKS.items():
        for i, (op, left, right, key) in enumerate(tasks):
            task = dict(kind='calculate_and_store', operation=op, left=left, right=right, store_as=key)
            group = f'{split}/task{i}'
            for shape in ((*TRAIN_SHAPES, *EVAL_SHAPES) if split == 'train' else EVAL_SHAPES):
                cohort = 'train' if shape in TRAIN_SHAPES else 'shape_'+split
                ids = []
                for fresh in (False, True):
                    cid = f'{group}/{shape}/{int(fresh)}'
                    row = as_row(order_session(task, shape, fresh), cid, split, group, shape)
                    rows.append({**row, 'cohort': cohort})
                    ids.append(cid)
                pairs.append(dict(pair_id=f'{group}/{shape}', left=ids[0], right=ids[1], cohort=cohort, shape=shape))
    for row in build_cases():
        if row['family'] in ('normal1', 'normal2'):
            rows.append({**row, 'input_sha256': digest(row['input']),
                         'cohort': 'train' if row['split']=='train' else 'unknown_dev'})
    return rows, pairs


def audit_freshness(rows, pairs):
    if (rows, pairs) != build_freshness():
        raise ValueError('Freshness generation/trajectory/label mismatch')
    if len(rows)!=128 or len(pairs)!=52 or len({r['case_id'] for r in rows})!=128:
        raise ValueError('Freshness counts mismatch')
    signatures, groups, manifest = {}, {}, []
    train_hashes = {digest(r['input']) for r in rows if r['cohort']=='train'}
    for row in rows:
        s = replay(row)
        h = s.recorder.trajectory()
        ref = observable_relations(s.context.snapshot(len(h)), h)
        expected = (ref['get_after_last_target_put'] is True and ref['get_matches_calculation'] is True) if ref['get_observed'] else None
        if read_facts(row['input']) != row['facts'] or row['facts']['F'] is not expected:
            raise ValueError('Independent F label mismatch')
        if row['facts']['C'] is not True:
            raise ValueError('Calculation not observed')
        if row['family'] not in ('normal1','normal2') and row['facts']['M'] is not True:
            raise ValueError('Order task memory mismatch')
        sig = digest(row['input'])
        if sig in signatures or row['input_sha256']!=sig:
            raise ValueError('Duplicate or changed visible input')
        signatures[sig] = row['facts']['F']
        groups.setdefault(row['leakage_group'], set()).add(row['split'])
        size = len(row['input'].encode())+2
        if size>CONTEXT or (row['cohort']!='train' and sig in train_hashes):
            raise ValueError('Context or train overlap')
        manifest.append(dict(case_id=row['case_id'], input_sha256=sig, byte_tokens=size, cohort=row['cohort']))
    if len(groups)!=12 or any(len(v)!=1 for v in groups.values()):
        raise ValueError('Task split leakage')
    lookup = {r['case_id']:r for r in rows}
    for p in pairs:
        a,b = (lookup[p[k]] for k in ('left','right'))
        da,db = (json.loads(r['input']) for r in (a,b))
        if (da['task']!=db['task'] or da['memory']!=db['memory']
                or Counter(canonical(e) for e in da['events'])!=Counter(canonical(e) for e in db['events'])
                or len(a['input'].encode())!=len(b['input'].encode())
                or a['facts']['F'] is not False or b['facts']['F'] is not True):
            raise ValueError('Matched order pair mismatch')
        if p['shape'] in ('tail_put','tail_get') and da['events'][-1]!=db['events'][-1]:
            raise ValueError('Shared suffix mismatch')
    train = [r for r in rows if r['cohort']=='train']
    return dict(schema_version=SCHEMA, suite_sha256=digest(rows), pairs_sha256=digest(pairs),
                manifest_sha256=digest(manifest), manifest=manifest,
                counts=dict(Counter(r['cohort'] for r in rows)), unique_inputs=128, task_groups=12,
                train_labels=dict(Counter(str(r['facts']['F']) for r in train)),
                max_byte_tokens=max(m['byte_tokens'] for m in manifest),
                train_eval_overlap=0, all_prefixes_replayed=True, test_constructed=0, test_scored=0,
                independent_holdout=False)
