from copy import deepcopy
import pytest

from aster.benchmark.history_order import audit_order,build_order,training_slots


def test_executed_order_suite_and_matched_control():
    rows,pairs = build_order()
    result = audit_order(rows,pairs)
    assert result['counts'] == dict(old_train=80,old_dev=40,added_train=32,shape_train=48,shape_dev=24,contrast_train=20,contrast_dev=20)
    assert result['shape_overlap'] == 0
    b,c = (training_slots(rows,a) for a in ('repeat','order'))
    assert all(x['facts']==y['facts'] and x['task']==y['task'] for x,y in zip(b,c,strict=True))
    assert b[:80] == c[:80]


def test_tampered_order_label_and_trajectory_rejected():
    rows,pairs = build_order()
    changed = deepcopy(rows)
    changed[120]['facts']['F'] = not changed[120]['facts']['F']
    with pytest.raises(ValueError):
        audit_order(changed,pairs)
    changed = deepcopy(rows)
    changed[120]['trajectory'][-1]['observation']['ok'] = False
    with pytest.raises(ValueError):
        audit_order(changed,pairs)


def test_shared_suffix_cannot_determine_order_label():
    import json
    rows,pairs = build_order()
    by_id = {r['case_id']:r for r in rows}
    for p in pairs:
        if p['pair_id'].startswith('order/') and p['shape'] in ('tail_put','tail_get'):
            a,b = (by_id[p[k]] for k in ('left','right'))
            assert json.loads(a['input'])['events'][-1] == json.loads(b['input'])['events'][-1]
            assert a['facts']['F'] is False and b['facts']['F'] is True
