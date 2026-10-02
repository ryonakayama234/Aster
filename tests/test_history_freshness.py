from copy import deepcopy
import json
import pytest

from aster.benchmark.history_freshness import audit_freshness,build_freshness


def test_executed_freshness_registration_and_unknown():
    rows,pairs = build_freshness()
    result = audit_freshness(rows,pairs)
    assert result['counts']==dict(train=48,shape_train=48,shape_dev=24,unknown_dev=8)
    assert result['train_labels']=={'False':16,'True':16,'None':16}
    assert result['train_eval_overlap']==0
    assert all(r['facts']['F'] is None for r in rows if r['family'] in ('normal1','normal2'))


def test_no_shared_tail_or_event_counts_solve_freshness_pairs():
    rows,pairs = build_freshness()
    lookup = {r['case_id']:r for r in rows}
    for p in pairs:
        a,b = [json.loads(lookup[p[k]]['input']) for k in ('left','right')]
        assert len(a['events'])==len(b['events'])
        if p['shape'] in ('tail_put','tail_get'):
            assert a['events'][-1]==b['events'][-1]


def test_tampered_freshness_truth_and_runtime_evidence_rejected():
    rows,pairs = build_freshness()
    bad = deepcopy(rows)
    bad[0]['facts']['F']=True
    with pytest.raises(ValueError):
        audit_freshness(bad,pairs)
    bad = deepcopy(rows)
    bad[0]['trajectory'][-1]['observation']['ok']=False
    with pytest.raises(ValueError):
        audit_freshness(bad,pairs)


def test_simple_train_derived_controls_fail_order_transfer():
    pytest.importorskip('torch')
    from aster.training.history_freshness import baselines
    rows,pairs = build_freshness()
    controls = baselines(rows,pairs)
    assert controls['majority']['pairs']['train']['both_correct']==0
    assert controls['last_tool']['pairs']['shape_dev']['both_correct']<12
