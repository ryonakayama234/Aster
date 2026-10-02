from copy import deepcopy
import json

import pytest

from aster.benchmark.history_contrasts import audit_contrasts, build_contrasts
from aster.training.history_contrasts import compare_previous_score, summarize_pairs, verify_registration


def test_executed_contrasts_count_shared_inputs_and_old_overlap():
    rows, pairs = build_contrasts()
    report = audit_contrasts(rows, pairs)
    assert report['unique_inputs'] == 40
    assert report['pair_count'] == 24
    assert report['reference_slots'] == 48
    assert report['task_groups'] == 8
    assert report['old_train_overlap'] == 8
    assert report['old_dev_overlap'] == 8
    assert report['model_updates'] == report['test_scored'] == 0


def test_order_changes_f_only_with_identical_event_multiset_and_memory():
    rows, _ = build_contrasts()
    by_id = {r['case_id']: r for r in rows}
    a, b = by_id['train/task0/same'], by_id['train/task0/fresh']
    x, y = json.loads(a['input']), json.loads(b['input'])
    assert x['task'] == y['task'] and x['memory'] == y['memory']
    assert len(a['input'].encode()) == len(b['input'].encode())
    assert sorted(json.dumps(e, sort_keys=True) for e in x['events']) == sorted(json.dumps(e, sort_keys=True) for e in y['events'])
    assert a['facts'] == {'C': True, 'M': True, 'F': False}
    assert b['facts'] == {'C': True, 'M': True, 'F': True}


def test_tampered_label_or_observation_cannot_pass_registration():
    rows, pairs = build_contrasts()
    bad = deepcopy(rows)
    bad[0]['facts']['F'] = False
    with pytest.raises(ValueError, match='registration generation mismatch'):
        audit_contrasts(bad, pairs)
    bad = deepcopy(rows)
    bad[0]['trajectory'][0]['observation']['output'] = 999
    with pytest.raises(ValueError, match='registration generation mismatch'):
        audit_contrasts(bad, pairs)


def test_prediction_invariance_without_correctness_is_not_success():
    rows, pairs = build_contrasts()
    registration = audit_contrasts(rows, pairs)
    predictions = [dict(case_id=r['case_id'], predicted={'C': False, 'M': None, 'F': None},
                        truth=r['facts'], correct=False, action_correct=False) for r in rows]
    scored, summaries = summarize_pairs(pairs, predictions, registration['manifest'])
    assert all(p['prediction_same'] for p in scored)
    assert not any(p['both_facts_correct'] for p in scored)
    assert summaries['rename']['dev']['prediction_same'] == 4
    assert summaries['rename']['dev']['both_facts_correct'] == 0


def test_altered_registration_is_rejected_before_model_scoring():
    rows, pairs = build_contrasts()
    report = audit_contrasts(rows, pairs)
    altered = deepcopy(report)
    altered['pairs_sha256'] = '0'*64
    with pytest.raises(ValueError, match='registration mismatch'):
        verify_registration(report, altered)


def test_cross_environment_tolerance_requires_identical_classes():
    before = dict(logits=[[1.0, 0.0, -1.0]]*3, predicted={'C': False, 'M': False, 'F': False})
    close = dict(logits=[[1.000001, 0.0, -1.0]]*3, predicted=before['predicted'])
    assert compare_previous_score(close, before) < 1e-5
    close['predicted'] = {'C': True, 'M': False, 'F': False}
    with pytest.raises(ValueError, match='logits/class mismatch'):
        compare_previous_score(close, before)
    distant = dict(logits=[[1.1, 0.0, -1.0]]*3, predicted=before['predicted'])
    with pytest.raises(ValueError, match='logits/class mismatch'):
        compare_previous_score(distant, before)
