import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from aster.benchmark.fixed_v1_diagnostics import Session,build_cases
from aster.records.transition import Action
from aster.training.decision_action_types import bind_type,features,prepare,score,load_checkpoint


def test_preflight_reports_feature_overlap_instead_of_holdout():
    report=prepare()[-1]
    assert report['train_unique_feature_patterns']==4
    assert report['dev_feature_train_overlap']==128
    assert report['new_initial_feature_train_overlap']==4
    assert report['collisions']==0 and report['test_scored']==0


def test_binding_refuses_missing_or_ambiguous_type_and_is_order_invariant():
    candidates=(Action.tool('calculator',operation='add',left=2,right=3),Action.stop('policy_stop'),Action.stop('goal_verified'))
    assert bind_type('stop',candidates)==Action.stop('goal_verified')
    assert bind_type('stop',tuple(reversed(candidates)))==Action.stop('goal_verified')
    with pytest.raises(ValueError,match='no_unique_candidate'):
        bind_type('memory.put',candidates)
    with pytest.raises(ValueError,match='no_unique_candidate'):
        bind_type('calculator',(candidates[0],Action.tool('calculator',operation='add',left=4,right=5)))


def test_feature_scope_and_target_independence():
    session=Session(dict(kind='calculate_and_store',operation='add',left=2,right=3,store_as='x'))
    ex=session.example();assert features(ex)==[0,0,0]
    assert features(replace(ex,teacher='different',target_index=1))==features(ex)
    session.execute(Action.tool('memory.get',key='missing'))
    with pytest.raises(ValueError,match='outside_scope'):
        features(session.example())


def test_linear_representability_is_not_a_learned_result():
    model=nn.Linear(3,4)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-3.,0.,0.],[2.,-2.,0.],[0.,1.,-1.],[0.,0.,1.]]))
        model.bias.copy_(torch.tensor([3.,0.,0.,0.]))
    for c in build_cases()[0]:
        assert score(model,c['example'])['correct_action']


def test_checkpoint_contract_mismatch_is_rejected_before_weights(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(feature_id='wrong',binder_id='wrong',goal_contract='wrong',types=[])))
    with pytest.raises(ValueError,match='contract mismatch'):
        load_checkpoint(tmp_path)
