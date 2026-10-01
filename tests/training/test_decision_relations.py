from dataclasses import replace
import json

import pytest
import torch

from aster.benchmark.fixed_v1_diagnostics import Session
from aster.benchmark.mixed_v1_preflight import build_training
from aster.records.decision_relations import observable_relations, relation_input, RELATION_SERIALIZER_ID
from aster.records.transition import Action
from aster.training.decision_fit import DecisionFitStudyConfig, _new_model
from aster.training.decision_mixed_v1 import load_v1_checkpoint, train_arm
from aster.training.decision_relations import prepare
from aster.training.decision_tokenizer import build_byte_decision_tokenizer
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite


def session():
    return Session(dict(kind="calculate_and_store",operation="add",left=2,right=3,store_as="total"))


def facts(s):
    ex=s.example()
    return observable_relations(ex.state,ex.trajectory)


def test_unknown_observation_and_temporal_cancellation():
    s=session()
    assert facts(s)['memory_matches_calculation'] is None
    assert facts(s)['get_after_last_target_put'] is None
    s.teacher_step(); s.teacher_step(); s.teacher_step()
    assert facts(s)['get_after_last_target_put'] is True
    assert facts(s)['get_matches_calculation'] is True
    s.execute(Action.tool('memory.put',key='unrelated',value=5))
    assert facts(s)['get_after_last_target_put'] is True
    s.execute(Action.tool('memory.put',key='total',value=5))
    assert facts(s)['memory_matches_calculation'] is True
    assert facts(s)['get_after_last_target_put'] is False
    s.execute(Action.tool('memory.get',key='total'))
    assert facts(s)['get_after_last_target_put'] is True


def test_repeated_wrong_write_is_invariant_and_metadata_independent():
    s=session(); s.teacher_step()
    s.execute(Action.tool('memory.put',key='total',value=-999))
    ex=s.example(); candidate=ex.candidates[0]
    before=relation_input(ex.state,ex.trajectory,candidate)
    changed=replace(ex.trajectory,transitions=tuple(replace(t,evaluation=replace(t.evaluation,goal_satisfied=True)) for t in ex.trajectory.transitions))
    assert before==relation_input(ex.state,changed,candidate)
    s.execute(Action.tool('memory.put',key='total',value=-999))
    after=s.example()
    assert before==relation_input(after.state,after.trajectory,candidate)
    assert facts(s)['memory_matches_calculation'] is False


def test_calculation_must_match_current_task_and_bool_is_not_numeric_value():
    s=session()
    s.execute(Action.tool('calculator',operation='add',left=99,right=1))
    assert facts(s)['calculation_observed'] is False
    s.teacher_step()
    ex=s.example()
    state=replace(ex.state,memory={'total':True})
    assert observable_relations(state,ex.trajectory)['memory_matches_calculation'] is False


def test_registered_preflight_collision_and_common_exclusions():
    from pathlib import Path
    original=json.loads((Path(__file__).resolve().parents[2]/'reports/decision-mixed-v1-preflight-v0.json').read_text())
    _,dev,pairs,slots,report=prepare(original)
    assert len(slots)==64 and len(dev)==128
    assert all(r['collisions']==0 and r['max_tokens']<=2048 for r in report['representations'].values())
    assert report['common_nonoverlap_pairs']==sum(p['nonoverlap'] for p in pairs)
    assert all(c['overlap_union_train']==any(c['overlap_by_serializer'].values()) for c in dev)


def test_relation_checkpoint_rejects_compact_reload(tmp_path):
    torch.set_num_threads(2)
    raw,_=build_training()
    cases=tuple(BenchmarkCase(c['case_id'],'train',c['group'],c['leakage_group'],c['example']) for c in raw[:2])
    suite=BenchmarkSuite('debug-relations',cases)
    tk=build_byte_decision_tokenizer(); cfg=DecisionFitStudyConfig(epochs=1)
    initial={k:v.detach().clone() for k,v in _new_model(tk,cfg).state_dict().items()}
    path=tmp_path/'relations-v1'
    train_arm(path,suite,[c.case_id for c in cases],tk,initial,cfg,source={'git_sha':'debug'},serializer=relation_input,serializer_id=RELATION_SERIALIZER_ID)
    load_v1_checkpoint(path/'checkpoint',serializer_id=RELATION_SERIALIZER_ID)
    with pytest.raises(ValueError,match='contract mismatch'):
        load_v1_checkpoint(path/'checkpoint')
