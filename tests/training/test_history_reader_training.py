import json
from pathlib import Path

import pytest
import torch
from torch import nn

from aster.benchmark.history_reader import build_cases, make_prefix
from aster.training.history_reader import (CLASSES, CLASSIFIER_HASH, CONFIG, HistoryReader,
    encode, execute_proposal, load_reader, predict, select_action, target_ids, write_json)
from aster.training.decision_fit import _state_digest


def test_byte_limits_and_fact_classes():
    assert encode('猫').tolist() == [[256, *'猫'.encode('utf-8'), 257]]
    with pytest.raises(ValueError, match='context_exceeded'):
        encode('a'*2047)
    assert target_ids({'C':False,'M':None,'F':None}).tolist() == [0,2,2]


def test_contradiction_abstains_before_binding():
    task = dict(kind='calculate_and_store',operation='add',left=2,right=3,store_as='x')
    session = make_prefix(task,'normal0')
    action, error = select_action(nn.Linear(3,4),{'C':False,'M':True,'F':None},session)
    assert action is None and error == 'inconsistent_facts_abstain'


def test_reader_reload_and_contract_rejection(tmp_path):
    import hashlib
    model = HistoryReader().eval()
    weights = tmp_path/'weights.pt'
    torch.save(model.state_dict(),weights)
    meta = dict(input_id='aster-history-reader-input-0',classes=list(CLASSES),facts=['C','M','F'],config=CONFIG,
                goal_contract='calculate-and-store-current-v1',classifier_model_sha256=CLASSIFIER_HASH,
                model_sha256=_state_digest(model.state_dict()),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    write_json(tmp_path/'manifest.json',meta)
    text = build_cases()[0]['input']
    rng = torch.get_rng_state().clone()
    loaded = load_reader(tmp_path,CLASSIFIER_HASH)
    assert torch.equal(rng,torch.get_rng_state())
    assert predict(model,text) == predict(loaded,text)
    meta['input_id'] = 'wrong'
    write_json(tmp_path/'manifest.json',meta)
    with pytest.raises(ValueError,match='contract mismatch'):
        load_reader(tmp_path,CLASSIFIER_HASH)


def test_failed_proposal_is_recorded_as_real_observation():
    from aster.records.transition import Action
    task = dict(kind='calculate_and_store',operation='add',left=2,right=3,store_as='x')
    session = make_prefix(task,'normal0')
    transition = execute_proposal(session,Action.tool('memory.get',key='missing'))
    assert not transition.observation.ok
    assert len(session.recorder.trajectory()) == 1
    assert transition.observation.error is not None
