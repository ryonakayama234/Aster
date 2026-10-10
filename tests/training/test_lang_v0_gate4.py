"""Synthetic only: no private corpus/checkpoints, no weight updates."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def runner():
    spec = importlib.util.spec_from_file_location('lang_gate4_runner', ROOT / 'scripts/run_lang_v0_gate4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_policy_and_repetition_overlap_helpers():
    r = runner()
    assert len(r.PROMPTS) == 6 and len(set(r.PROMPTS)) == 6
    assert r.STEPS == (0, 200)
    assert r.MAX_NEW_TOKENS == 64
    assert r.has_repeated_trigram([1, 2, 3, 1, 2, 3])
    assert not r.has_repeated_trigram([1, 2, 3, 4, 5])
    chunk = ('朝の天気は良いでしょう。'*5).encode()
    snippet = chunk[3:55]
    assert r.has_long_training_match(snippet, (chunk,))
    assert not r.has_long_training_match(b'a'*32, (chunk,))
    assert not r.has_long_training_match(b'a'*31, (b'a'*100,))
    assert r.token_bytes(SimpleNamespace(model=SimpleNamespace(vocab={1: b'A', 2: b'B'})), [1, 99, 2]) == b'AB'


def test_report_is_sha_locked_and_sealed_test_unused(tmp_path, monkeypatch):
    r = runner()
    data = {'run_id': r.RUN_ID, 'sealed_test_scored': False,
            'reload_exact_match': True, 'mode': 'read_only_report',
            'checkpoint_evaluations': [
                {'step': step, 'checkpoint_sha256': 'f'*64}
                for step in range(0, 201, 50)
            ]}
    path = tmp_path / 'lang-gate3-report.json'
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(r, 'REPORT_SHA256', hashlib.sha256(raw).hexdigest())
    assert r.read_frozen_report(tmp_path)['run_id'] == r.RUN_ID
    changed = json.dumps(dict(data, sealed_test_scored=True)).encode()
    path.write_bytes(changed)
    with pytest.raises(ValueError, match='digest'):
        r.read_frozen_report(tmp_path)
    monkeypatch.setattr(r, 'REPORT_SHA256', hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match='status'):
        r.read_frozen_report(tmp_path)


def test_checkpoint_sha_mismatch_is_rejected_before_load(tmp_path, monkeypatch):
    r = runner()
    (tmp_path / 'checkpoint-000000.pt').write_bytes(b'fake payload')
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Must not load unverified checkpoint')
    monkeypatch.setattr(r, 'load_checkpoint', forbidden)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        r.verify_checkpoint(tmp_path, {'step': 0, 'tokens_seen': 0, 'checkpoint_sha256': 'f'*64},
                            {'step': 0, 'tokens_seen': 0, 'checkpoint_sha256': 'f'*64,
                             'checkpoint': 'checkpoint-000000.pt'}, {}, {}, 'tokenizer-id')



def test_verified_checkpoint_snapshot_is_the_bytes_loaded_even_if_path_changes(tmp_path, monkeypatch):
    r = runner()
    path = tmp_path / 'checkpoint-000200.pt'
    verified = b'frozen checkpoint bytes'
    path.write_bytes(verified)
    expected = hashlib.sha256(verified).hexdigest()
    checkpoint = {'step': 200, 'tokens_seen': 202172, 'checkpoint_sha256': expected}
    observation = {**checkpoint, 'checkpoint': path.name}
    loaded_bytes = []

    def fake_load(file_like, expected_tokenizer_id):
        # A path re-open here would read the swapped payload, not verified data.
        path.write_bytes(b'swapped after verification')
        loaded_bytes.append(file_like.read())
        assert expected_tokenizer_id == 'payload-id'
        return SimpleNamespace(training=False), SimpleNamespace(), {}

    monkeypatch.setattr(r, 'load_checkpoint', fake_load)
    monkeypatch.setattr(r, 'verify_checkpoint_observation', lambda *_args: None)
    model, _tok = r.verify_checkpoint(
        tmp_path, checkpoint, observation, {}, {}, 'payload-id')
    assert not model.training
    assert loaded_bytes == [verified]
    assert path.read_bytes() != verified


def test_private_output_is_0600_and_never_overwritten(tmp_path):
    r = runner()
    path = tmp_path / r.PRIVATE_OUTPUT
    r.write_private_once(path, {'continuation': '非公開の文章'})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    previous = path.read_bytes()
    with pytest.raises(FileExistsError):
        r.write_private_once(path, {'continuation': '上書き'})
    assert path.read_bytes() == previous


def test_full_audit_runs_exactly_two_pairs_with_silent_stdout(tmp_path, monkeypatch, capsys):
    r = runner()
    run_path = tmp_path / 'runs' / r.RUN_ID
    run_path.mkdir(parents=True)
    experiment = {'config': {'threads': 2}, 'tokenizer_id': 'payload',
                  'tokenizer_artifact_id': 'artifact', 'code_sha256': {}}
    (run_path / 'experiment.json').write_text(json.dumps(experiment))
    (run_path / 'run.json').write_text(json.dumps({
        'run_id': r.RUN_ID, 'status': 'completed',
        'inputs': {'tokenizer_artifact_id': 'artifact'}}))
    (run_path / 'training-bundle.json').write_text(json.dumps({
        'run_id': r.RUN_ID, 'status': 'completed', 'experiment': experiment}))
    checkpoints = [{'step': i, 'tokens_seen': i,
                    'checkpoint_sha256': 'e'*64} for i in range(0, 201, 50)]
    preflight = {'view_id': 'view', 'tokenizer_payload_sha256': 'payload',
                 'sealed_test_text_read': False}
    report = {'training_view_id': 'view', 'tokenizer_artifact_id': 'artifact',
              'pilot_config': {'threads': 2}, 'preflight': preflight,
              'checkpoint_evaluations': checkpoints}
    tok = SimpleNamespace(model=SimpleNamespace(vocab={1: b'x', 2: b'y'}), eos_id=999)
    tok.encode = lambda _text, add_bos=True: [10, 1]
    monkeypatch.setattr(r, 'read_frozen_report', lambda _run: report)
    monkeypatch.setattr(r, 'preflight', lambda *_a: (
        {'pilot_config': {'threads': 2}, 'tokenizer_artifact_id': 'artifact'},
        tok, {'train': [SimpleNamespace(ids=[1, 2, 1, 2])]}, preflight))
    monkeypatch.setattr(r, 'verify_scoring_dependencies', lambda *_args: {})
    monkeypatch.setattr(r, 'verify_run_sources', lambda *_args: None)
    monkeypatch.setattr(r, 'validate_run_observations', lambda *_args: checkpoints)
    visited = []
    def fake_verify(_run, entry, _obs, *_args):
        visited.append(entry['step'])
        return SimpleNamespace(training=False), tok
    monkeypatch.setattr(r, 'verify_checkpoint', fake_verify)
    monkeypatch.setattr(r, 'generate_ids', lambda *_args: [1, 2])
    monkeypatch.setattr(r, 'describe_ids', lambda *_args: {'text': '秘密の続き', 'utf8_valid': True})
    result = r.audit(tmp_path)
    assert visited == [0, 200]
    assert result['pair_count'] == 6
    assert result['training_updates'] == 0
    assert result['sealed_test_text_read'] is False
    assert len(result['metrics_by_step']) == 2
    assert '秘密の続き' not in json.dumps(result, ensure_ascii=False)
    assert capsys.readouterr().out == ''
    stored = json.loads((run_path / r.PRIVATE_OUTPUT).read_text())
    assert [c['step'] for c in stored['checkpoints']] == [0, 200]
    assert [v['prompt'] for v in stored['checkpoints'][0]['results']] == list(r.PROMPTS)
    assert [v['prompt'] for v in stored['checkpoints'][1]['results']] == list(r.PROMPTS)
    assert stored['checkpoints'][1]['results'][0]['continuation'] == '秘密の続き'
    with pytest.raises(FileExistsError):
        r.audit(tmp_path)
