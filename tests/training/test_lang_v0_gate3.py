"""Synthetic-only tests: do not use private corpus or sealed test outcomes."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aster.training.lang_gate3 import fit_unigram, git_blob_sha, preflight, score_model, score_unigram
from aster.training.dataset import Window


class ToyTokenizer:
    # 0='a', 1='猫' (three UTF-8 bytes), 2='!', BOS=3, EOS=4
    bos_id, eos_id, vocab_size = 3, 4, 5
    model = SimpleNamespace(vocab={0: b'a', 1: '猫'.encode('utf-8'), 2: b'!'})

    def encode(self, text, add_bos=False, add_eos=False):
        codes = {'a': 0, '猫': 1, '!': 2}
        return ([self.bos_id] if add_bos else []) + [codes[c] for c in text] + ([self.eos_id] if add_eos else [])


@pytest.fixture
def example():
    tok = ToyTokenizer()
    train = [Window((3, 1, 0, 4), 't1', 'ja', 0), Window((3, 0, 4), 't2', 'en', 0)]
    dev = [Window((3, 0, 1, 4), 'd1', 'ja', 0), Window((3, 1, 4), 'd2', 'en', 0)]
    return tok, train, dev


def test_unigram_fit_only_train_and_laplace_normalized(example):
    tok, train, dev = example
    baseline = fit_unigram(train, tok)
    assert baseline['target_counts'] == [2, 1, 0, 0, 2]
    assert baseline['train_target_tokens'] == 5
    assert sum((n + 1) / 10 for n in baseline['target_counts']) == pytest.approx(1)
    scored = score_unigram(dev, tok, baseline)
    assert baseline['target_counts'] == [2, 1, 0, 0, 2]
    assert scored['all']['target_tokens'] == 5
    assert scored['all']['text_bytes'] == 7
    assert scored['all']['text_target_tokens'] == 3
    assert scored['all']['eos_target_tokens'] == 2
    text_nll = -math.log(3 / 10) - 2 * math.log(2 / 10)
    assert scored['all']['text_bits_per_byte'] == pytest.approx(text_nll / (7 * math.log(2)))
    assert sum(s['total_nll_nats'] for s in scored['by_domain'].values()) == pytest.approx(scored['all']['total_nll_nats'])
    assert sum(s['text_bytes'] for s in scored['by_domain'].values()) == 7


def test_score_model_masks_unequal_length_padding(example):
    tok, _, dev = example

    class Flat(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, tok.vocab_size)

    model = Flat()
    model.train()
    scored = score_model(model, dev, tok, batch_size=2)
    assert model.training
    assert scored['all']['target_tokens'] == 5
    assert scored['all']['text_bytes'] == 7
    assert scored['all']['total_nll_nats'] == pytest.approx(5 * math.log(5), rel=1e-6)
    assert scored['all']['text_bits_per_byte'] == pytest.approx(3 * math.log(5) / (7 * math.log(2)), rel=1e-6)


def test_rare_and_invalid_unigram_and_special_tokens(example):
    tok, train, dev = example
    baseline = fit_unigram(train, tok)
    score_unigram([Window((3, 2, 4), 'unseen', 'ja', 0)], tok, baseline)
    with pytest.raises(ValueError, match='BOS'):
        fit_unigram([Window((3, 3), 'bad', 'ja', 0)], tok)
    with pytest.raises(ValueError, match='counts'):
        score_unigram(dev, tok, {**baseline, 'target_counts': [-1] * 5})
    with pytest.raises(ValueError, match='target'):
        score_unigram([Window((3, 9), 'bad', 'ja', 0)], tok, baseline)


def test_bpb_excludes_eos_and_handles_empty_text(example):
    tok, _, _ = example
    train = [Window((3, 4), 'empty', 'ja', 0)]
    baseline = fit_unigram(train, tok)
    metric = score_unigram(train, tok, baseline)['all']
    assert metric['target_tokens'] == metric['eos_target_tokens'] == 1
    assert metric['text_bytes'] == 0
    assert metric['text_bits_per_byte'] is None


def _build_view(root: Path):
    files = {}
    for split, name, text in (('train', 'a', '猫a'), ('dev', 'b', '猫!'), ('test', 'c', 'a猫')):
        path = f'{split}/{name}.txt'
        files[path] = text.encode('utf-8')
    rows = [{'split': split, 'record_id': name, 'text_path': f'{split}/{name}.txt',
             'text_sha256': hashlib.sha256(files[f'{split}/{name}.txt']).hexdigest(),
             'leakage_group': name, 'domain': 'ja'}
            for split, name in [('train', 'a'), ('dev', 'b'), ('test', 'c')]]
    files['samples.jsonl'] = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    manifest = json.dumps({'schema_version': 'aster-training-view-0',
                           'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}},
                          sort_keys=True, separators=(',', ':')).encode()
    view = root / 'data/training' / hashlib.sha256(manifest).hexdigest()
    for name, data in files.items():
        file = view / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    (view / 'manifest.json').write_bytes(manifest)
    return view


def test_preflight_does_not_read_test_bytes(tmp_path, monkeypatch):
    from aster.training import lang_gate3 as gate
    view = _build_view(tmp_path)
    (tmp_path / 'configs').mkdir()
    config = {'mode': 'pilot', 'steps': 200, 'eval_every': 50, 'context_length': 128}
    config_bytes = json.dumps(config).encode()
    (tmp_path / 'configs/pilot.json').write_bytes(config_bytes)
    spec = {'schema_version': gate.SCHEMA, 'training_view_id': view.name,
            'tokenizer_artifact_id': 'fake-tokenizer', 'vocab_size': 5,
            'pilot_config_path': 'configs/pilot.json', 'pilot_config': config,
            'source_git_blobs': {'configs/pilot.json': git_blob_sha(config_bytes)}}
    spec_path = tmp_path / 'configs/gate3.json'
    spec_path.write_text(json.dumps(spec))
    tok = ToyTokenizer()
    monkeypatch.setattr(gate, 'load_training_tokenizer', lambda *a, **kw: (tok, {'fixture': True}))
    original_read = Path.read_bytes

    def guarded_read(path):
        if '/test/' in str(path):
            raise AssertionError('Sealed test bytes must never be read')
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    _, _, windows, audit = preflight(tmp_path, spec_path)
    assert audit['samples_by_split'] == {'train': 1, 'dev': 1, 'test': 1}
    assert all(w.record_id != 'c' for ww in windows.values() for w in ww)
    assert audit['sealed_test_text_read'] is False


def test_preflight_rejects_config_and_leakage(tmp_path, monkeypatch):
    from aster.training import lang_gate3 as gate
    view = _build_view(tmp_path)
    (tmp_path / 'configs').mkdir()
    cfg = {'mode': 'pilot', 'steps': 200, 'eval_every': 50, 'context_length': 128}
    cbytes = json.dumps(cfg).encode()
    (tmp_path / 'configs/pilot.json').write_bytes(cbytes)
    spec = {'schema_version': gate.SCHEMA, 'training_view_id': view.name,
            'tokenizer_artifact_id': 'fake-tokenizer', 'vocab_size': 5,
            'pilot_config_path': 'configs/pilot.json', 'pilot_config': cfg,
            'source_git_blobs': {'configs/pilot.json': git_blob_sha(cbytes)}}
    spec_file = tmp_path / 'configs/gate.json'
    spec_file.write_text(json.dumps(spec))
    monkeypatch.setattr(gate, 'load_training_tokenizer', lambda *a, **kw: (ToyTokenizer(), {'fixture': True}))
    preflight(tmp_path, spec_file)
    (tmp_path / 'configs/pilot.json').write_text('{}')
    with pytest.raises(ValueError, match='Pinned'):
        preflight(tmp_path, spec_file)
    (tmp_path / 'configs/pilot.json').write_bytes(cbytes)
    rows_path = view / 'samples.jsonl'
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    rows[1]['leakage_group'] = rows[0]['leakage_group']
    rows_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = json.loads((view / 'manifest.json').read_text())
    manifest['files']['samples.jsonl'] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    rewritten = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    (view / 'manifest.json').write_bytes(rewritten)
    moved = view.rename(view.parent / hashlib.sha256(rewritten).hexdigest())
    spec['training_view_id'] = moved.name
    spec_file.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='Cross-split'):
        preflight(tmp_path, spec_file)


def test_chunked_windows_conserve_utf8_bytes_and_eos(example):
    tok, train, _ = example
    # One document BOS,猫,a,!,EOS partitioned into context=2 windows.
    pieces = [Window((3, 1, 0), 'doc', 'ja', 0), Window((0, 2, 4), 'doc', 'ja', 2)]
    scored = score_unigram(pieces, tok, fit_unigram(train, tok))['all']
    assert scored['target_tokens'] == 4
    assert scored['text_bytes'] == len('猫a!'.encode('utf-8')) == 5
    assert scored['eos_target_tokens'] == 1

    class Flat(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, tok.vocab_size)

    measured1 = score_model(Flat(), pieces, tok, batch_size=1)
    measured2 = score_model(Flat(), pieces, tok, batch_size=2)
    assert measured1['all']['text_bits_per_byte'] == pytest.approx(measured2['all']['text_bits_per_byte'])


def test_training_windows_never_open_sealed_test(tmp_path, monkeypatch):
    from aster.training.dataset import prepare_windows
    from aster.training.tokenizer_run import verify_view

    view = _build_view(tmp_path)
    original_bytes = Path.read_bytes
    original_text = Path.read_text

    def guarded_bytes(path):
        if '/test/' in str(path):
            raise AssertionError('Training must not read sealed-test bytes')
        return original_bytes(path)

    def guarded_text(path, *args, **kwargs):
        if '/test/' in str(path):
            raise AssertionError('Training must not read sealed-test text')
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', guarded_bytes)
    monkeypatch.setattr(Path, 'read_text', guarded_text)
    splits = prepare_windows(view, ToyTokenizer(), 2)
    assert splits['train'] and splits['dev']
    assert all(w.record_id != 'c' for windows in splits.values() for w in windows)
    # Full-integrity default for non-training callers remains unchanged.
    with pytest.raises(AssertionError, match='sealed-test bytes'):
        verify_view(view)


def test_training_view_detects_tampered_train_bytes(tmp_path):
    from aster.training.dataset import prepare_windows
    view = _build_view(tmp_path)
    (view / 'train/a.txt').write_text('aa', encoding='utf-8')
    with pytest.raises(ValueError, match='Training file hash'):
        prepare_windows(view, ToyTokenizer(), 2)


def test_original_view_verifier_still_detects_test_corruption(tmp_path):
    from aster.training.tokenizer_run import verify_view
    view = _build_view(tmp_path)
    (view / 'test/c.txt').write_text('aa', encoding='utf-8')
    with pytest.raises(ValueError, match='Training file hash'):
        verify_view(view)
    # Intentional train/dev-only mode remains sealed-text blind;
    # the manifest still pins the test record identity.
    assert len(verify_view(view, verify_test_text=False)) == 3
