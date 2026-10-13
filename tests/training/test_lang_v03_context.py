"""Synthetic-only context controls, snapshot integrity and full diagnostic Run."""
import importlib.util
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from aster.corpus.pipeline import digest, json_bytes
from aster.model.checkpoint import save_checkpoint
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.tokenizer.artifact import train_aster_tokenizer
from aster.training.dataset import Window, tokenizer_payload
from aster.training.lang_gate3 import fit_unigram, git_blob_sha, score_model, score_unigram
from aster.training import lang_context as ctx

ROOT = Path(__file__).resolve().parents[2]


class ToyTokenizer:
    bos_id, eos_id, vocab_size = 3, 4, 5
    model = SimpleNamespace(vocab={0: b'a', 1: '猫'.encode(), 2: b'!'})


def example():
    t = ToyTokenizer()
    train = [Window((3, 1, 0, 4), 't1', 'ja', 0), Window((3, 0, 4), 't2', 'ja', 0)]
    dev = [Window((3, 0, 1, 4), 'd1', 'ja', 0), Window((3, 2, 4), 'd1', 'ja', 3)]
    return t, train, dev


def test_bigram_is_train_only_normalized_and_backs_off():
    t, train, dev = example()
    baseline = ctx.fit_bigram(train, t)
    assert baseline['unigram'] == fit_unigram(train, t)
    assert sum(baseline['rows'].values()) == 5
    before = ctx.baseline_digest(baseline)
    losses = ctx.bigram_losses(dev, t, baseline)
    assert ctx.baseline_digest(baseline) == before
    assert baseline['pairs'][0, 1] == 0  # Dev-only pair was not fitted.
    q0 = [(n + 1) / 10 for n in baseline['unigram']['target_counts']]
    for u in range(t.vocab_size):
        probs = [(baseline['pairs'][u, target] + 16 * q0[target]) /
                 (baseline['rows'][u] + 16) for target in range(t.vocab_size)]
        assert sum(probs) == pytest.approx(1)
        if baseline['rows'][u] == 0:
            assert probs == pytest.approx(q0)
    assert losses[1][-1] == pytest.approx(-math.log(q0[t.eos_id]))


def test_bigram_does_not_add_document_or_window_boundary_pairs():
    t = ToyTokenizer()
    train = [Window((3, 0, 1), 't1', 'ja', 0), Window((1, 2, 4), 't1', 'ja', 2),
             Window((3, 0, 4), 't2', 'ja', 0)]
    b = ctx.fit_bigram(train, t)
    assert b['pairs'][1, 1] == 0
    assert b['pairs'][4, 3] == 0
    assert b['pairs'][1, 2] == 1
    assert sum(b['rows'].values()) == sum(len(w.ids) - 1 for w in train)
    with pytest.raises(ValueError, match='BOS'):
        ctx.fit_bigram([Window((0, 3), 'bad', 'ja', 0)], t)
    with pytest.raises(ValueError, match='predecessor'):
        ctx.fit_bigram([Window((9, 0), 'bad', 'ja', 0)], t)


def test_model_bigram_match_target_masks_bpb_eos_and_original_evaluator():
    t, train, dev = example()
    model = TinyLM(ModelConfig(5, context_length=8, width=8, heads=2, layers=1))
    old_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    rows = ctx.model_losses(model, dev, t, 2)
    actual = ctx.score_rows(dev, t, rows)
    original = score_model(model, dev, t, 1)
    assert actual['all']['mean_nll_nats'] == pytest.approx(original['all']['mean_nll_nats'], abs=1e-6)
    assert model.training
    assert all(torch.equal(old_state[k], v) for k, v in model.state_dict().items())
    bigram = ctx.score_rows(dev, t, ctx.bigram_losses(dev, t, ctx.fit_bigram(train, t)))
    for field, value in [('target_tokens', 5), ('text_target_tokens', 3), ('text_bytes', 5), ('eos_target_tokens', 2)]:
        assert bigram['all'][field] == actual['all'][field] == value
    assert actual['all']['text_bits_per_byte'] == pytest.approx(
        (actual['all']['total_nll_nats'] - actual['all']['eos_nll_nats']) / (5 * math.log(2)))
    with pytest.raises(ValueError, match='length'):
        ctx.score_rows(dev, t, [[0], [0]])


def test_prefix_selection_has_no_target_or_score_dependency():
    t = ToyTokenizer()
    w = [Window((3, 0, 1, 2, 0, 1, 2, 4), 'd', 'ja', 0)]
    changed_target = [Window((3, 0, 1, 2, 0, 1, 2, 1), 'd', 'ja', 0)]
    cases = ctx.select_prefix_cases(w, t)
    assert cases == ctx.select_prefix_cases(changed_target, t)
    assert 0 < len(cases) <= 6
    assert ctx.select_prefix_cases([Window((3, 0, 0, 0, 0, 4), 'd', 'ja', 0)], t) == []
    for c in cases:
        prefix = w[0].ids[:c['target_position'] + 1]
        modified = ctx.reverse_earlier(prefix, t.bos_id)
        assert modified[0] == prefix[0] == t.bos_id
        assert modified[-1] == prefix[-1]
        assert len(modified) == len(prefix)
        assert sorted(modified) == sorted(prefix)
        assert modified != prefix


@pytest.mark.parametrize('contextual,expected', [(False, 0), (True, 1)])
def test_prefix_sensitivity_distinguishes_last_token_only_model(contextual, expected):
    t = ToyTokenizer()
    windows = [Window((3, 0, 1, 2, 0, 4), 'd', 'ja', 0)]
    class Controlled(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(*x.shape, t.vocab_size)
            logits[:, -1, 0] = x[:, 1] if contextual else x[:, -1]
            return logits
    model = Controlled()
    model.train()
    results = ctx.prefix_sensitivity(model, windows, t, [{'window_index': 0, 'target_position': 3}])
    assert sum(r['detected_change'] for r in results) == expected
    assert model.training
    assert results[0]['prefix_length'] == 4
    assert results[0]['original_input_sha256'] != results[0]['changed_input_sha256']


def build_view(root, tokenizer, dev_text='fedcba fedcba'):
    data = [('train', 'train', 'abcdef abcdef'), ('dev', 'dev', dev_text), ('test', 'sealed', 'NEVER READ')]
    files, samples = {}, []
    for split, record, text in data:
        name = f'{split}/{record}.txt'
        files[name] = text.encode()
        samples.append({'split': split, 'record_id': record, 'text_path': name,
                        'text_sha256': digest(files[name]), 'leakage_group': record, 'domain': 'ja'})
    files['samples.jsonl'] = ''.join(json.dumps(s) + '\n' for s in samples).encode()
    manifest = json_bytes({'schema_version': 'aster-training-view-0', 'files': {k: digest(v) for k, v in files.items()}})
    view = root / 'data/training' / digest(manifest)
    for name, raw in {**files, 'manifest.json': manifest}.items():
        p = view / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    return view


def test_snapshot_windows_roundtrip_and_never_read_sealed_test(tmp_path, monkeypatch):
    t = train_aster_tokenizer(['abcdef'], target_vocab_size=256)
    view = build_view(tmp_path, t)
    read = Path.read_bytes
    def guarded(p):
        if '/test/' in str(p):
            raise AssertionError('Sealed test read')
        return read(p)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    windows = ctx.verified_windows(view, t, 8)
    assert set(windows) == {'train', 'dev'}
    assert sum(len(w.ids) - 1 for w in windows['dev']) == 14
    (view / 'dev/dev.txt').write_text('tampered')
    with pytest.raises(ValueError, match='snapshot mismatch'):
        ctx.verified_windows(view, t, 8)


def test_snapshot_windows_match_original_universal_newline_semantics(tmp_path):
    t = train_aster_tokenizer(['abcdef'], target_vocab_size=256)
    view = build_view(tmp_path, t, dev_text='a\r\nb\rc')
    windows = ctx.verified_windows(view, t, 128)
    assert windows['dev'][0].ids == tuple(t.encode('a\nb\nc', add_bos=True, add_eos=True))


def runner():
    spec = importlib.util.spec_from_file_location('lang_v03_runner', ROOT / 'scripts/run_lang_v03_context.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_source_attestation_and_no_overwrite(tmp_path, monkeypatch):
    r = runner()
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'SOURCE_FILES', ('source.py',))
    (tmp_path / 'source.py').write_text('print(1)\n')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'source.py'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    sha = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD']).decode().strip()
    hashes, snapshots = r.attest(tmp_path, sha)
    assert hashes['source.py'] == digest(snapshots['source.py'])
    with pytest.raises(ValueError, match='same checkout'):
        r.attest(tmp_path / 'elsewhere', sha)
    with pytest.raises(ValueError, match='reviewed commit'):
        r.attest(tmp_path, '0' * 40)
    (tmp_path / 'source.py').write_text('tampered')
    with pytest.raises(ValueError, match='dirty'):
        r.attest(tmp_path, sha)
    output = tmp_path / 'result.json'
    r.write_once(output, {'aggregate': True})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        r.write_once(output, {'overwrite': True})


def test_verified_helper_ignores_changed_disk_or_bytecode(tmp_path):
    r = runner()
    path = tmp_path / 'helper.py'
    path.write_text('raise RuntimeError("unverified disk source")')
    module = r.execute_snapshot(b'answer = 42\n', path, '_synthetic_context_loader')
    assert module.answer == 42


@pytest.mark.parametrize('relative,ignored', [
    ('scripts/torch.py', False), ('src/torch.py', False),
    ('src/torch/__init__.py', False), ('src/torch.py', True),
    ('scripts/torch.pyc', False), ('src/torch', False),
])
def test_attestation_rejects_untracked_import_shadows(tmp_path, monkeypatch, relative, ignored):
    r = runner()
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'SOURCE_FILES', ('source.py',))
    (tmp_path / 'source.py').write_text('pass\n')
    if ignored:
        (tmp_path / '.gitignore').write_text('src/torch.py\n')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Synthetic',
                    '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    sha = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD']).decode().strip()
    shadow = tmp_path / relative
    shadow.parent.mkdir(parents=True, exist_ok=True)
    if relative == 'src/torch':
        target = tmp_path / 'outside-package'
        target.mkdir()
        (target / '__init__.py').write_text('raise RuntimeError("must not execute")')
        shadow.symlink_to(target, target_is_directory=True)
        before = shadow.readlink()
    else:
        shadow.write_bytes(b'raise RuntimeError("must not execute")')
        before = shadow.read_bytes()
    with pytest.raises(ValueError, match='Untracked importable'):
        r.attest(tmp_path, sha)
    assert (shadow.readlink() if shadow.is_symlink() else shadow.read_bytes()) == before


def test_nonisolated_cli_stops_before_stdlib_or_project_imports():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_lang_v03_context.py')], capture_output=True)
    assert result.returncode != 0
    assert b'with -I' in result.stderr
    assert b'Traceback' not in result.stderr


@pytest.mark.parametrize('isolated', [False, True])
def test_bootstrap_never_executes_local_stdlib_shadow(tmp_path, isolated):
    script = tmp_path / 'run_context.py'
    script.write_bytes((ROOT / 'scripts/run_lang_v03_context.py').read_bytes())
    (tmp_path / 'json.py').write_text('raise RuntimeError("UNREVIEWED_JSON_EXECUTED")')
    (tmp_path / '__future__.py').write_text('raise RuntimeError("UNREVIEWED_FUTURE_EXECUTED")')
    command = [sys.executable] + (['-I'] if isolated else []) + [str(script)]
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0  # Nonisolated refusal or argparse missing required SHA.
    assert b'UNREVIEWED_' not in result.stderr
    assert b'Traceback' not in result.stderr


@pytest.mark.parametrize('corruption', [None, 'checkpoint', 'runtime', 'target_count'])
def test_full_synthetic_private_run_and_frozen_source_preservation(tmp_path, monkeypatch, corruption):
    r = runner()
    from aster.training import lang_gate3 as scoring
    gs = importlib.util.spec_from_file_location('_synthetic_gate4_context', ROOT / 'scripts/run_lang_v0_gate4.py')
    gate = importlib.util.module_from_spec(gs)
    gs.loader.exec_module(gate)
    tokenizer = train_aster_tokenizer(['abcdef'], target_vocab_size=256)
    view = build_view(tmp_path, tokenizer)
    windows = ctx.verified_windows(view, tokenizer, 8)
    config = {'mode': 'pilot', 'steps': 200, 'eval_every': 50, 'context_length': 8,
              'width': 8, 'heads': 2, 'layers': 1, 'batch_size': 2, 'threads': 1}
    payload = tokenizer_payload(tokenizer)
    preflight = {'view_id': view.name, 'tokenizer_payload_sha256': digest(json_bytes(payload)),
                 'sealed_test_text_read': False}
    source = tmp_path / 'src/aster/training/pretrain.py'
    source.parent.mkdir(parents=True)
    source.write_text('synthetic trainer fixture')
    model = TinyLM(ModelConfig(tokenizer.vocab_size, 8, 8, 2, 1))
    spec = {'training_view_id': view.name, 'pilot_config': config, 'tokenizer_artifact_id': 'synthetic',
            'vocab_size': tokenizer.vocab_size, 'model_parameter_count': sum(p.numel() for p in model.parameters()),
            'source_git_blobs': {'src/aster/training/pretrain.py': git_blob_sha(source.read_bytes())}}
    experiment = {'view_id': view.name, 'config': config, 'device': 'cpu',
                  'torch_version': str(torch.__version__), 'python_version': 'synthetic Python',
                  'tokenizer_id': preflight['tokenizer_payload_sha256'], 'tokenizer_artifact_id': 'synthetic',
                  'parameter_count': spec['model_parameter_count'],
                  'code_sha256': {'training/pretrain.py': digest(source.read_bytes())}}
    summary = {'run_id': gate.RUN_ID, 'kind': 'pretrain', 'status': 'completed',
               'inputs': {'view_id': view.name, 'config': config, 'tokenizer_artifact_id': 'synthetic'}}
    run = tmp_path / 'runs' / gate.RUN_ID
    run.mkdir(parents=True)
    observations, entries = [], []
    for step in (0, 50, 100, 150, 200):
        name = f'checkpoint-{step:06d}.pt'
        save_checkpoint(run / name, model, payload, {**experiment, 'step': step, 'tokens_seen': step * 10})
        scores = score_model(model, windows['dev'], tokenizer, 2)
        obs = {'step': step, 'tokens_seen': step * 10, 'checkpoint': name,
               'checkpoint_sha256': digest((run / name).read_bytes()),
               'dev': {'loss': scores['all']['mean_nll_nats'], 'target_tokens': scores['all']['target_tokens']}}
        observations.append(obs)
        entries.append({**obs, 'dev': scores})
    baseline = fit_unigram(windows['train'], tokenizer)
    report = {'run_id': gate.RUN_ID, 'mode': 'read_only_report', 'reload_exact_match': True,
              'sealed_test_scored': False, 'preflight': preflight, 'training_view_id': view.name,
              'tokenizer_artifact_id': 'synthetic', 'pilot_config': config,
              'checkpoint_evaluations': entries,
              'baseline': {'counts_sha256': digest(json_bytes(baseline)),
                           'dev': score_unigram(windows['dev'], tokenizer, baseline)}}
    records = {'experiment.json': experiment, 'run.json': summary,
               'training-bundle.json': {'run_id': gate.RUN_ID, 'status': 'completed', 'reload_exact_match': True,
                                        'experiment': experiment, 'observations': observations},
               'training-windows.json': [{'record_id': w.record_id, 'offset': w.offset, 'domain': w.domain} for w in windows['train']],
               'lang-gate3-report.json': report}
    for name, value in records.items():
        (run / name).write_bytes(json_bytes(value))
    gate.REPORT_SHA256 = digest((run / 'lang-gate3-report.json').read_bytes())
    monkeypatch.setattr(gate, 'preflight', lambda *a: (spec, tokenizer, windows, preflight))
    monkeypatch.setattr(r, 'attest', lambda *a: ({}, {}))
    monkeypatch.setattr(r, 'load_modules', lambda *a: (gate, ctx, scoring))
    before = {p.name: digest(p.read_bytes()) for p in run.iterdir()}
    read = Path.read_bytes
    def guarded(p):
        if '/test/' in str(p):
            raise AssertionError('Sealed test read')
        return read(p)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    if corruption == 'checkpoint':
        (run / 'checkpoint-000200.pt').write_bytes(b'tampered weights')
    elif corruption == 'runtime':
        experiment['torch_version'] = 'incompatible+cpu'
        (run / 'experiment.json').write_bytes(json_bytes(experiment))
        records['training-bundle.json']['experiment'] = experiment
        (run / 'training-bundle.json').write_bytes(json_bytes(records['training-bundle.json']))
    elif corruption == 'target_count':
        report['baseline']['dev']['all']['target_tokens'] += 1
        (run / 'lang-gate3-report.json').write_bytes(json_bytes(report))
        gate.REPORT_SHA256 = digest((run / 'lang-gate3-report.json').read_bytes())
    if corruption:
        progress = {}
        with pytest.raises(ValueError) as caught:
            r.audit(tmp_path, 'a' * 40, check_only=True, progress=progress)
        block = r.blocked_result(caught.value, progress)
        assert block['stage'] == {'checkpoint': 'checkpoint_200', 'runtime': 'dependency_versions',
                                  'target_count': 'baseline_replay'}[corruption]
        assert block['check_only'] is True
        assert block['model_scoring_performed'] is False
        assert not (tmp_path / 'runs' / r.OUTPUT_NAME).exists()
        with pytest.raises(ValueError):
            r.audit(tmp_path, 'a' * 40)
        output = tmp_path / 'runs' / r.OUTPUT_NAME
        assert not (output / 'aggregate.json').exists()
        if corruption == 'checkpoint':
            failed = json.loads((output / 'run.json').read_bytes())
            assert failed['status'] == 'failed'
            assert failed['blocked']['stage'] == 'checkpoint_200'
        return
    # The check-only path must never use either neural or statistical predictors.
    with monkeypatch.context() as patch:
        def forbidden(*a, **kw):
            raise AssertionError('Check-only must not score any model')
        patch.setattr(ctx, 'model_losses', forbidden)
        patch.setattr(ctx, 'prefix_sensitivity', forbidden)
        patch.setattr(ctx, 'fit_bigram', forbidden)
        patch.setattr(scoring, 'score_unigram', forbidden)
        checked = r.audit(tmp_path, 'a' * 40, check_only=True)
    assert checked['status'] == 'PREFLIGHT_PASSED'
    assert checked['diagnostic_output_created'] is False
    assert not (tmp_path / 'runs' / r.OUTPUT_NAME).exists()
    assert before == {p.name: digest(p.read_bytes()) for p in run.iterdir()}
    public = r.audit(tmp_path, 'a' * 40)
    assert public['status'] == 'context_diagnostic_completed'
    assert public['training_updates'] == 0
    assert public['diagnostic']['capability_pass'] is False
    assert public['models']['200']['prefix_cases'] == 6
    assert before == {p.name: digest(p.read_bytes()) for p in run.iterdir()}
    output = tmp_path / 'runs' / r.OUTPUT_NAME
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir())
    private = json.loads((output / 'private-report.json').read_bytes())
    assert private['inputs']['prefix_cases'] == ctx.select_prefix_cases(windows['dev'], tokenizer)
    assert 'per_position_nll' not in json.dumps(public)
    with pytest.raises(FileExistsError):
        r.audit(tmp_path, 'a' * 40)
    existing_before = {p.name: digest(p.read_bytes()) for p in output.iterdir()}
    existing = r.audit(tmp_path, 'a' * 40, check_only=True)
    assert existing['status'] == 'EXISTING_DIAGNOSTIC'
    assert existing['existing_run_status'] == 'completed'
    assert existing_before == {p.name: digest(p.read_bytes()) for p in output.iterdir()}


@pytest.mark.parametrize('error,stage,code', [
    (FileNotFoundError(2, 'PRIVATE_TEXT_SECRET', '/private/source/lang-gate3-report.json'), 'source_report', 'MISSING_LOCAL_FILE'),
    (ValueError('Training and evaluation PyTorch versions differ'), 'dependency_versions', 'PYTORCH_VERSION_MISMATCH'),
    (ValueError('Prefix original score differs from matched model target'), 'prefix_replay', 'PREFIX_REPLAY_MISMATCH'),
    (FileExistsError('PRIVATE_TEXT_SECRET'), 'existing_output', 'EXISTING_DIAGNOSTIC'),
    (RuntimeError('PRIVATE_TEXT_SECRET'), 'model_scoring', 'GATE_CHECK_FAILED'),
])
def test_blocked_output_is_actionable_without_private_exception_leak(error, stage, code):
    r = runner()
    result = r.blocked_result(error, {'stage': stage, 'check_only': True,
                                     'training_torch': '2.14.0+cpu', 'evaluation_torch': '2.14.1+cpu'})
    assert result['stage'] == stage
    assert result['code'] == code
    assert result['note']
    assert 'PRIVATE_TEXT_SECRET' not in json.dumps(result)
    assert '/private/source' not in json.dumps(result)
    if code == 'PYTORCH_VERSION_MISMATCH':
        assert result['training_torch'] == '2.14.0+cpu'
        assert result['evaluation_torch'] == '2.14.1+cpu'


def test_existing_output_inspection_never_opens_private_report_or_relabels_bad_status(tmp_path, monkeypatch):
    r = runner()
    (tmp_path / 'run.json').write_text(json.dumps({'status': 'PRIVATE_TEXT_SECRET'}))
    (tmp_path / 'private-report.json').write_text('PRIVATE_TEXT_SECRET')
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == 'private-report.json':
            raise AssertionError('Private report content must not be opened')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    result = r.existing_output_summary(tmp_path)
    assert result['existing_run_status'] == 'unknown'
    assert 'PRIVATE_TEXT_SECRET' not in json.dumps(result)


@pytest.mark.parametrize('python_status', [0, 2])
def test_check_launcher_restores_branch_and_preserves_python_exit_status(tmp_path, python_status):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()
    (repo / 'tracked.txt').write_text('original')
    git('add', '.')
    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'original')
    original_branch, original_sha = git('symbolic-ref', '--short', 'HEAD'), git('rev-parse', 'HEAD')
    (repo / 'scripts').mkdir()
    shell = (ROOT / 'scripts/check_lang_v03_context.sh').read_bytes()
    (repo / 'scripts/check_lang_v03_context.sh').write_bytes(shell)
    (repo / 'tracked.txt').write_text('reviewed')
    git('add', '.')
    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'reviewed')
    revision = git('rev-parse', 'HEAD')
    git('reset', '--hard', original_sha)  # Disposable synthetic repository only.
    (repo / '.venv/bin').mkdir(parents=True)
    fake_python = repo / '.venv/bin/python'
    fake_python.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@"\nexit {python_status}\n')
    fake_python.chmod(0o700)
    result = subprocess.run(['bash', '-s', '--', revision], input=shell, cwd=repo, capture_output=True)
    assert result.returncode == python_status
    assert b'--check-only' in result.stdout
    assert b'-I' in result.stdout
    assert git('symbolic-ref', '--short', 'HEAD') == original_branch
    assert git('rev-parse', 'HEAD') == original_sha
    assert (repo / 'tracked.txt').read_text() == 'original'


def test_check_launcher_refuses_to_overwrite_ignored_local_file(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()
    (repo / '.gitignore').write_text('private.txt\n.venv/\n')
    git('add', '.')
    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'original')
    original_branch, original_sha = git('symbolic-ref', '--short', 'HEAD'), git('rev-parse', 'HEAD')
    (repo / 'private.txt').write_text('tracked content in target revision')
    git('add', '-f', 'private.txt')
    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'reviewed')
    revision = git('rev-parse', 'HEAD')
    git('reset', '--hard', original_sha)  # Disposable synthetic repository only.
    (repo / 'private.txt').write_text('private local sentinel')
    (repo / '.venv/bin').mkdir(parents=True)
    fake_python = repo / '.venv/bin/python'
    fake_python.write_text('#!/bin/sh\necho PYTHON_EXECUTED\n')
    fake_python.chmod(0o700)
    shell = (ROOT / 'scripts/check_lang_v03_context.sh').read_bytes()
    result = subprocess.run(['bash', '-s', '--', revision], input=shell, cwd=repo, capture_output=True)
    assert result.returncode != 0
    assert b'PYTHON_EXECUTED' not in result.stdout
    assert git('symbolic-ref', '--short', 'HEAD') == original_branch
    assert git('rev-parse', 'HEAD') == original_sha
    assert (repo / 'private.txt').read_text() == 'private local sentinel'


@pytest.mark.parametrize('delta,changes,verdict', [
    (0.1, [True], 'observed_on_one_development_document'),
    (-0.1, [False], 'useful_multi_token_context_not_observed'),
    (0.1, [], 'mechanism_not_distinguished'),
    (-0.1, [True], 'mechanism_not_distinguished'),
])
def test_descriptive_verdict_is_never_capability_pass(delta, changes, verdict):
    result = ctx.diagnostic_summary(delta, [{'detected_change': c, 'rank_changed': False} for c in changes])
    assert result['verdict'] == verdict
    assert result['capability_pass'] is False
