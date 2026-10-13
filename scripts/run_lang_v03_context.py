"""One read-only LANG v0.3 Trial on the preserved step-0/200 private TinyLM.

Stdout is aggregate-only. Sources, original Runs, Corpus and weights are never
modified. New private diagnostic output is exclusive and cannot be overwritten.
"""
import sys

# sys is built in. Refuse before importing even stdlib modules that a local
# scripts/json.py could shadow. -I removes script/CWD/PYTHONPATH search roots.
if __name__ == '__main__' and not sys.flags.isolated:
    raise SystemExit('Use the existing Python environment with -I: python -I scripts/run_lang_v03_context.py ...')

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'scripts/run_lang_v03_context.py',
    'scripts/run_lang_v01_utf8.py',
    'scripts/run_lang_v0_gate4.py',
    'src/aster/training/lang_context.py',
    'docs/LangContextUse-v03.md',
)
OUTPUT_NAME = 'lang-v03-context-diagnostic'

STAGE_NOTES = {
    'source_attestation': '実行するGit版と、手元のコードの一致を確認しています。',
    'module_loading': '確認済みコードとPython依存を読み込んでいます。',
    'source_run_path': '元の保存済みRunの場所を確認しています。',
    'existing_output': '診断Runが既に作られていないか確認しています。',
    'source_report': '元のGate 3レポートの存在と識別情報を確認しています。',
    'corpus_preflight': '固定Corpus・Tokenizer・設定を確認しています。',
    'verified_windows': '学習と評価の入力を、確認した本文から再構成しています。',
    'run_records': '元の学習記録と固定設定の一致を確認しています。',
    'dependency_versions': '学習時と現在のPyTorch版の一致を確認しています。',
    'run_sources': '学習時の実装と現在の実装の一致を確認しています。',
    'training_selection': '実際に学習した入力の一覧を確認しています。',
    'baseline_replay': '保存済みの頻度基準と入力数の一致を確認しています。',
    'checkpoint_0': '学習前のcheckpointの識別情報を確認しています。',
    'checkpoint_200': '学習後のcheckpointの識別情報を確認しています。',
    'model_scoring': 'TinyLMを同じ評価位置で採点しています。',
    'prefix_replay': '文脈診断の元入力と通常の採点が一致するか確認しています。',
    'publication': '新しい診断Runへ結果を保存しています。',
    'unknown': '停止箇所を特定できませんでした。',
}


def blocked_result(error: Exception, progress: dict) -> dict:
    """Only controlled codes/hints go to stdout, never exception text/tracebacks."""
    stage = progress.get('stage', 'unknown')
    if stage not in STAGE_NOTES:
        stage = 'unknown'
    code, note = 'GATE_CHECK_FAILED', STAGE_NOTES[stage] + ' この確認を通過できませんでした。'
    message = str(error)  # Matching only; never echo private exception content.
    known = (
        ('Checkout differs from reviewed commit', 'SOURCE_REVISION_MISMATCH', '指定したGit版と手元の版が異なります。'),
        ('Tracked checkout is dirty', 'TRACKED_CHANGES', '未コミットの変更があるため停止しました。変更はそのまま保護しています。'),
        ('Untracked importable code', 'UNTRACKED_IMPORT_CODE', 'srcまたはscriptsに未追跡のコードがあります。削除せずに停止しました。'),
        ('Training and evaluation PyTorch versions differ', 'PYTORCH_VERSION_MISMATCH', '学習時と現在のPyTorch版が異なります。表示された版を確認してください。'),
        ('Frozen Gate 3 score replay mismatch', 'SCORE_REPLAY_MISMATCH', '保存済みの評価値と今回の採点が一致しません。'),
        ('Prefix original score differs', 'PREFIX_REPLAY_MISMATCH', '文脈診断の元入力の採点が、通常の採点と一致しません。'),
        ('Matched target accounting differs', 'TARGET_ACCOUNTING_MISMATCH', '保存済みの評価対象数と今回の対象数が一致しません。'),
    )
    for needle, candidate, explanation in known:
        if needle in message:
            code, note = candidate, explanation
            break
    if isinstance(error, FileNotFoundError):
        code, note = 'MISSING_LOCAL_FILE', STAGE_NOTES[stage] + ' 必要なローカルファイルが見つかりません。'
    elif isinstance(error, FileExistsError):
        code, note = 'EXISTING_DIAGNOSTIC', '診断Runが既に存在します。上書きや再実行はしていません。'
    elif isinstance(error, PermissionError):
        code, note = 'FILE_PERMISSION', 'ファイルの読み込みまたは新しい結果の保存が許可されていません。'
    elif isinstance(error, subprocess.CalledProcessError):
        code, note = 'GIT_CHECK_FAILED', 'Gitによる版の確認に失敗しました。'
    elif isinstance(error, ImportError):
        code, note = 'IMPORT_FAILED', '確認済みコードまたはPython依存を読み込めませんでした。'
    elif isinstance(error, json.JSONDecodeError):
        code, note = 'INVALID_METADATA_JSON', '記録ファイルのJSONを読み取れませんでした。'
    result = {'status': 'BLOCKED', 'stage': stage, 'code': code, 'note': note,
              'training_updates': 0, 'model_scoring_performed': progress.get('model_scoring_performed') is True,
              'check_only': progress.get('check_only') is True}
    if code == 'PYTORCH_VERSION_MISMATCH':
        for key in ('training_torch', 'evaluation_torch'):
            value = progress.get(key)
            if isinstance(value, str) and re.fullmatch(r'\d+\.\d+\.\d+(?:\+cpu)?', value):
                result[key] = value
    if isinstance(error, FileNotFoundError) and error.filename:
        name = Path(error.filename).name
        if name in {'lang-gate3-report.json', 'manifest.json', 'samples.jsonl',
                    'checkpoint-000000.pt', 'checkpoint-000200.pt', 'experiment.json',
                    'run.json', 'training-bundle.json', 'training-windows.json'}:
            result['required_file'] = name
    return result


def existing_output_summary(output: Path) -> dict:
    """Inspect only fixed metadata names; never score, overwrite or read private text."""
    files = ('inputs.json', 'run.json', 'private-report.json', 'aggregate.json')
    present = {name: False for name in files}
    status = 'unknown'
    if output.is_dir() and not output.is_symlink():
        present = {name: (output / name).is_file() and not (output / name).is_symlink() for name in files}
        if present['run.json']:
            try:
                # Bound reads; parse only the status. Do not open private-report.
                with (output / 'run.json').open('rb') as stream:
                    record = json.loads(stream.read(1024 * 1024 + 1))
                candidate = record.get('status') if isinstance(record, dict) else None
                if candidate in ('completed', 'failed', 'running', 'interrupted'):
                    status = candidate
            except (ValueError, OSError):
                status = 'unreadable'
    return {'status': 'EXISTING_DIAGNOSTIC', 'stage': 'existing_output',
            'existing_run_status': status, 'existing_files': present,
            'check_only': True, 'training_updates': 0, 'model_scoring_performed': False,
            'note': '既存の診断を確認しました。結果を削除せず、この表示を共有してください。再実行はしていません。'}


def attest(root: Path, expected_commit: str) -> tuple[dict[str, str], dict[str, bytes]]:
    if root != ROOT.resolve():
        raise ValueError('Runner and --root must refer to the same checkout')
    if not re.fullmatch(r'[0-9a-f]{40}', expected_commit):
        raise ValueError('Exact reviewed commit required')

    def git(*args: str) -> bytes:
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError('Not the exact checkout root')
    if git('rev-parse', 'HEAD').decode().strip() != expected_commit:
        raise ValueError('Checkout differs from reviewed commit')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Tracked checkout is dirty; preserve local work')
    # The verified helper adds src/ and scripts/ to sys.path. Reject untracked
    # importable code there, including ignored files and directory symlinks.
    # Ordinary __pycache__ files are not sourceless module candidates and our
    # project source loader never executes them. Top-level .pyc is rejected.
    untracked = git('ls-files', '-z', '--others', '--exclude-standard', '--', 'src', 'scripts')
    ignored = git('ls-files', '-z', '--others', '--ignored', '--exclude-standard', '--', 'src', 'scripts')
    for raw in (untracked + ignored).split(b'\x00'):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        path = root / relative
        if path.is_symlink() or (relative.suffix.lower() in ('.py', '.so', '.pyd', '.pyo', '.pyc')
                                 and '__pycache__' not in relative.parts):
            raise ValueError('Untracked importable code in diagnostic search roots; preserve local work')
        if relative.suffix.lower() == '.py':
            raise ValueError('Untracked importable code in diagnostic search roots; preserve local work')
    hashes, snapshots = {}, {}
    for relative in SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or symlinked diagnostic source')
        snapshot = path.read_bytes()
        if snapshot != git('show', f'{expected_commit}:{relative}'):
            raise ValueError('Diagnostic source differs from Git')
        snapshots[relative] = snapshot
        hashes[relative] = hashlib.sha256(snapshot).hexdigest()
    return hashes, snapshots


def execute_snapshot(snapshot: bytes, path: Path, name: str):
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), ''
    sys.modules[name] = module
    exec(compile(snapshot, str(path), 'exec'), module.__dict__)
    return module


def load_modules(root: Path, commit: str, hashes: dict[str, str], snapshots: dict[str, bytes]):
    # Reuse the established verified-source finder, not its v0.1 experiment.
    # Compile the helper itself from checked bytes, bypassing stale .pyc files.
    if any(n == 'aster' or n.startswith('aster.') for n in sys.modules):
        raise ValueError('Aster imported before source attestation')
    helper = execute_snapshot(snapshots['scripts/run_lang_v01_utf8.py'],
                              root / 'scripts/run_lang_v01_utf8.py', '_lang_v03_source_loader')
    sys.meta_path.insert(0, helper.PinnedAsterFinder(root, commit, hashes))
    gate = execute_snapshot(snapshots['scripts/run_lang_v0_gate4.py'],
                            root / 'scripts/run_lang_v0_gate4.py', '_lang_v03_gate4')
    context = importlib.import_module('aster.training.lang_context')
    scoring = importlib.import_module('aster.training.lang_gate3')
    return gate, context, scoring


def write_once(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def verify_matched_targets(actual: dict, frozen: dict) -> None:
    for key in ('target_tokens', 'text_target_tokens', 'text_bytes', 'eos_target_tokens'):
        if actual['all'][key] != frozen['all'][key]:
            raise ValueError('Matched target accounting differs from Gate 3')


def verify_matched_scores(actual: dict, frozen: dict) -> None:
    verify_matched_targets(actual, frozen)
    for key in ('mean_nll_nats', 'text_bits_per_byte'):
        if not math.isclose(actual['all'][key], frozen['all'][key], abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError('Frozen Gate 3 score replay mismatch')


def audit(root: Path, expected_commit: str, *, check_only: bool = False,
          progress: dict | None = None) -> dict:
    progress = {} if progress is None else progress
    progress.update(stage='source_attestation', check_only=check_only, model_scoring_performed=False)
    root = root.resolve()
    hashes, snapshots = attest(root, expected_commit)
    progress['stage'] = 'module_loading'
    gate, context, scoring = load_modules(root, expected_commit, hashes, snapshots)
    import torch

    progress['stage'] = 'source_run_path'
    source_run = root / 'runs' / gate.RUN_ID
    if (root / 'runs').is_symlink() or source_run.is_symlink() or source_run.resolve().parent != (root / 'runs').resolve():
        raise ValueError('Unexpected source Run path')
    output = root / 'runs' / OUTPUT_NAME
    progress['stage'] = 'existing_output'
    if output.exists() or output.is_symlink():
        if check_only:
            return existing_output_summary(output)
        raise FileExistsError('Diagnostic Run already exists; never overwrite or rerun')
    progress['stage'] = 'source_report'
    frozen = gate.read_frozen_report(source_run)
    progress['stage'] = 'corpus_preflight'
    spec, tokenizer, original_windows, preflight = gate.preflight(root, root / 'configs/lang-v0-gate3.json')
    if (frozen.get('preflight') != preflight
            or frozen.get('training_view_id') != spec['training_view_id']
            or frozen.get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']
            or frozen.get('pilot_config') != spec['pilot_config']):
        raise ValueError('Frozen report/preflight identity mismatch')
    progress['stage'] = 'verified_windows'
    windows = context.verified_windows(root / 'data/training' / spec['training_view_id'],
                                       tokenizer, spec['pilot_config']['context_length'])
    if windows != original_windows:
        raise ValueError('Verified snapshots differ from preflight windows')
    if len({w.record_id for w in windows['dev']}) != 1:
        raise ValueError('This Trial requires the original one-document dev')
    progress['stage'] = 'run_records'
    experiment = gate.load_json(source_run / 'experiment.json')
    summary = gate.load_json(source_run / 'run.json')
    bundle = gate.load_json(source_run / 'training-bundle.json')
    if (summary.get('run_id') != gate.RUN_ID or bundle.get('run_id') != gate.RUN_ID
            or summary.get('kind') != 'pretrain' or summary.get('status') != 'completed'
            or bundle.get('status') != 'completed' or bundle.get('reload_exact_match') is not True
            or experiment.get('config') != spec['pilot_config']
            or experiment.get('view_id') != spec['training_view_id']
            or experiment.get('device') != 'cpu'
            or experiment.get('tokenizer_id') != preflight['tokenizer_payload_sha256']
            or experiment.get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']
            or summary.get('inputs', {}).get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']):
        raise ValueError('Frozen Run provenance mismatch')
    progress.update(stage='dependency_versions', training_torch=experiment.get('torch_version'),
                    evaluation_torch=str(torch.__version__))
    dependencies = scoring.verify_scoring_dependencies(experiment, str(torch.__version__), sys.version)
    progress['stage'] = 'run_sources'
    scoring.verify_run_sources(root, spec, experiment.get('code_sha256'))
    scoring.configure_scoring_runtime(spec['pilot_config'])
    observations = scoring.validate_run_observations(bundle.get('observations'), experiment, summary, bundle)
    progress['stage'] = 'training_selection'
    selected = json.loads((source_run / 'training-windows.json').read_bytes())
    if selected != [{'record_id': w.record_id, 'offset': w.offset, 'domain': w.domain} for w in windows['train']]:
        raise ValueError('Actual training selection mismatch')
    if check_only:
        progress['stage'] = 'baseline_replay'
        train_counts = scoring.fit_unigram(windows['train'], tokenizer)
        if scoring.digest(scoring.json_bytes(train_counts)) != frozen['baseline']['counts_sha256']:
            raise ValueError('Frozen train unigram differs')
        # Count target identities/masks/bytes without asking any model to predict.
        targets = [t for w in windows['dev'] for t in w.ids[1:]]
        text_targets = [t for t in targets if t != tokenizer.eos_id]
        counts = {'all': {'target_tokens': len(targets), 'text_target_tokens': len(text_targets),
                          'text_bytes': sum(len(tokenizer.model.vocab[t]) for t in text_targets),
                          'eos_target_tokens': len(targets) - len(text_targets)}}
        verify_matched_targets(counts, frozen['baseline']['dev'])
        for step in gate.STEPS:
            progress['stage'] = f'checkpoint_{step}'
            entry = next(e for e in frozen['checkpoint_evaluations'] if e['step'] == step)
            obs = next(o for o in observations if o['step'] == step)
            verify_matched_targets(counts, entry['dev'])
            gate.verify_checkpoint(source_run, entry, obs, experiment, spec,
                                   preflight['tokenizer_payload_sha256'])
        return {'status': 'PREFLIGHT_PASSED', 'check_only': True,
                'training_updates': 0, 'model_scoring_performed': False,
                'diagnostic_output_created': False, 'sealed_test_text_read': False,
                'note': 'ソース・入力・実行記録・checkpointの確認は通過しました。モデル採点と結果作成は行っていません。'}
    progress['stage'] = 'baseline_replay'
    baseline = context.fit_bigram(windows['train'], tokenizer)
    unigram = scoring.score_unigram(windows['dev'], tokenizer, baseline['unigram'])
    if scoring.digest(scoring.json_bytes(baseline['unigram'])) != frozen['baseline']['counts_sha256']:
        raise ValueError('Frozen train unigram differs')
    verify_matched_scores(unigram, frozen['baseline']['dev'])
    bigram_rows = context.bigram_losses(windows['dev'], tokenizer, baseline)
    bigram = context.score_rows(windows['dev'], tokenizer, bigram_rows)
    cases = context.select_prefix_cases(windows['dev'], tokenizer)
    # Seal selected positions before any TinyLM scoring or prefix intervention.
    progress['stage'] = 'publication'
    output.mkdir(mode=0o700)
    manifest = {'schema_version': 'aster-lang-context-inputs-0', 'experiment_mode': 'trial',
                'cycle_id': OUTPUT_NAME, 'evidence_scope': 'development_diagnostic',
                'source_run_id': gate.RUN_ID, 'gate3_report_sha256': gate.REPORT_SHA256,
                'source_git_sha': expected_commit, 'preflight': preflight,
                'target_positions_sha256': context.target_digest(windows['dev']),
                'bigram_smoothing': context.SMOOTHING, 'prefix_cases': cases,
                'prefix_policy': 'reverse earlier tokens; preserve leading BOS and last token',
                'numeric_tolerance': context.NUMERIC_TOLERANCE,
                'training_updates': 0, 'sealed_test_text_read': False}
    write_once(output / 'inputs.json', manifest)
    began = time.perf_counter()
    try:
        checkpoints, public_models = [], {}
        for step in gate.STEPS:
            progress['stage'] = f'checkpoint_{step}'
            entry = next(e for e in frozen['checkpoint_evaluations'] if e['step'] == step)
            obs = next(o for o in observations if o['step'] == step)
            model, checkpoint_tokenizer = gate.verify_checkpoint(
                source_run, entry, obs, experiment, spec, preflight['tokenizer_payload_sha256'])
            progress.update(stage='model_scoring', model_scoring_performed=True)
            rows = context.model_losses(model, windows['dev'], checkpoint_tokenizer,
                                         spec['pilot_config']['batch_size'])
            scores = context.score_rows(windows['dev'], checkpoint_tokenizer, rows)
            verify_matched_scores(scores, entry['dev'])
            scoring.verify_recorded_loss(obs, 'dev', scores)
            progress['stage'] = 'prefix_replay'
            sensitivity = context.prefix_sensitivity(model, windows['dev'], checkpoint_tokenizer, cases)
            for case in sensitivity:
                original_nll = rows[case['window_index']][case['target_position']]
                if not math.isclose(-case['original_target_logp'], original_nll,
                                    abs_tol=1e-5, rel_tol=1e-5):
                    raise ValueError('Prefix original score differs from matched model target')
            checkpoints.append({'step': step, 'checkpoint_sha256': entry['checkpoint_sha256'],
                                'dev': scores, 'per_position_nll': rows, 'prefix_results': sensitivity})
            public_models[str(step)] = {'dev': scores['all'],
                                         'prefix_cases': len(sensitivity),
                                         'detected_prefix_changes': sum(s['detected_change'] for s in sensitivity),
                                         'rank_changes': sum(s['rank_changed'] for s in sensitivity)}
        trained = checkpoints[-1]
        delta = bigram['all']['mean_nll_nats'] - trained['dev']['all']['mean_nll_nats']
        diagnostic = context.diagnostic_summary(delta, trained['prefix_results'])
        predecessors = [u for w in windows['dev'] for u in w.ids[:-1]]
        public = {'status': 'context_diagnostic_completed', 'source_git_sha': expected_commit,
                  'source_run_id': gate.RUN_ID, 'experiment_mode': 'trial',
                  'evidence_scope': 'development_diagnostic', 'independent_dev_documents': 1,
                  'training_updates': 0, 'sealed_test_text_read': False,
                  'unigram_dev': unigram['all'], 'bigram_dev': bigram['all'],
                  'bigram_smoothing': context.SMOOTHING,
                  'dev_unseen_predecessor_targets': sum(baseline['rows'][u] == 0 for u in predecessors),
                  'dev_rare_predecessor_targets_le16': sum(baseline['rows'][u] <= 16 for u in predecessors),
                  'train_target_tokens': baseline['unigram']['train_target_tokens'],
                  'parent_training_token_presentations': observations[-1]['tokens_seen'],
                  'models': public_models, 'diagnostic': diagnostic,
                  'seconds': time.perf_counter() - began}
        private = {'schema_version': 'aster-lang-context-private-0', 'inputs': manifest,
                   'inputs_sha256': hashlib.sha256((output / 'inputs.json').read_bytes()).hexdigest(),
                   'dependency_versions': dependencies, 'verified_source_sha256': hashes,
                   'bigram_counts_sha256': context.baseline_digest(baseline),
                   'bigram_per_position_nll': bigram_rows, 'checkpoints': checkpoints,
                   'aggregate': public,
                   'limitations': 'One dev document; reversed prefixes may be OOD. Sensitivity is not '
                                  'semantic understanding. Diagnostic only, no model promotion.'}
        progress['stage'] = 'publication'
        write_once(output / 'private-report.json', private)
        public['private_report_sha256'] = hashlib.sha256((output / 'private-report.json').read_bytes()).hexdigest()
        write_once(output / 'aggregate.json', public)
        write_once(output / 'run.json', {'schema_version': 'aster-run-0', 'run_id': OUTPUT_NAME,
                                         'kind': 'lang_context_diagnostic', 'status': 'completed',
                                         'inputs': manifest, 'outputs': ['aggregate.json', 'private-report.json']})
        return public
    except BaseException as error:
        failure = blocked_result(error, progress) if isinstance(error, Exception) else {
            'status': 'INTERRUPTED', 'stage': progress['stage']}
        write_once(output / 'run.json', {'schema_version': 'aster-run-0', 'run_id': OUTPUT_NAME,
                                         'kind': 'lang_context_diagnostic', 'status': 'failed',
                                         'inputs': manifest, 'blocked': failure,
                                         'note': 'Inspect locally; do not overwrite or rerun.'})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--check-only', action='store_true',
                        help='Read-only input/metadata check; no model scoring or new diagnostic Run')
    args = parser.parse_args()
    progress: dict = {}
    try:
        result = audit(args.root, args.expected_commit, check_only=args.check_only, progress=progress)
    except Exception as error:
        print(json.dumps(blocked_result(error, progress), ensure_ascii=False, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
