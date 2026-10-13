"""One read-only LANG v0.3 Trial on the preserved step-0/200 private TinyLM.

Stdout is aggregate-only. Sources, original Runs, Corpus and weights are never
modified. New private diagnostic output is exclusive and cannot be overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
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


def verify_matched_scores(actual: dict, frozen: dict) -> None:
    for key in ('target_tokens', 'text_target_tokens', 'text_bytes', 'eos_target_tokens'):
        if actual['all'][key] != frozen['all'][key]:
            raise ValueError('Matched target accounting differs from Gate 3')
    for key in ('mean_nll_nats', 'text_bits_per_byte'):
        if not math.isclose(actual['all'][key], frozen['all'][key], abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError('Frozen Gate 3 score replay mismatch')


def audit(root: Path, expected_commit: str) -> dict:
    root = root.resolve()
    hashes, snapshots = attest(root, expected_commit)
    gate, context, scoring = load_modules(root, expected_commit, hashes, snapshots)
    import torch

    source_run = root / 'runs' / gate.RUN_ID
    if (root / 'runs').is_symlink() or source_run.is_symlink() or source_run.resolve().parent != (root / 'runs').resolve():
        raise ValueError('Unexpected source Run path')
    output = root / 'runs' / OUTPUT_NAME
    if output.exists() or output.is_symlink():
        raise FileExistsError('Diagnostic Run already exists; never overwrite or rerun')
    frozen = gate.read_frozen_report(source_run)
    spec, tokenizer, original_windows, preflight = gate.preflight(root, root / 'configs/lang-v0-gate3.json')
    if (frozen.get('preflight') != preflight
            or frozen.get('training_view_id') != spec['training_view_id']
            or frozen.get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']
            or frozen.get('pilot_config') != spec['pilot_config']):
        raise ValueError('Frozen report/preflight identity mismatch')
    windows = context.verified_windows(root / 'data/training' / spec['training_view_id'],
                                       tokenizer, spec['pilot_config']['context_length'])
    if windows != original_windows:
        raise ValueError('Verified snapshots differ from preflight windows')
    if len({w.record_id for w in windows['dev']}) != 1:
        raise ValueError('This Trial requires the original one-document dev')
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
    dependencies = scoring.verify_scoring_dependencies(experiment, str(torch.__version__), sys.version)
    scoring.verify_run_sources(root, spec, experiment.get('code_sha256'))
    scoring.configure_scoring_runtime(spec['pilot_config'])
    observations = scoring.validate_run_observations(bundle.get('observations'), experiment, summary, bundle)
    selected = json.loads((source_run / 'training-windows.json').read_bytes())
    if selected != [{'record_id': w.record_id, 'offset': w.offset, 'domain': w.domain} for w in windows['train']]:
        raise ValueError('Actual training selection mismatch')
    baseline = context.fit_bigram(windows['train'], tokenizer)
    unigram = scoring.score_unigram(windows['dev'], tokenizer, baseline['unigram'])
    if scoring.digest(scoring.json_bytes(baseline['unigram'])) != frozen['baseline']['counts_sha256']:
        raise ValueError('Frozen train unigram differs')
    verify_matched_scores(unigram, frozen['baseline']['dev'])
    bigram_rows = context.bigram_losses(windows['dev'], tokenizer, baseline)
    bigram = context.score_rows(windows['dev'], tokenizer, bigram_rows)
    cases = context.select_prefix_cases(windows['dev'], tokenizer)
    # Seal selected positions before any TinyLM scoring or prefix intervention.
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
            entry = next(e for e in frozen['checkpoint_evaluations'] if e['step'] == step)
            obs = next(o for o in observations if o['step'] == step)
            model, checkpoint_tokenizer = gate.verify_checkpoint(
                source_run, entry, obs, experiment, spec, preflight['tokenizer_payload_sha256'])
            rows = context.model_losses(model, windows['dev'], checkpoint_tokenizer,
                                         spec['pilot_config']['batch_size'])
            scores = context.score_rows(windows['dev'], checkpoint_tokenizer, rows)
            verify_matched_scores(scores, entry['dev'])
            scoring.verify_recorded_loss(obs, 'dev', scores)
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
        write_once(output / 'private-report.json', private)
        public['private_report_sha256'] = hashlib.sha256((output / 'private-report.json').read_bytes()).hexdigest()
        write_once(output / 'aggregate.json', public)
        write_once(output / 'run.json', {'schema_version': 'aster-run-0', 'run_id': OUTPUT_NAME,
                                         'kind': 'lang_context_diagnostic', 'status': 'completed',
                                         'inputs': manifest, 'outputs': ['aggregate.json', 'private-report.json']})
        return public
    except BaseException:
        write_once(output / 'run.json', {'schema_version': 'aster-run-0', 'run_id': OUTPUT_NAME,
                                         'kind': 'lang_context_diagnostic', 'status': 'failed',
                                         'inputs': manifest, 'note': 'Inspect locally; do not overwrite or rerun.'})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    try:
        result = audit(args.root, args.expected_commit)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, subprocess.CalledProcessError):
        # No private path/text/hash from exception messages enters shareable stdout.
        print(json.dumps({'status': 'BLOCKED', 'training_updates': 0,
                          'note': 'Source, input, runtime, replay, or existing-output gate failed.'}))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
