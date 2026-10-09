"""LANG v0 Gate 3A: read-only held-out scoring with an explicit train-only baseline.

This module never fits or updates a neural model. Test/holdout document bytes are
never read by the Gate 3 preflight or report path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch.nn import functional as F

from aster.corpus.pipeline import digest, json_bytes
from aster.model.checkpoint import load_checkpoint
from aster.training.dataset import IGNORE_INDEX, Window, collate, load_training_tokenizer

SCHEMA = 'aster-lang-gate3-0'


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path.name}')
    return result


def _in_dir(base: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Unsafe relative file path')
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()) or (base / relative).is_symlink():
        raise ValueError('Path escapes the selected view')
    return target


def _view_samples_no_test_read(view: Path) -> list[dict[str, Any]]:
    """Audit listed files and train/dev bytes without touching sealed test contents."""
    manifest_data = (view / 'manifest.json').read_bytes()
    if digest(manifest_data) != view.name:
        raise ValueError('Training view manifest identity mismatch')
    manifest = json.loads(manifest_data)
    if manifest.get('schema_version') != 'aster-training-view-0':
        raise ValueError('Unknown training-view schema')
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise ValueError('Missing training-view file hashes')
    actual = {p.relative_to(view).as_posix() for p in view.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ValueError('Training-view file inventory mismatch')
    for name, sha in files.items():
        file = _in_dir(view, name)
        if not file.is_file():
            raise ValueError('Training-view file missing')
        if not name.startswith('test/') and digest(file.read_bytes()) != sha:
            raise ValueError('Training-view file hash mismatch: ' + name)
    samples = [json.loads(line) for line in (view / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    if not samples:
        raise ValueError('Empty selected view')
    seen_ids: set[str] = set()
    indexed_paths: set[str] = set()
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for row in samples:
        split, record_id, name = row['split'], row['record_id'], row['text_path']
        if split not in ('train', 'dev', 'test') or name != f'{split}/{record_id}.txt':
            raise ValueError('Unexpected split or path')
        if record_id in seen_ids or name in indexed_paths or name not in files:
            raise ValueError('Duplicate/unindexed record')
        seen_ids.add(record_id)
        indexed_paths.add(name)
        if files[name] != row['text_sha256']:
            raise ValueError('Record text hash and manifest mismatch')
        for mapping, value in ((groups, row['leakage_group']), (hashes, row['text_sha256'])):
            if value in mapping and mapping[value] != split:
                raise ValueError('Cross-split group or exact-text leakage')
            mapping[value] = split
    if indexed_paths != {name for name in files if name.endswith('.txt')}:
        raise ValueError('Unindexed view text')
    return samples


def verify_evaluator_source(spec: dict[str, Any], source: Path = Path(__file__)) -> str:
    """Refuse to evaluate unless this evaluator matches the frozen spec."""
    expected_sha = spec.get('evaluator_git_blob_sha')
    actual_sha = git_blob_sha(source.read_bytes())
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ValueError('Pinned evaluator source changed or hash is missing')
    return actual_sha


def preflight(root: Path, spec_path: Path) -> tuple[dict[str, Any], Any, dict[str, list[Window]], dict[str, Any]]:
    root = root.resolve()
    spec = _load_json(spec_path)
    if spec.get('schema_version') != SCHEMA:
        raise ValueError('Unknown Gate 3 schema')
    verify_evaluator_source(spec)
    for name, expected_sha in spec['source_git_blobs'].items():
        if git_blob_sha(_in_dir(root, name).read_bytes()) != expected_sha:
            raise ValueError('Pinned source/config changed: ' + name)
    config = _load_json(_in_dir(root, spec['pilot_config_path']))
    if config != spec['pilot_config']:
        raise ValueError('Pilot config drift')
    if config['mode'] != 'pilot' or config['steps'] != 200 or config['eval_every'] != 50:
        raise ValueError('Gate 3 only permits the frozen pilot')
    view = root / 'data/training' / spec['training_view_id']
    tokenizer_dir = root / 'artifacts/tokenizers/lang-v0' / spec['tokenizer_artifact_id']
    samples = _view_samples_no_test_read(view)
    tokenizer, payload = load_training_tokenizer(tokenizer_dir, view.name,
                                                  expected_artifact_id=spec['tokenizer_artifact_id'])
    if tokenizer.vocab_size != spec['vocab_size']:
        raise ValueError('Frozen Tokenizer vocabulary size mismatch')
    windows: dict[str, list[Window]] = {'train': [], 'dev': []}
    splits = Counter(row['split'] for row in samples)
    if not splits['train'] or not splits['dev']:
        raise ValueError('Both train and dev must exist')
    for row in sorted(samples, key=lambda x: x['record_id']):
        if row['split'] == 'test':
            continue  # Never read sealed test text, even to generate previews.
        text = _in_dir(view, row['text_path']).read_text(encoding='utf-8')
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        length = config['context_length']
        for offset in range(0, len(ids) - 1, length):
            windows[row['split']].append(Window(tuple(ids[offset:offset+length+1]),
                                                  row['record_id'], row['domain'], offset))
    return spec, tokenizer, windows, {'view_id': view.name, 'samples_by_split': dict(splits),
                                      'windows_by_split': {k: len(v) for k, v in windows.items()},
                                      'tokenizer_payload_sha256': digest(json_bytes(payload)),
                                      'sealed_test_text_read': False}


@dataclass
class NLLStats:
    total_nll: float = 0.0
    targets: int = 0
    text_nll: float = 0.0
    text_targets: int = 0
    text_bytes: int = 0
    eos_nll: float = 0.0
    eos_targets: int = 0

    def add(self, token_id: int, nll: float, tokenizer: Any) -> None:
        if not math.isfinite(nll) or nll < -1e-5:
            raise ValueError('Invalid token negative log-likelihood')
        if token_id == tokenizer.bos_id:
            raise ValueError('BOS must not appear as target')
        self.total_nll += nll
        self.targets += 1
        if token_id == tokenizer.eos_id:
            self.eos_nll += nll
            self.eos_targets += 1
        else:
            raw = tokenizer.model.vocab.get(token_id)
            if not isinstance(raw, bytes) or not raw:
                raise ValueError('Missing/empty BPE token bytes')
            self.text_nll += nll
            self.text_targets += 1
            self.text_bytes += len(raw)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            'total_nll_nats': self.total_nll,
            'target_tokens': self.targets,
            'mean_nll_nats': self.total_nll / self.targets if self.targets else None,
            'text_nll_nats': self.text_nll,
            'text_target_tokens': self.text_targets,
            'text_bytes': self.text_bytes,
            'text_bits_per_byte': self.text_nll / (math.log(2) * self.text_bytes) if self.text_bytes else None,
            'eos_nll_nats': self.eos_nll,
            'eos_target_tokens': self.eos_targets,
        }


def _score(windows: list[Window], tokenizer: Any, losses: list[list[float]]) -> dict[str, Any]:
    if len(losses) != len(windows):
        raise ValueError('Window-score count mismatch')
    overall = NLLStats()
    by_domain: dict[str, NLLStats] = defaultdict(NLLStats)
    for window, row in zip(windows, losses):
        targets = window.ids[1:]
        if len(row) != len(targets):
            raise ValueError('Mask/target length mismatch')
        for token, nll in zip(targets, row):
            overall.add(token, nll, tokenizer)
            by_domain[window.domain].add(token, nll, tokenizer)
    if overall.targets == 0:
        raise ValueError('No evaluation targets')
    return {'all': overall.as_dict(), 'by_domain': {k: v.as_dict() for k,v in sorted(by_domain.items())},
            'windows': len(windows)}


def fit_unigram(train: list[Window], tokenizer: Any) -> dict[str, Any]:
    counts = [0] * tokenizer.vocab_size
    for window in train:
        for token in window.ids[1:]:
            if token == tokenizer.bos_id:
                raise ValueError('BOS cannot be a training target')
            if token < 0 or token >= len(counts):
                raise ValueError('Invalid training target token')
            counts[token] += 1
    if not sum(counts):
        raise ValueError('No unigram training targets')
    return {'schema_version': 'aster-lang-unigram-0', 'smoothing': 1,
            'vocab_size': tokenizer.vocab_size, 'target_counts': counts,
            'train_target_tokens': sum(counts)}


def score_unigram(windows: list[Window], tokenizer: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    if baseline['vocab_size'] != tokenizer.vocab_size or baseline['smoothing'] != 1:
        raise ValueError('Unigram baseline identity mismatch')
    counts = baseline['target_counts']
    if len(counts) != tokenizer.vocab_size or any(type(c) is not int or c < 0 for c in counts):
        raise ValueError('Invalid unigram counts')
    denom = sum(counts) + tokenizer.vocab_size
    losses = []
    for window in windows:
        row = []
        for token in window.ids[1:]:
            if token < 0 or token >= len(counts):
                raise ValueError('Invalid evaluation target token')
            row.append(math.log(denom) - math.log(counts[token] + 1))
        losses.append(row)
    return _score(windows, tokenizer, losses)


@torch.no_grad()
def score_model(model: torch.nn.Module, windows: list[Window], tokenizer: Any,
                batch_size: int) -> dict[str, Any]:
    if not windows or batch_size <= 0:
        raise ValueError('Invalid evaluation windows or batch size')
    was_training = model.training
    model.eval()
    losses: list[list[float]] = []
    try:
        for start in range(0, len(windows), batch_size):
            chunk = windows[start:start + batch_size]
            x, y = collate(chunk, tokenizer.eos_id)
            logits = model(x)
            per_token = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=IGNORE_INDEX,
                                        reduction='none')
            for i, window in enumerate(chunk):
                size = len(window.ids) - 1
                if not torch.all(y[i, size:] == IGNORE_INDEX):
                    raise ValueError('Unmasked padding targets')
                losses.append(per_token[i, :size].tolist())
    finally:
        model.train(was_training)
    return _score(windows, tokenizer, losses)


def verify_run_sources(root: Path, spec: dict[str, Any], code: Any) -> None:
    """Require all and only pinned model, tokenizer and trainer sources."""
    frozen_code = {name.removeprefix('src/aster/'): sha
                   for name, sha in spec['source_git_blobs'].items()
                   if name.startswith('src/aster/')}
    if not isinstance(code, dict) or not frozen_code or set(code) != set(frozen_code):
        raise ValueError('Run code provenance file set does not match frozen Gate 3')
    for relative, recorded_sha in sorted(code.items()):
        source = _in_dir(root / 'src/aster', relative)
        data = source.read_bytes()
        if digest(data) != recorded_sha:
            raise ValueError('Run source code changed since training: ' + relative)
        if git_blob_sha(data) != frozen_code[relative]:
            raise ValueError('Run source code differs from frozen Gate 3: ' + relative)


def configure_scoring_runtime(config: dict[str, Any]) -> None:
    """Match the frozen CPU training runtime before inference starts."""
    threads = config.get('threads')
    if type(threads) is not int or threads <= 0:
        raise ValueError('Frozen scoring threads must be a positive integer')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def verify_scoring_dependencies(experiment: dict[str, Any],
                                evaluation_torch_version: str,
                                evaluation_python_version: str) -> dict[str, str]:
    """Require identical PyTorch builds; record both evaluation runtimes."""
    train_torch = experiment.get('torch_version')
    train_python = experiment.get('python_version')
    if not isinstance(train_torch, str) or not train_torch:
        raise ValueError('Missing training PyTorch version')
    if not isinstance(train_python, str) or not train_python:
        raise ValueError('Missing training Python version')
    if train_torch != evaluation_torch_version:
        raise ValueError('Training and evaluation PyTorch versions differ')
    if not evaluation_python_version:
        raise ValueError('Missing evaluation Python version')
    return {'training_torch_version': train_torch,
            'evaluation_torch_version': evaluation_torch_version,
            'training_python_version': train_python,
            'evaluation_python_version': evaluation_python_version}


def validate_run_observations(
    observations: Any, experiment: dict[str, Any], summary: dict[str, Any],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reject mixed/partial Run records before reading checkpoint weights."""
    if bundle.get('experiment') != experiment:
        raise ValueError('Training bundle experiment provenance mismatch')
    inputs = summary.get('inputs')
    if not isinstance(inputs, dict) or inputs.get('view_id') != experiment.get('view_id') or inputs.get('config') != experiment.get('config'):
        raise ValueError('Run inputs do not match experiment provenance')
    steps = list(range(0, 201, 50))
    if not isinstance(observations, list) or len(observations) != len(steps):
        raise ValueError('Pilot checkpoint count differs from frozen cadence')
    previous = -1
    for obs, step in zip(observations, steps):
        if not isinstance(obs, dict) or type(obs.get('step')) is not int or obs['step'] != step:
            raise ValueError('Pilot checkpoint steps differ from frozen cadence')
        count = obs.get('tokens_seen')
        if type(count) is not int or count < 0 or (count != 0 if step == 0 else count <= previous):
            raise ValueError('Invalid or nonmonotonic tokens_seen')
        if obs.get('checkpoint') != f'checkpoint-{step:06d}.pt':
            raise ValueError('Unexpected checkpoint filename or step')
        previous = count
    return observations


def verify_checkpoint_observation(
    checkpoint: dict[str, Any], obs: dict[str, Any], experiment: dict[str, Any],
) -> None:
    """The saved weights must carry the same provenance, step and token counter."""
    expected = {**experiment, 'step': obs['step'], 'tokens_seen': obs['tokens_seen']}
    if checkpoint.get('metadata') != expected:
        raise ValueError('Checkpoint metadata/provenance or tokens_seen mismatch')


def verify_recorded_loss(obs: dict[str, Any], split: str, scored: dict[str, Any]) -> None:
    """Cross-check trainer-observed loss against independent checkpoint rescoring."""
    recorded = obs.get(split)
    if not isinstance(recorded, dict):
        raise ValueError('Missing recorded ' + split + ' evaluation')
    loss = recorded.get('loss')
    target_count = recorded.get('target_tokens')
    measured = scored['all']
    if (type(loss) not in (float, int) or not math.isfinite(loss)
            or type(target_count) is not int or target_count != measured['target_tokens']
            or not math.isclose(loss, measured['mean_nll_nats'], abs_tol=1e-5, rel_tol=1e-5)):
        raise ValueError('Recorded ' + split + ' NLL/target count differs from checkpoint rescoring')


def report_existing_run(root: Path, spec_path: Path, run_path: Path) -> Path:
    """Re-score a completed frozen pilot; never train or inspect test targets."""
    root = root.resolve()
    run_path = run_path.resolve()
    if run_path.parent != (root / 'runs').resolve():
        raise ValueError('Run must be an immediate child of root/runs')
    spec, tokenizer, windows, audit = preflight(root, spec_path)
    configure_scoring_runtime(spec['pilot_config'])
    summary = _load_json(run_path / 'run.json')
    experiment = _load_json(run_path / 'experiment.json')
    bundle = _load_json(run_path / 'training-bundle.json')
    if summary.get('kind') != 'pretrain' or summary.get('status') != 'completed':
        raise ValueError('A completed pretrain Run is required')
    if experiment.get('view_id') != audit['view_id'] or experiment.get('config') != spec['pilot_config']:
        raise ValueError('Run view/config does not match frozen Gate 3')
    if experiment.get('tokenizer_id') != audit['tokenizer_payload_sha256']:
        raise ValueError('Run Tokenizer payload mismatch')
    if bundle.get('status') != 'completed' or bundle.get('reload_exact_match') is not True:
        raise ValueError('Missing checkpoint reload verification')
    if experiment.get('device') != 'cpu':
        raise ValueError('Gate 3 requires CPU pilot')
    dependencies = verify_scoring_dependencies(experiment, str(torch.__version__), sys.version)
    if summary.get('run_id') != run_path.name or bundle.get('run_id') != run_path.name:
        raise ValueError('Run and bundle identity mismatch')
    verify_run_sources(root, spec, experiment.get('code_sha256'))
    observations = validate_run_observations(bundle.get('observations'), experiment, summary, bundle)
    selected = json.loads((run_path / 'training-windows.json').read_text(encoding='utf-8'))
    expected = [{'record_id': w.record_id, 'offset': w.offset, 'domain': w.domain}
                for w in windows['train']]
    if selected != expected:
        raise ValueError('Run selected training windows differ from frozen view')
    baseline = fit_unigram(windows['train'], tokenizer)
    reports = []
    for obs in observations:
        file = _in_dir(run_path, obs['checkpoint'])
        if digest(file.read_bytes()) != obs['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA mismatch')
        model, checkpoint_tokenizer, checkpoint = load_checkpoint(file, audit['tokenizer_payload_sha256'])
        verify_checkpoint_observation(checkpoint, obs, experiment)
        train_scored = score_model(model, windows['train'], checkpoint_tokenizer,
                                   spec['pilot_config']['batch_size'])
        dev_scored = score_model(model, windows['dev'], checkpoint_tokenizer,
                                 spec['pilot_config']['batch_size'])
        verify_recorded_loss(obs, 'train', train_scored)
        verify_recorded_loss(obs, 'dev', dev_scored)
        reports.append({'step': obs['step'], 'tokens_seen': obs['tokens_seen'],
                        'checkpoint_sha256': obs['checkpoint_sha256'],
                        'train': train_scored, 'dev': dev_scored})
    payload = {'schema_version': SCHEMA, 'mode': 'read_only_report',
               'run_id': summary['run_id'], 'training_view_id': audit['view_id'],
               'tokenizer_artifact_id': spec['tokenizer_artifact_id'],
               'pilot_config': spec['pilot_config'],
               'source_git_blobs': spec['source_git_blobs'],
               'gate3_spec_sha256': digest(spec_path.read_bytes()),
               'evaluator_git_blob': spec['evaluator_git_blob_sha'],
               'preflight': audit, 'baseline': {'fit_split': 'train',
               'counts_sha256': digest(json_bytes(baseline)), 'train_targets': baseline['train_target_tokens'],
               'dev': score_unigram(windows['dev'], tokenizer, baseline)},
               'checkpoint_evaluations': reports,
               'dependency_versions': dependencies,
               'training_seconds': summary.get('seconds'),
               'training_peak_rss_kib': summary.get('peak_rss_kib'),
               'reload_exact_match': True, 'sealed_test_scored': False,
               'note': 'Text BPB excludes EOS loss and padding; mean NLL includes EOS.'}
    output = run_path / 'lang-gate3-report.json'
    content = json_bytes(payload)
    if output.exists() and output.read_bytes() != content:
        raise ValueError('Existing Gate 3 report differs; no overwrite')
    if not output.exists():
        output.write_bytes(content)
    return output
