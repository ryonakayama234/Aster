"""Gate 4: immutable, read-only step-0/200 Japanese generation comparison.

Private continuations are written ONLY under the existing local Run. Stdout
contains aggregate counts; no private training text or generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from aster.inference.generate import describe_ids, generate_ids
from aster.model.checkpoint import load_checkpoint
from aster.training.lang_gate3 import (
    preflight, validate_run_observations, verify_checkpoint_observation,
    verify_run_sources, verify_scoring_dependencies,
)

RUN_ID = 'f884ea724e854ab09ed6a3944f03f8f2'
REPORT_SHA256 = '08f1c62ee949e45ccc32f67f5ded9c9c48044804c15587d45672571e8ba555f1'
POLICY = 'greedy; BOS excluded; EOS stops; sliding context'
MAX_NEW_TOKENS = 64
STEPS = (0, 200)
PROMPTS = (
    '朝、窓を開けると',
    '雨がやんだので',
    '机の上には',
    '昨日読んだ本には',
    '静かな駅で',
    'それから、私は',
)
MATCH_THRESHOLD_BYTES = 32
PRIVATE_OUTPUT = 'lang-gate4-private.json'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + path.name)
    return value


def read_frozen_report(run: Path) -> dict:
    raw = (run / 'lang-gate3-report.json').read_bytes()
    if sha256(raw) != REPORT_SHA256:
        raise ValueError('Gate 3 report digest differs from observed frozen Run')
    report = json.loads(raw)
    if (report.get('run_id') != RUN_ID or report.get('sealed_test_scored') is not False
            or report.get('reload_exact_match') is not True
            or report.get('mode') != 'read_only_report'):
        raise ValueError('Gate 3 report status or Run identity mismatch')
    entries = report.get('checkpoint_evaluations')
    if not isinstance(entries, list) or [x.get('step') for x in entries] != [0, 50, 100, 150, 200]:
        raise ValueError('Gate 3 checkpoint cadence mismatch')
    for x in entries:
        if not isinstance(x.get('checkpoint_sha256'), str) or len(x['checkpoint_sha256']) != 64:
            raise ValueError('Gate 3 checkpoint digest missing')
    return report


def has_repeated_trigram(ids: list[int]) -> bool:
    # A simple warning flag, not an assessment of syntax or semantics.
    triples = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
    return len(triples) != len(set(triples))


def has_long_training_match(completion: bytes, training_chunks: tuple[bytes, ...]) -> bool:
    """Any exact 32-byte span in a train window; neither proof nor exclusion of memorization."""
    if len(completion) < MATCH_THRESHOLD_BYTES:
        return False
    return any(completion[i:i+MATCH_THRESHOLD_BYTES] in chunk
               for i in range(len(completion)-MATCH_THRESHOLD_BYTES+1)
               for chunk in training_chunks)


def token_bytes(tokenizer, ids: list[int] | tuple[int, ...]) -> bytes:
    return b''.join(tokenizer.model.vocab[token] for token in ids if token in tokenizer.model.vocab)


def verify_checkpoint(run: Path, entry: dict, obs: dict, experiment: dict,
                      spec: dict, tokenizer_payload_id: str):
    step = entry['step']
    if obs['step'] != step or obs['checkpoint'] != f'checkpoint-{step:06d}.pt':
        raise ValueError('Unmatched Run checkpoint step')
    if (entry.get('checkpoint_sha256') != obs.get('checkpoint_sha256')
            or entry.get('tokens_seen') != obs.get('tokens_seen')):
        raise ValueError('Gate 3 report and Run checkpoint observations disagree')
    path = run / obs['checkpoint']
    snapshot = path.read_bytes()
    if sha256(snapshot) != entry['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint content SHA-256 mismatch')
    # Torch accepts a file-like object: infer from the *same verified bytes*.
    # Reopening a path after checking its hash would admit a file-swap race.
    model, tokenizer, payload = load_checkpoint(io.BytesIO(snapshot), tokenizer_payload_id)
    verify_checkpoint_observation(payload, obs, experiment, spec, model)
    if model.training:
        raise ValueError('Checkpoint must load in evaluation mode')
    return model, tokenizer


def write_private_once(path: Path, payload: dict) -> None:
    """No overwrites; 0600 and no stdout leak even on a second invocation."""
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def audit(root: Path) -> dict:
    root = root.resolve()
    run = root / 'runs' / RUN_ID
    if run.is_symlink() or run.resolve().parent != (root / 'runs').resolve():
        raise ValueError('Unexpected Run location')
    output = run / PRIVATE_OUTPUT
    if output.exists() or output.is_symlink():
        raise FileExistsError('Private Gate 4 audit already exists; never overwrite')
    report = read_frozen_report(run)
    spec, tokenizer, windows, preflight_summary = preflight(root, root / 'configs/lang-v0-gate3.json')
    if (report.get('training_view_id') != preflight_summary['view_id']
            or report.get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']
            or report.get('pilot_config') != spec['pilot_config']
            or report.get('preflight') != preflight_summary
            or preflight_summary['sealed_test_text_read'] is not False):
        raise ValueError('Gate 3 report and frozen local preflight differ')
    torch.set_num_threads(spec['pilot_config']['threads'])
    torch.use_deterministic_algorithms(True)
    experiment = load_json(run / 'experiment.json')
    summary = load_json(run / 'run.json')
    bundle = load_json(run / 'training-bundle.json')
    if (experiment.get('config') != spec['pilot_config']
            or summary.get('status') != 'completed' or summary.get('run_id') != RUN_ID
            or bundle.get('status') != 'completed' or bundle.get('run_id') != RUN_ID
            or bundle.get('experiment') != experiment
            or experiment.get('tokenizer_id') != preflight_summary['tokenizer_payload_sha256']
            or summary.get('inputs', {}).get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']
            or experiment.get('tokenizer_artifact_id') != spec['tokenizer_artifact_id']):
        raise ValueError('Run provenance mismatch')
    verify_scoring_dependencies(experiment, str(torch.__version__), sys.version)
    verify_run_sources(root, spec, experiment.get('code_sha256'))
    observations = validate_run_observations(bundle.get('observations'), experiment, summary, bundle)
    chunks = tuple(token_bytes(tokenizer, list(w.ids)) for w in windows['train'])
    entries = report['checkpoint_evaluations']
    comparisons = []
    counters: dict[str, dict[str, int]] = {}
    for step in STEPS:
        entry = next(x for x in entries if x['step'] == step)
        obs = next(x for x in observations if x['step'] == step)
        model, checkpoint_tokenizer = verify_checkpoint(
            run, entry, obs, experiment, spec, preflight_summary['tokenizer_payload_sha256'])
        results = []
        for index, prompt in enumerate(PROMPTS):
            prompt_ids = checkpoint_tokenizer.encode(prompt, add_bos=True)
            generated = generate_ids(model, checkpoint_tokenizer, prompt_ids, MAX_NEW_TOKENS)
            description = describe_ids(checkpoint_tokenizer, generated)
            raw = token_bytes(checkpoint_tokenizer, generated)
            results.append({
                'prompt_index': index, 'prompt': prompt,
                'continuation': description['text'],
                'utf8_valid': description['utf8_valid'],
                'generated_tokens': len(generated),
                'ended_with_eos': bool(generated and generated[-1] == checkpoint_tokenizer.eos_id),
                'repeated_token_trigram': has_repeated_trigram(generated),
                'train_exact_32_byte_match': has_long_training_match(raw, chunks),
                'prompt_in_train_window': any(prompt.encode('utf-8') in chunk for chunk in chunks),
            })
        comparisons.append({'step': step, 'checkpoint_sha256': entry['checkpoint_sha256'], 'results': results})
        counters[str(step)] = {
            'outputs': len(results),
            'valid_utf8': sum(r['utf8_valid'] for r in results),
            'repeated_trigram': sum(r['repeated_token_trigram'] for r in results),
            'train_32byte_match': sum(r['train_exact_32_byte_match'] for r in results),
            'eos_stopped': sum(r['ended_with_eos'] for r in results),
            'prompt_in_train': sum(r['prompt_in_train_window'] for r in results),
        }
    private = {
        'schema_version': 'aster-lang-gate4-private-0', 'run_id': RUN_ID,
        'gate3_report_sha256': REPORT_SHA256,
        'policy': POLICY, 'max_new_tokens': MAX_NEW_TOKENS,
        'prompts': list(PROMPTS), 'checkpoints': comparisons,
        'limitations': 'Only one dev document; this exact-match scan is within train token windows, '
                       'so it can miss copying across boundaries. Repetition/overlap flags are not '
                       'proof of memorization or conversational ability. Private; do not upload.',
    }
    write_private_once(output, private)
    return {'status': 'generation_audit_completed', 'run_id': RUN_ID,
            'gate3_report_sha256': REPORT_SHA256,
            'pair_count': len(PROMPTS), 'decoding': POLICY,
            'max_new_tokens': MAX_NEW_TOKENS, 'metrics_by_step': counters,
            'sealed_test_text_read': False, 'training_updates': 0,
            'private_result_path': str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
