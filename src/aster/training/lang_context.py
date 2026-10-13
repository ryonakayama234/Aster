"""Read-only, matched-position LANG context diagnostic; never train or read test."""
from __future__ import annotations

from collections import Counter
import json
import io
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from aster.corpus.pipeline import digest, json_bytes
from aster.training.dataset import IGNORE_INDEX, Window, collate
from aster.training.lang_gate3 import _in_dir, _score, fit_unigram

SMOOTHING = 16
PREFIX_CASE_LIMIT = 6
NUMERIC_TOLERANCE = 1e-5


def verified_windows(view: Path, tokenizer: Any, context: int) -> dict[str, list[Window]]:
    """Tokenize the very train/dev bytes verified against the frozen manifest.

    Called after Gate 3 preflight (which checks split/group integrity). The
    second, snapshot-bound verification closes its hash-then-reopen race.
    Test paths are metadata only and are never opened here.
    """
    if view.is_symlink() or context <= 0:
        raise ValueError('Invalid view or context')
    manifest_bytes = (view / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != view.name:
        raise ValueError('View identity mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 'aster-training-view-0':
        raise ValueError('Unexpected view schema')
    files = manifest['files']
    index_bytes = _in_dir(view, 'samples.jsonl').read_bytes()
    if digest(index_bytes) != files.get('samples.jsonl'):
        raise ValueError('Sample index snapshot mismatch')
    rows = [json.loads(line) for line in index_bytes.decode('utf-8').splitlines() if line.strip()]
    result: dict[str, list[Window]] = {'train': [], 'dev': []}
    for row in sorted(rows, key=lambda r: r['record_id']):
        split = row['split']
        if split == 'test':
            continue
        if split not in result or row['text_path'] != f"{split}/{row['record_id']}.txt":
            raise ValueError('Unexpected text path')
        snapshot = _in_dir(view, row['text_path']).read_bytes()
        if digest(snapshot) != files.get(row['text_path']) or digest(snapshot) != row['text_sha256']:
            raise ValueError('Text snapshot mismatch')
        # Match the trainer's Path.read_text universal-newline semantics, but
        # operate on the same verified bytes rather than reopening the file.
        text = io.StringIO(snapshot.decode('utf-8'), newline=None).read()
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if ids[0] != tokenizer.bos_id or ids[-1] != tokenizer.eos_id or tokenizer.decode(ids) != text:
            raise ValueError('Tokenizer round-trip/BOS/EOS mismatch')
        for offset in range(0, len(ids) - 1, context):
            result[split].append(Window(tuple(ids[offset:offset + context + 1]),
                                        row['record_id'], row['domain'], offset))
    if not result['train'] or not result['dev']:
        raise ValueError('Empty train/dev')
    return result


def fit_bigram(train: list[Window], tokenizer: Any) -> dict[str, Any]:
    unigram = fit_unigram(train, tokenizer)
    pairs: Counter[tuple[int, int]] = Counter()
    rows: Counter[int] = Counter()
    for window in train:
        for u, t in zip(window.ids[:-1], window.ids[1:]):
            if not 0 <= u < tokenizer.vocab_size:
                raise ValueError('Invalid predecessor')
            pairs[u, t] += 1
            rows[u] += 1
    return {'schema_version': 'aster-lang-bigram-0', 'smoothing': SMOOTHING,
            'unigram': unigram, 'pairs': pairs, 'rows': rows}


def bigram_losses(windows: list[Window], tokenizer: Any, baseline: dict[str, Any]) -> list[list[float]]:
    if baseline['smoothing'] != SMOOTHING or baseline['unigram']['vocab_size'] != tokenizer.vocab_size:
        raise ValueError('Bigram identity mismatch')
    counts = baseline['unigram']['target_counts']
    denom = sum(counts) + tokenizer.vocab_size
    losses = []
    for window in windows:
        row = []
        for u, t in zip(window.ids[:-1], window.ids[1:]):
            if not 0 <= u < tokenizer.vocab_size or not 0 <= t < tokenizer.vocab_size or t == tokenizer.bos_id:
                raise ValueError('Invalid bigram token')
            q0 = (counts[t] + 1) / denom
            q1 = (baseline['pairs'][u, t] + SMOOTHING * q0) / (baseline['rows'][u] + SMOOTHING)
            row.append(-math.log(q1))
        losses.append(row)
    return losses


def baseline_digest(baseline: dict[str, Any]) -> str:
    serial = {'smoothing': baseline['smoothing'], 'unigram': baseline['unigram'],
              'pairs': [[u, t, n] for (u, t), n in sorted(baseline['pairs'].items())]}
    return digest(json_bytes(serial))


@torch.no_grad()
def model_losses(model: torch.nn.Module, windows: list[Window], tokenizer: Any,
                 batch_size: int) -> list[list[float]]:
    if not windows or batch_size <= 0:
        raise ValueError('Invalid scoring input')
    was_training = model.training
    losses: list[list[float]] = []
    model.eval()
    try:
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            x, y = collate(batch, tokenizer.eos_id)
            logits = model(x)
            per_token = F.cross_entropy(logits.transpose(1, 2), y,
                                        ignore_index=IGNORE_INDEX, reduction='none')
            for i, window in enumerate(batch):
                losses.append(per_token[i, :len(window.ids) - 1].tolist())
    finally:
        model.train(was_training)
    return losses


def score_rows(windows: list[Window], tokenizer: Any, rows: list[list[float]]) -> dict[str, Any]:
    return _score(windows, tokenizer, rows)


def target_digest(windows: list[Window]) -> str:
    return digest(json_bytes([{'record_id': w.record_id, 'offset': w.offset,
                              'targets': list(w.ids[1:])} for w in windows]))


def reverse_earlier(prefix: tuple[int, ...], bos_id: int) -> tuple[int, ...]:
    """Preserve BOS if present, last token, token multiset, length and positions."""
    if len(prefix) < 4:
        raise ValueError('Prefix too short')
    start = 1 if prefix[0] == bos_id else 0
    return prefix[:start] + tuple(reversed(prefix[start:-1])) + prefix[-1:]


def select_prefix_cases(windows: list[Window], tokenizer: Any) -> list[dict[str, int]]:
    """No model scores or target identities used to select diagnostic positions.

    Eligibility: >=4 model-visible prefix tokens, reversal changes the input.
    Sort windows as provided by verified_windows; traverse target positions
    ascending; evenly spread up to six cases over this ordered candidate list.
    """
    eligible = []
    for wi, window in enumerate(windows):
        for position in range(3, len(window.ids) - 1):
            prefix = window.ids[:position + 1]
            if reverse_earlier(prefix, tokenizer.bos_id) != prefix:
                eligible.append({'window_index': wi, 'target_position': position})
    count = min(PREFIX_CASE_LIMIT, len(eligible))
    if count == 0:
        return []
    if count == 1:
        return [eligible[0]]
    return [eligible[i * (len(eligible) - 1) // (count - 1)] for i in range(count)]


@torch.no_grad()
def prefix_sensitivity(model: torch.nn.Module, windows: list[Window], tokenizer: Any,
                       cases: list[dict[str, int]]) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    results = []
    try:
        for case in cases:
            w = windows[case['window_index']]
            position = case['target_position']
            prefix, target = w.ids[:position + 1], w.ids[position + 1]
            changed = reverse_earlier(prefix, tokenizer.bos_id)
            if changed == prefix or changed[-1] != prefix[-1] or len(changed) != len(prefix):
                raise ValueError('Prefix intervention invariant failed')
            x = torch.tensor([prefix, changed], dtype=torch.long)
            logits = model(x)[:, -1, :]
            logp = logits.log_softmax(dim=-1)[:, target]
            if not torch.isfinite(logp).all():
                raise ValueError('Nonfinite prefix scores')
            ranks = (logits > logits[:, target:target + 1]).sum(dim=-1) + 1
            delta = float(logp[1] - logp[0])
            results.append({**case, 'prefix_length': len(prefix),
                            'original_input_sha256': digest(json_bytes(list(prefix))),
                            'changed_input_sha256': digest(json_bytes(list(changed))),
                            'original_target_logp': float(logp[0]), 'changed_target_logp': float(logp[1]),
                            'delta_logp_changed_minus_original': delta,
                            'original_rank': int(ranks[0]), 'changed_rank': int(ranks[1]),
                            'detected_change': abs(delta) > NUMERIC_TOLERANCE,
                            'rank_changed': bool(ranks[0] != ranks[1])})
    finally:
        model.train(was_training)
    return results


def diagnostic_summary(delta: float, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not math.isfinite(delta):
        raise ValueError('Nonfinite NLL contrast')
    changes = sum(c['detected_change'] for c in cases)
    if delta > NUMERIC_TOLERANCE and changes:
        verdict = 'observed_on_one_development_document'
        next_question = 'prospective independent multi-document context evaluation'
    elif delta <= NUMERIC_TOLERANCE and not changes and cases:
        verdict = 'useful_multi_token_context_not_observed'
        next_question = 'isolate corpus/exposure from capacity in a separate Trial'
    else:
        verdict = 'mechanism_not_distinguished'
        next_question = 'revisit diagnostic controls before expanding training'
    return {'delta_context_nll_nats': delta, 'numeric_tolerance': NUMERIC_TOLERANCE,
            'prefix_cases': len(cases), 'detected_prefix_changes': changes,
            'rank_changes': sum(c['rank_changed'] for c in cases),
            'verdict': verdict, 'next_question': next_question,
            'evidence_scope': 'development_diagnostic', 'capability_pass': False}
