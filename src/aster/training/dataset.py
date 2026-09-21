"""固定viewの本文を、文書をまたがない次token予測課題へ変換する。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch
from aster.corpus.pipeline import digest, json_bytes
from aster.tokenizer.artifact import AsterTokenizer, TokenizerManifest, load_tokenizer
from aster.tokenizer.bpe import BPEModel
from aster.training.tokenizer_run import verify_view

IGNORE_INDEX = -100


def tokenizer_payload(tokenizer):
    return {'manifest': asdict(tokenizer.manifest),
            'vocab': {str(i): b.hex() for i, b in tokenizer.model.vocab.items()},
            'merges': [{'left': p[0], 'right': p[1], 'new_id': i}
                       for p, i in tokenizer.model.merges],
            'special_tokens': dict(tokenizer.special_tokens)}


def tokenizer_from_payload(payload):
    vocab = {int(k): bytes.fromhex(v) for k, v in payload['vocab'].items()}
    if any(vocab.get(i) != bytes([i]) for i in range(256)):
        raise ValueError('Invalid byte vocabulary')
    merges = []
    known = set(range(256))
    for i, m in enumerate(payload['merges'], 256):
        left, right, new_id = m['left'], m['right'], m['new_id']
        if new_id != i or left not in known or right not in known:
            raise ValueError('Invalid merge IDs')
        if vocab.get(i) != vocab[left] + vocab[right]:
            raise ValueError('Merge bytes disagree with vocabulary')
        known.add(i)
        merges.append(((left, right), i))
    if set(vocab) != known:
        raise ValueError('Unreferenced vocabulary entries')
    special = payload['special_tokens']
    if special != {'<|bos|>': len(vocab), '<|eos|>': len(vocab) + 1}:
        raise ValueError('Expected contiguous BOS/EOS IDs')
    manifest = TokenizerManifest(**payload['manifest'])
    if manifest.actual_bpe_vocab_size != len(vocab) or manifest.special_token_count != 2:
        raise ValueError('Tokenizer manifest mismatch')
    return AsterTokenizer(BPEModel(merges, vocab), special, manifest)


def load_training_tokenizer(path, view_id):
    path = Path(path)
    # Require the existing evaluator's provenance, not a similarly named tokenizer.
    evaluation = json.loads((path / 'evaluation.json').read_text(encoding='utf-8'))
    if evaluation['experiment']['view_id'] != view_id:
        raise ValueError('Tokenizer and training view do not match')
    payload = tokenizer_payload(load_tokenizer(path))
    return tokenizer_from_payload(payload), payload


@dataclass(frozen=True)
class Window:
    ids: tuple[int, ...]  # x=ids[:-1], y=ids[1:]
    record_id: str
    domain: str
    offset: int


def prepare_windows(view, tokenizer, context_length):
    if type(context_length) is not int or context_length < 1:
        raise ValueError('context_length must be positive')
    view = Path(view).resolve()
    samples = sorted(verify_view(view), key=lambda s: s['record_id'])
    splits = {'train': [], 'dev': []}
    for sample in samples:
        if sample['split'] not in splits:
            continue  # Never use test for iterative model selection or previews.
        text = (view / sample['text_path']).read_text(encoding='utf-8')
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        for offset in range(0, len(ids) - 1, context_length):
            splits[sample['split']].append(Window(tuple(ids[offset:offset + context_length + 1]),
                                                   sample['record_id'], sample['domain'], offset))
    if not splits['train']:
        raise ValueError('No training windows')
    return splits


def collate(windows, eos_id, device='cpu'):
    if not windows:
        raise ValueError('Cannot batch zero windows')
    length = max(len(w.ids) - 1 for w in windows)
    x = torch.full((len(windows), length), eos_id, dtype=torch.long, device=device)
    y = torch.full_like(x, IGNORE_INDEX)
    for i, window in enumerate(windows):
        size = len(window.ids) - 1
        if size < 1:
            raise ValueError('Window needs at least one prediction target')
        x[i, :size] = torch.tensor(window.ids[:-1], device=device)
        y[i, :size] = torch.tensor(window.ids[1:], device=device)
    # Right padding cannot affect earlier valid positions under causal attention.
    return x, y
