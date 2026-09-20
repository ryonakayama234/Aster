"""Train BPE on a verified train view and measure held-out text separately."""

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.tokenizer.artifact import train_aster_tokenizer, save_tokenizer
from aster.training.view import read_jsonl


def verify_view(view):
    manifest_bytes = (view / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != view.name:
        raise ValueError('Training manifest identity mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest['schema_version'] != 'aster-training-view-0':
        raise ValueError('Unsupported training view')
    actual = {p.relative_to(view).as_posix() for p in view.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Unexpected or missing view files')
    for name, expected in manifest['files'].items():
        path = view / name
        if path.is_symlink() or not path.resolve().is_relative_to(view.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError('Training file hash/path mismatch: ' + name)
    samples = read_jsonl((view / 'samples.jsonl').read_bytes())
    ids, groups, texts = set(), {}, {}
    paths = set()
    for s in samples:
        if s['split'] not in {'train', 'dev', 'test'} or s['text_path'] != f"{s['split']}/{s['record_id']}.txt":
            raise ValueError('Sample split/path mismatch')
        if s['record_id'] in ids or s['text_path'] not in manifest['files']:
            raise ValueError('Duplicate sample or untracked text')
        ids.add(s['record_id'])
        paths.add(s['text_path'])
        body = (view / s['text_path']).read_bytes()
        if digest(body) != s['text_sha256']:
            raise ValueError('Sample text hash mismatch')
        for mapping, key in [(groups, s['leakage_group']), (texts, s['text_sha256'])]:
            if key in mapping and mapping[key] != s['split']:
                raise ValueError('Cross-split leakage')
            mapping[key] = s['split']
    if paths != {n for n in actual if n.endswith('.txt')}:
        raise ValueError('Unindexed training text')
    return samples


def evaluate(root, view, vocab_size=512):
    root, view = root.resolve(), view.resolve()
    run = RunLog(root, 'tokenizer', {'view_id': view.name, 'target_vocab_size': vocab_size})
    try:
        samples = verify_view(view)
        samples.sort(key=lambda s: s['record_id'])
        train = [s for s in samples if s['split'] == 'train']
        if not train:
            raise ValueError('No train samples')
        started = time.perf_counter()
        tokenizer = train_aster_tokenizer([(view / s['text_path']).read_text(encoding='utf-8') for s in train],
                                          target_vocab_size=vocab_size, min_pair_frequency=2)
        run.event('bpe_trained', {'train_samples': len(train), 'seconds': time.perf_counter() - started,
                                 'bpe_vocab': len(tokenizer.model.vocab)})
        buckets, examples = defaultdict(list), []
        shown = set()
        for index, sample in enumerate(samples):
            text = (view / sample['text_path']).read_text(encoding='utf-8')
            ids = tokenizer.encode(text)
            if tokenizer.decode(ids) != text:
                raise ValueError('Roundtrip failed: ' + sample['record_id'])
            buckets[(sample['split'], sample['domain'])].append((len(text.encode()), len(text), len(ids)))
            key = (sample['split'], sample['domain'])
            if key not in shown:
                shown.add(key)
                prefix = text[:120]
                token_ids = tokenizer.encode(prefix)
                examples.append({'sample_id': sample['record_id'], 'split': sample['split'], 'domain': sample['domain'],
                                 'text': prefix, 'utf8_hex': prefix.encode().hex(), 'token_ids': token_ids,
                                 'token_hex': [tokenizer.model.vocab[t].hex() for t in token_ids]})
            if (index + 1) % 100 == 0:
                run.event('evaluation_progress', {'completed': index + 1, 'total': len(samples)})
        metrics = []
        for (split, domain), values in sorted(buckets.items()):
            byte_count, chars, tokens = map(sum, zip(*values))
            lengths = sorted(v[2] for v in values)
            metrics.append({'split': split, 'domain': domain, 'samples': len(values),
                            'bytes': byte_count, 'characters': chars, 'bpe_tokens': tokens,
                            'byte_baseline_tokens': byte_count, 'bytes_per_token': byte_count / tokens,
                            'tokens_per_character': tokens / chars,
                            'median_document_tokens': lengths[len(lengths) // 2],
                            'p95_document_tokens': lengths[max(0, (95 * len(lengths) + 99) // 100 - 1)]})
        code_dir = Path(__file__).parents[1] / 'tokenizer'
        spec = {'schema_version': 'aster-tokenizer-experiment-0', 'view_id': view.name,
                'target_vocab_size': vocab_size, 'min_pair_frequency': 2,
                'code_sha256': {p.name: digest(p.read_bytes()) for p in [Path(__file__), code_dir / 'artifact.py', code_dir / 'bpe.py']}}
        dest = root / 'artifacts/tokenizers' / digest(json_bytes(spec))
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = {'experiment': spec, 'roundtrip_samples': len(samples), 'roundtrip_passed': len(samples),
                  'metrics': metrics, 'examples': examples,
                  'note': 'Token counts exclude BOS/EOS. Byte baseline is one token per UTF-8 byte. No TinyLM trained.'}
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.building-') as temp:
            stage = Path(temp) / 'tokenizer'
            save_tokenizer(tokenizer, stage)
            (stage / 'evaluation.json').write_bytes(json_bytes(result))
            if dest.exists():
                if {p.name for p in dest.iterdir()} != {p.name for p in stage.iterdir()} or any((dest / p.name).read_bytes() != p.read_bytes() for p in stage.iterdir()):
                    raise ValueError('Existing tokenizer experiment differs')
            else:
                os.rename(stage, dest)
        bundle = {'schema_version': 'aster-tokenizer-bundle-0', 'mode': 'saved_replay', 'evaluation': result,
                  'vocab': json.loads((dest / 'vocab.json').read_text()),
                  'merges': json.loads((dest / 'merges.json').read_text()),
                  'special_tokens': tokenizer.special_tokens}
        (run.path / 'tokenizer-bundle.json').write_bytes(json_bytes(bundle))
        run.finish('completed', output=str(dest), roundtrip_samples=len(samples))
        return dest, run.path
    except BaseException as error:
        run.finish('failed', error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--view', required=True, type=Path)
    parser.add_argument('--vocab-size', type=int, default=512)
    args = parser.parse_args()
    output, run = evaluate(args.root, args.view, args.vocab_size)
    print(json.dumps({'tokenizer': str(output), 'run': str(run)}))


if __name__ == '__main__':
    main()
