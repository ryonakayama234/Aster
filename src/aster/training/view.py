"""Build a deterministic pretraining pilot plus an observable execution record."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import docutils

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.training.extract import Deferred, extract


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()


def read_jsonl(data):
    return [json.loads(line) for line in data.decode().split('\n') if line.strip()]


def load_canonical(root, build_id):
    if not isinstance(build_id, str) or not re.fullmatch('[0-9a-f]{64}', build_id):
        raise ValueError('A full canonical build ID is required')
    path = root / 'data/canonical/builds' / build_id
    manifest_bytes = (path / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != build_id:
        raise ValueError('Canonical manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    content = (path / 'records.jsonl').read_bytes()
    if digest(content) != manifest['records_sha256'] or digest((path / 'report.json').read_bytes()) != manifest['report_sha256']:
        raise ValueError('Canonical content hash mismatch')
    records = read_jsonl(content)
    if len(records) != manifest['record_count'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Canonical record count or identity mismatch')
    for r in records:
        body = {k: v for k, v in r.items() if k != 'id'}
        if r['schema_version'] != 'aster-canonical-0' or r['id'] != 'rec_' + digest(json_bytes(body)):
            raise ValueError('Canonical record hash/schema mismatch')
    return records


def split_and_dedup(candidates, recipe):
    """Join original groups and whitespace-equivalent texts before assigning splits."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = sorted([find(a), find(b)])
        parent[b] = a

    first = {}
    for item in candidates:
        group = item['record']['group_id']
        find(group)
        signature = digest(re.sub(r'\s+', '', item['extracted'].text).encode())
        item['duplicate_signature'] = signature
        item['exact_signature'] = digest(item['extracted'].text.encode())
        if signature in first:
            union(group, first[signature])
        first[signature] = group
    clusters = defaultdict(list)
    for item in candidates:
        clusters[find(item['record']['group_id'])].append(item)
    duplicates = []
    for cluster, members in sorted(clusters.items()):
        groups = sorted({m['record']['group_id'] for m in members})
        overrides = {recipe['split_overrides'][g] for g in groups if g in recipe['split_overrides']}
        if len(overrides) > 1:
            raise ValueError('Duplicate/source cluster has conflicting split overrides: ' + ', '.join(groups))
        official_train = any(m['record']['transform']['adapter'] == 'gsm8k' for m in members)
        if overrides:
            split = next(iter(overrides))
        else:
            bucket = int(digest(f"{recipe['seed']}:{cluster}".encode())[:8], 16) % 100
            split = ('dev' if bucket < 10 else 'train') if official_train else ('test' if bucket < 10 else 'dev' if bucket < 20 else 'train')
        if split not in {'train', 'dev', 'test'} or (official_train and split == 'test'):
            raise ValueError('Invalid split; GSM8K official train cannot become independent test')
        seen = {}
        for item in sorted(members, key=lambda x: x['record']['id']):
            item.update(split=split, leakage_group=cluster)
            signature = item['exact_signature']
            item['duplicate_of'] = seen.get(signature)
            if item['duplicate_of']:
                duplicates.append({'record_id': item['record']['id'], 'duplicate_of': item['duplicate_of']})
            else:
                seen[signature] = item['record']['id']
    return duplicates


def publish(root, payloads, manifest):
    manifest['files'] = {name: digest(content) for name, content in sorted(payloads.items())}
    payloads['manifest.json'] = json_bytes(manifest)
    view_id = digest(payloads['manifest.json'])
    dest = root / 'data/training' / view_id
    if dest.exists():
        actual = {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
        if actual != set(payloads) or any((dest / name).read_bytes() != data for name, data in payloads.items()):
            raise ValueError('Existing training view was modified; refusing overwrite')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.building-', dir=dest.parent) as temp:
        stage = Path(temp) / 'view'
        stage.mkdir()
        for name, content in payloads.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        os.rename(stage, dest)
    return dest


def build_view(root: Path, recipe_path: Path):
    root = root.resolve()
    run = RunLog(root, 'training_view', {'recipe_path': str(recipe_path)})
    try:
        recipe_bytes = recipe_path.read_bytes()
        recipe = json.loads(recipe_bytes)
        if recipe['schema_version'] != 'aster-training-recipe-0' or recipe['purpose'] != 'pretrain-pilot-not-corpus-v0':
            raise ValueError('Only the explicit pretraining pilot recipe is supported')
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in recipe['train_byte_budgets'].values()):
            raise ValueError('Budgets must be nonnegative integers')
        records = load_canonical(root, recipe['canonical_build'])
        candidates, decisions, transforms = [], [], []
        for record in records:
            path, rid = record['provenance']['raw_path'], record['id']
            reason = recipe['defer_paths'].get(path)
            if record['provenance']['origin'] not in recipe['allowed_origins']:
                reason = 'origin not admitted by pilot recipe'
            if record['domain'] not in recipe['train_byte_budgets']:
                reason = 'domain not admitted by pilot recipe'
            try:
                if reason:
                    raise Deferred(reason)
                result = extract(record)
                if not result.text.strip():
                    raise Deferred('empty extracted text')
            except Deferred as error:
                decision = {'record_id': rid, 'status': 'deferred', 'reason': str(error), 'raw_path': path}
                decisions.append(decision)
                run.event('sample_deferred', decision)
                continue
            candidates.append({'record': record, 'extracted': result})
            transforms.append({'record_id': rid, 'raw_path': path, 'raw_sha256': record['provenance']['raw_sha256'],
                               'segments': result.segments, 'changes': result.changes,
                               'output_sha256': digest(result.text.encode())})
            run.event('sample_transformed', {'record_id': rid, 'bytes': len(result.text.encode()), 'changes': len(result.changes)})
        duplicates = split_and_dedup(candidates, recipe)
        usage = Counter()
        samples, payloads, preview = [], {}, []
        for item in sorted(candidates, key=lambda x: digest(f"{recipe['seed']}:{x['record']['id']}".encode())):
            r, text = item['record'], item['extracted'].text
            count, split = len(text.encode()), item['split']
            status, reason = 'selected', 'pilot pretraining selection'
            if item['duplicate_of']:
                status, reason = 'duplicate', item['duplicate_of']
            elif split == 'train' and usage[r['domain']] + count > recipe['train_byte_budgets'][r['domain']]:
                status, reason = 'over_budget', 'complete document exceeds remaining domain budget; not truncated'
            decision = {'record_id': r['id'], 'raw_path': r['provenance']['raw_path'],
                        'group_id': r['group_id'], 'leakage_group': item['leakage_group'],
                        'split': split, 'status': status, 'reason': reason, 'domain': r['domain'], 'bytes': count}
            decisions.append(decision)
            if status != 'selected':
                continue
            if split == 'train':
                usage[r['domain']] += count
            relative = f"{split}/{r['id']}.txt"
            payloads[relative] = text.encode()
            sample = {**decision, 'text_path': relative, 'text_sha256': digest(text.encode()),
                      'canonical_id': r['id'], 'response_exemplar': False}
            samples.append(sample)
            preview.append({'sample': sample, 'canonical': r, 'training_text': text,
                            'transform': next(t for t in transforms if t['record_id'] == r['id'])})
        if not any(s['split'] == 'train' for s in samples):
            raise ValueError('No train samples fit this recipe')
        samples.sort(key=lambda s: s['record_id'])
        decisions.sort(key=lambda d: d['record_id'])
        transforms.sort(key=lambda t: t['record_id'])
        summary = {'purpose': recipe['purpose'], 'canonical_build': recipe['canonical_build'],
                   'source_records': len(records), 'selected_records': len(samples),
                   'split_counts': dict(Counter(s['split'] for s in samples)),
                   'status_counts': dict(Counter(d['status'] for d in decisions)),
                   'train_bytes': dict(usage), 'train_byte_budgets': recipe['train_byte_budgets'],
                   'duplicate_pairs': duplicates, 'near_duplicate_scope': 'exact texts deduplicated; whitespace-only variants grouped but preserved; semantic and partial overlap not checked',
                   'limits': recipe.get('notes', [])}
        payloads.update({'samples.jsonl': jsonl(samples), 'decisions.jsonl': jsonl(decisions),
                         'transformations.jsonl': jsonl(transforms), 'summary.json': json_bytes(summary),
                         'recipe.json': recipe_bytes})
        manifest = {'schema_version': 'aster-training-view-0', 'purpose': recipe['purpose'],
                    'canonical_build': recipe['canonical_build'], 'recipe_sha256': digest(recipe_bytes),
                    'docutils_version': docutils.__version__,
                    'code_sha256': {p.name: digest(p.read_bytes()) for p in [Path(__file__), Path(__file__).with_name('extract.py')]}}
        dest = publish(root, payloads, manifest)
        run.event('view_published', {'view_id': dest.name, 'summary': summary})
        # A private, local replay bundle; do not upload corpus text with the Site.
        bundle = {'schema_version': 'aster-observation-bundle-0', 'mode': 'saved_replay',
                  'view_id': dest.name, 'manifest': json.loads((dest / 'manifest.json').read_text()),
                  'summary': summary, 'samples': sorted(preview, key=lambda x: x['sample']['record_id'])}
        (run.path / 'observation-bundle.json').write_bytes(json_bytes(bundle))
        run.finish('completed', view_id=dest.name, output=str(dest))
        return dest, run.path
    except BaseException as error:
        run.finish('failed', error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--recipe', type=Path, default=Path('configs/training-pilot-v0.json'))
    args = parser.parse_args()
    recipe = args.recipe if args.recipe.is_absolute() else args.root / args.recipe
    view, run = build_view(args.root, recipe)
    print(json.dumps({'view': str(view), 'run': str(run)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
