import json
from pathlib import Path

import pytest

from aster.corpus.pipeline import digest, json_bytes
from aster.training.extract import Deferred, extract, rst_extract
from aster.training.view import build_view, split_and_dedup


def record(text, group='g', domain='prose'):
    result = {'schema_version': 'aster-canonical-0', 'kind': 'document', 'text': text,
              'text_format': 'plain', 'group_id': group, 'domain': domain,
              'provenance': {'raw_path': group + '.txt', 'raw_sha256': digest(text.encode()), 'origin': 'external_article'},
              'transform': {'adapter': 'text_document'}}
    result['id'] = 'rec_' + digest(json_bytes(result))
    return result


def setup(root, rows, **overrides):
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    report = b'{}'
    manifest = json_bytes({'records_sha256': digest(content), 'report_sha256': digest(report), 'record_count': len(rows)})
    bid = digest(manifest)
    dest = root / 'data/canonical/builds' / bid
    dest.mkdir(parents=True)
    (dest / 'records.jsonl').write_bytes(content)
    (dest / 'report.json').write_bytes(report)
    (dest / 'manifest.json').write_bytes(manifest)
    recipe = {'schema_version': 'aster-training-recipe-0', 'purpose': 'pretrain-pilot-not-corpus-v0',
              'canonical_build': bid, 'seed': 123, 'allowed_origins': ['external_article'],
              'train_byte_budgets': {'prose': 100}, 'split_overrides': {'g': 'train'}, 'defer_paths': {}}
    recipe.update(overrides)
    path = root / 'recipe.json'
    path.write_text(json.dumps(recipe))
    return path, dest


def test_rst_preserves_code_and_reports_removed_metadata():
    result = rst_extract('Title\n=====\n\n.. index:: ignored\n\n説明::\n\n    def f():\n        return "Ａ"\n\n.. method:: list.append(x)\n\n   Add x.\n')
    assert 'def f():\n    return "Ａ"' in result.text
    assert 'ignored' not in result.text
    assert 'list.append(x)' in result.text and 'Add x.' in result.text
    assert any(c['reason'] == 'rst_metadata:index' for c in result.changes if 'reason' in c)
    for segment in result.segments:
        a, b = segment['output_chars']
        assert result.text[a:b].strip()


@pytest.mark.parametrize('text', ['.. include:: /etc/passwd', '.. raw:: html\n\n   <script>x</script>', '.. mystery:: abc'])
def test_rst_external_content_and_unknown_directives_are_deferred(text):
    with pytest.raises(Deferred):
        rst_extract(text)


def test_conversation_preserves_explanatory_code_without_claiming_tool_execution():
    r = {'kind': 'conversation', 'messages': [{'role': 'user', 'content': 'どう書く？'},
        {'role': 'assistant', 'content': '```python\nprint(1)\n```\n', 'source_lines': [3, 5]}]}
    result = extract(r)
    assert 'Assistant:\n```python\nprint(1)\n```' in result.text
    assert len(result.segments) == 2
    r['messages'][1]['content'] = 'Browsing web\nresult'
    with pytest.raises(Deferred):
        extract(r)


def test_cross_group_duplicates_cannot_leak_or_override_conflicting_splits():
    items = [{'record': record(t, g), 'extracted': extract(record(t, g))} for t, g in [('a b', 'g'), ('ab', 'other')]]
    recipe = {'seed': 1, 'split_overrides': {'g': 'train'}}
    duplicates = split_and_dedup(items, recipe)
    assert len(duplicates) == 0  # Whitespace can change code meaning: group, do not delete.
    assert {i['split'] for i in items} == {'train'}
    recipe['split_overrides']['other'] = 'test'
    with pytest.raises(ValueError, match='conflicting'):
        split_and_dedup(items, recipe)


def test_gsm_official_train_never_becomes_test():
    r = record('Q', domain='math')
    r['transform']['adapter'] = 'gsm8k'
    item = {'record': r, 'extracted': extract(r)}
    with pytest.raises(ValueError, match='official train'):
        split_and_dedup([item], {'seed': 1, 'split_overrides': {'g': 'test'}})


def test_view_reproduces_and_keeps_run_identity_separate(tmp_path):
    recipe, canonical = setup(tmp_path, [record('learn'), record('test text', 'test')],
                              split_overrides={'g': 'train', 'test': 'test'})
    view, run1 = build_view(tmp_path, recipe)
    again, run2 = build_view(tmp_path, recipe)
    assert view == again and run1 != run2
    samples = [json.loads(s) for s in (view / 'samples.jsonl').read_text().split('\n') if s]
    assert {s['split'] for s in samples} == {'train', 'test'}
    assert all(not s['response_exemplar'] for s in samples)
    events = [json.loads(s) for s in (run1 / 'events.jsonl').read_text().split('\n') if s]
    assert [e['seq'] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]['kind'] == 'completed'
    assert json.loads((run1 / 'observation-bundle.json').read_text())['mode'] == 'saved_replay'
    (view / samples[0]['text_path']).write_text('tampered')
    with pytest.raises(ValueError, match='refusing overwrite'):
        build_view(tmp_path, recipe)
    assert any(json.loads(p.read_text())['status'] == 'failed' for p in (tmp_path / 'runs').glob('*/run.json'))


def test_budget_preserves_whole_records_and_reports_exclusions(tmp_path):
    recipe, _ = setup(tmp_path, [record('small'), record('x' * 80, 'large')],
                      train_byte_budgets={'prose': 6}, split_overrides={'g': 'train', 'large': 'train'})
    view, _ = build_view(tmp_path, recipe)
    summary = json.loads((view / 'summary.json').read_text())
    assert summary['train_bytes']['prose'] == 5
    assert summary['status_counts']['over_budget'] == 1


def test_changed_canonical_fails_with_a_failure_record(tmp_path):
    recipe, canonical = setup(tmp_path, [record('learn')])
    (canonical / 'records.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        build_view(tmp_path, recipe)
    runs = list((tmp_path / 'runs').glob('*/run.json'))
    assert len(runs) == 1 and json.loads(runs[0].read_text())['status'] == 'failed'
    assert not (tmp_path / 'data/training').exists()


def test_exact_duplicate_is_removed_after_groups_join():
    items = [{'record': record('same', g), 'extracted': extract(record('same', g))} for g in ['a', 'b']]
    duplicates = split_and_dedup(items, {'seed': 1, 'split_overrides': {'a': 'train'}})
    assert len(duplicates) == 1 and {i['split'] for i in items} == {'train'}


def test_tokenizer_learns_only_train_and_roundtrips_holdout(tmp_path):
    from aster.training.tokenizer_run import evaluate, verify_view
    recipe, _ = setup(tmp_path, [record('aaaaaaaa'), record('bbbbbbbbbbbbbbbb', 'held')],
                      split_overrides={'g': 'train', 'held': 'dev'})
    view, _ = build_view(tmp_path, recipe)
    artifact, run = evaluate(tmp_path, view, vocab_size=257)
    vocab = json.loads((artifact / 'vocab.json').read_text())
    assert vocab['256'] == b'aa'.hex()
    result = json.loads((artifact / 'evaluation.json').read_text())
    assert result['roundtrip_passed'] == 2
    assert {m['split'] for m in result['metrics']} == {'train', 'dev'}
    (view / 'train/injected.txt').write_text('unexpected')
    with pytest.raises(ValueError, match='Unexpected'):
        verify_view(view)
