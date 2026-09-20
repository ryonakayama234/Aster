import json
from pathlib import Path

import pytest

from aster.corpus.pipeline import adapt, build, digest, normalize, parse_conversation


def test_normalization_preserves_code_and_unicode():
    raw = '\ufeffdef f():\r\n\treturn "Ａ　e\u0301"\r\n'.encode()
    assert normalize(raw) == 'def f():\n\treturn "Ａ　e\u0301"\n'


def test_role_headers_inside_code_are_not_roles():
    text = '# 質問\nコードを説明して\n# 解答\n```python\n# 質問\nx = 1\n```\n説明\n# 質問\n続き\n# 回答\nはい\n'
    messages = parse_conversation(text)
    assert [m['role'] for m in messages] == ['user', 'assistant', 'user', 'assistant']
    assert '# 質問\nx = 1' in messages[1]['content']
    assert messages[0]['source_lines'] == [2, 2]


@pytest.mark.parametrize('text', [
    '# 解答\nはい', '# 質問\n問いだけ', '# 質問\n\n# 解答\nはい',
    '不明な前置き\n# 質問\n何\n# 解答\nはい',
    '# 質問\n何\n# 質問\n別の問い\n# 解答\nはい',
    '# 質問\n何\n# 解答\n```\n未閉鎖',
])
def test_ambiguous_conversations_are_deferred(text):
    with pytest.raises(ValueError):
        parse_conversation(text)


def test_gsm_keeps_original_problem_identity_across_line_positions():
    item = json.dumps({'question': '2+3?', 'answer': '2+3=<<2+3=5>>5\n#### 5'})
    first = adapt('gsm8k', item)[0]
    second = adapt('gsm8k', '\n' + item)[0]
    assert first[2] == second[2]
    assert '<<' not in first[1]['text']
    assert '#### 5' in first[1]['text']
    with pytest.raises(ValueError):
        adapt('gsm8k', json.dumps({'question': '2+3?', 'answer': '5'}))


def test_unicode_line_separator_inside_json_string_is_not_a_record_boundary():
    item = json.dumps({'question': 'a\u2028b?', 'answer': '#### 1'}, ensure_ascii=False)
    rows = adapt('gsm8k', item + '\n')
    assert len(rows) == 1
    assert 'a\u2028b?' in rows[0][1]['text']


def fixture_root(tmp_path):
    for name in ['configs', 'docs/corpus', 'data/raw/pool']:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / 'configs/canonical-v0.json').write_text(json.dumps({
        'schema_version': 'aster-canonical-0',
        'adapters_by_origin': {'external_article': 'text_document'},
        'candidate_statuses': ['candidate_needs_review']}))
    notes = b'{}'
    (tmp_path / 'docs/corpus/source-notes-v0.json').write_bytes(notes)
    raw = b'hello\r\n'
    (tmp_path / 'data/raw/pool/a.txt').write_bytes(raw)
    inventory = {'source_notes_sha256': digest(notes), 'files': [{
        'path': 'a.txt', 'sha256': digest(raw), 'origin': 'external_article',
        'status': 'candidate_needs_review', 'group_id': 'original-a',
        'source_verification': 'test', 'license_status': 'unknown', 'flags': []}]}
    (tmp_path / 'docs/corpus/pool-inventory-v0.json').write_text(json.dumps(inventory))
    return tmp_path


def test_build_reproducible_preserves_raw_and_catches_tampering(tmp_path):
    root = fixture_root(tmp_path)
    result = build(root)
    assert build(root) == result
    record = json.loads((result / 'records.jsonl').read_text())
    assert record['text'] == 'hello\n'
    assert record['group_id'] == 'original-a'
    assert record['split'] is None
    assert record['review']['training_status'] == 'not_selected'
    assert (root / 'data/raw/pool/a.txt').read_bytes() == b'hello\r\n'
    (result / 'records.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        build(root)


@pytest.mark.parametrize('mutation', ['raw', 'notes', 'new_file'])
def test_stale_inventory_cannot_publish(tmp_path, mutation):
    root = fixture_root(tmp_path)
    if mutation == 'raw':
        (root / 'data/raw/pool/a.txt').write_text('changed')
    elif mutation == 'notes':
        (root / 'docs/corpus/source-notes-v0.json').write_text('{"updated": true}')
    else:
        (root / 'data/raw/pool/new.txt').write_text('new')
    with pytest.raises(ValueError, match='regenerate inventory'):
        build(root)
    assert not (root / 'data/canonical/builds').exists()
