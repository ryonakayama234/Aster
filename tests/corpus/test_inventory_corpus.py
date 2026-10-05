import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inventory_corpus.py"
SPEC = importlib.util.spec_from_file_location("inventory_corpus_script", SCRIPT)
inventory_corpus = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inventory_corpus)


def fixture_root(tmp_path, *, active_files, pool_files, ledger_paths):
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs/corpus").mkdir(parents=True)
    (tmp_path / "data/raw/pool").mkdir(parents=True)

    notes = {
        "schema_version": 1,
        "default_usage_status": "proposal_pending_user_preference",
        "groups": [{
            "id": "test-group",
            "paths": ledger_paths,
            "origin": "external_article",
            "url": None,
            "verification": "test",
            "group": "test-group",
        }],
    }
    (tmp_path / "docs/corpus/source-notes-v0.json").write_text(
        json.dumps(notes), encoding="utf-8"
    )
    (tmp_path / "docs/corpus/external-seed-manifest.json").write_text(
        json.dumps({"sources": []}), encoding="utf-8"
    )
    (tmp_path / "configs/lang-v0-corpus.json").write_text(
        json.dumps({
            "schema_version": 1,
            "experiment": "lang-v0",
            "expected_files": active_files,
        }),
        encoding="utf-8",
    )

    for rel, data in pool_files.items():
        path = tmp_path / "data/raw/pool" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path


def configure(monkeypatch, root):
    monkeypatch.setattr(inventory_corpus, "ROOT", root)
    monkeypatch.setattr(inventory_corpus, "POOL", root / "data/raw/pool")
    monkeypatch.setattr(inventory_corpus, "DOCS", root / "docs/corpus")
    monkeypatch.setattr(
        inventory_corpus, "ACTIVE_MANIFEST", root / "configs/lang-v0-corpus.json"
    )


def test_ledger_history_may_be_absent_when_active_manifest_is_complete(tmp_path, monkeypatch):
    root = fixture_root(
        tmp_path,
        active_files=[{"path": "prose/ja/001.txt", "bytes": 5}],
        pool_files={"prose/ja/001.txt": b"hello"},
        ledger_paths=["prose/ja/001.txt", "prose/ja/002.txt"],
    )
    configure(monkeypatch, root)

    inventory = inventory_corpus.build_inventory()

    assert inventory["ledger_not_present"] == ["prose/ja/002.txt"]
    assert inventory["active_manifest"]["present_files"] == 1
    assert inventory["files"][0]["active_for_lang_v0"] is True


def test_active_raw_requires_source_mapping(tmp_path, monkeypatch):
    root = fixture_root(
        tmp_path,
        active_files=[{"path": "prose/ja/003.txt", "bytes": 1}],
        pool_files={"prose/ja/003.txt": b"x"},
        ledger_paths=["prose/ja/001.txt"],
    )
    configure(monkeypatch, root)

    with pytest.raises(ValueError, match="missing source mapping"):
        inventory_corpus.build_inventory()


def test_missing_active_raw_file_is_fatal(tmp_path, monkeypatch):
    root = fixture_root(
        tmp_path,
        active_files=[
            {"path": "prose/ja/001.txt", "bytes": 5},
            {"path": "prose/ja/003.txt", "bytes": 1},
        ],
        pool_files={"prose/ja/001.txt": b"hello"},
        ledger_paths=["prose/ja/001.txt"],
    )
    configure(monkeypatch, root)

    with pytest.raises(ValueError, match="Active manifest missing raw file"):
        inventory_corpus.build_inventory()


def test_active_manifest_byte_mismatch_is_fatal(tmp_path, monkeypatch):
    root = fixture_root(
        tmp_path,
        active_files=[{"path": "prose/ja/001.txt", "bytes": 6}],
        pool_files={"prose/ja/001.txt": b"hello"},
        ledger_paths=["prose/ja/001.txt"],
    )
    configure(monkeypatch, root)

    with pytest.raises(ValueError, match="Active manifest byte mismatch"):
        inventory_corpus.build_inventory()


def test_exact_duplicate_audit_is_preserved(tmp_path, monkeypatch):
    root = fixture_root(
        tmp_path,
        active_files=[
            {"path": "prose/ja/001.txt", "bytes": 5},
            {"path": "prose/ja/002.txt", "bytes": 5},
        ],
        pool_files={
            "prose/ja/001.txt": b"hello",
            "prose/ja/002.txt": b"hello",
        },
        ledger_paths=["prose/ja/001.txt", "prose/ja/002.txt"],
    )
    configure(monkeypatch, root)

    inventory = inventory_corpus.build_inventory()

    assert inventory["exact_lf_text_duplicates"] == [
        ["prose/ja/001.txt", "prose/ja/002.txt"]
    ]
