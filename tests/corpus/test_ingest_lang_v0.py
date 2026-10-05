import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest_lang_v0.py"
SPEC = importlib.util.spec_from_file_location("ingest_lang_v0_script", SCRIPT)
ingest_lang_v0 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ingest_lang_v0)


def fixture_root(tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "experiment": "lang-v0",
        "expected_files": [
            {"path": "dialogue/ja/003.txt", "bytes": 3},
            {"path": "prose/ja/001.txt", "bytes": 5},
        ],
        "expected_file_count": 2,
        "expected_total_bytes": 8,
    }
    (root / "configs/lang-v0-corpus.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def source_dir(tmp_path, *, prose=b"hello", dialogue=b"abc"):
    source = tmp_path / "download"
    (source / "nested").mkdir(parents=True)
    (source / "prose001.txt").write_bytes(prose)
    (source / "nested/dialogue003.txt").write_bytes(dialogue)
    return source


def test_ingest_maps_flat_or_nested_drive_names_and_is_idempotent(tmp_path):
    root = fixture_root(tmp_path)
    source = source_dir(tmp_path)

    first = ingest_lang_v0.ingest(source, root=root, run_inventory=False)
    second = ingest_lang_v0.ingest(source, root=root, run_inventory=False)

    assert first == {
        "files": 2,
        "bytes": 8,
        "copied": 2,
        "unchanged": 0,
        "dry_run": False,
        "inventory_ran": False,
    }
    assert second["copied"] == 0
    assert second["unchanged"] == 2
    assert (root / "data/raw/pool/prose/ja/001.txt").read_bytes() == b"hello"
    assert (root / "data/raw/pool/dialogue/ja/003.txt").read_bytes() == b"abc"


def test_missing_file_fails_before_any_write(tmp_path):
    root = fixture_root(tmp_path)
    source = tmp_path / "download"
    source.mkdir()
    (source / "prose001.txt").write_bytes(b"hello")

    with pytest.raises(ValueError, match="Missing LANG v0 source file"):
        ingest_lang_v0.ingest(source, root=root, run_inventory=False)

    assert not (root / "data/raw/pool").exists()


def test_byte_mismatch_fails_before_any_write(tmp_path):
    root = fixture_root(tmp_path)
    source = source_dir(tmp_path, dialogue=b"wrong")

    with pytest.raises(ValueError, match="Source byte mismatch"):
        ingest_lang_v0.ingest(source, root=root, run_inventory=False)

    assert not (root / "data/raw/pool").exists()


def test_unexpected_matching_file_is_fatal(tmp_path):
    root = fixture_root(tmp_path)
    source = source_dir(tmp_path)
    (source / "prose999.txt").write_bytes(b"x")

    with pytest.raises(ValueError, match="Unexpected LANG v0 source file"):
        ingest_lang_v0.ingest(source, root=root, run_inventory=False)


def test_existing_different_destination_is_never_overwritten(tmp_path):
    root = fixture_root(tmp_path)
    source = source_dir(tmp_path)
    destination = root / "data/raw/pool/prose/ja/001.txt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"xxxxx")

    with pytest.raises(ValueError, match="different bytes"):
        ingest_lang_v0.ingest(source, root=root, run_inventory=False)

    assert destination.read_bytes() == b"xxxxx"


def test_dry_run_writes_nothing(tmp_path):
    root = fixture_root(tmp_path)
    source = source_dir(tmp_path)

    result = ingest_lang_v0.ingest(
        source, root=root, run_inventory=True, dry_run=True
    )

    assert result["files"] == 2
    assert result["bytes"] == 8
    assert result["inventory_ran"] is False
    assert not (root / "data/raw/pool").exists()
