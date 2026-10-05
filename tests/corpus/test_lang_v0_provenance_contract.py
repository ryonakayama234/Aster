import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_lang_v0_source_ledger_exactly_matches_active_manifest():
    manifest = json.loads(
        (ROOT / "configs/lang-v0-corpus.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (ROOT / "docs/corpus/source-notes-v0.json").read_text(encoding="utf-8")
    )

    active_paths = [item["path"] for item in manifest["expected_files"]]
    ledger_paths = [
        path
        for group in ledger["groups"]
        for path in group["paths"]
    ]

    assert ledger["scope"] == "lang-v0-active-only"
    assert ledger["active_manifest"] == "configs/lang-v0-corpus.json"
    assert len(ledger_paths) == len(set(ledger_paths))
    assert set(ledger_paths) == set(active_paths)
    assert len(ledger_paths) == manifest["expected_file_count"]

    for group in ledger["groups"]:
        assert group["paths"]
        assert group["group"]
        assert "split" not in group


def test_lang_v0_source_ledger_contains_no_private_drive_locator():
    ledger_text = (
        ROOT / "docs/corpus/source-notes-v0.json"
    ).read_text(encoding="utf-8")

    assert "drive.google.com" not in ledger_text
    assert '"drive_id"' not in ledger_text
