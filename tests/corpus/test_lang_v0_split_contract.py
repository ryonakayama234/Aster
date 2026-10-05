import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_lang_v0_split_covers_every_provenance_group_once():
    ledger = load("docs/corpus/source-notes-v0.json")
    split = load("configs/lang-v0-split.json")

    ledger_groups = {group["group"]: group for group in ledger["groups"]}
    assignments = split["groups"]
    assigned_ids = [item["group_id"] for item in assignments]

    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids) == set(ledger_groups)
    assert {item["disposition"] for item in assignments} <= {
        "train", "dev", "test", "pending_excluded"
    }


def test_lang_v0_split_bytes_and_files_match_active_manifest():
    ledger = load("docs/corpus/source-notes-v0.json")
    active = load("configs/lang-v0-corpus.json")
    split = load("configs/lang-v0-split.json")

    bytes_by_path = {item["path"]: item["bytes"] for item in active["expected_files"]}
    groups = {group["group"]: group for group in ledger["groups"]}

    files_by_disposition = Counter()
    bytes_by_disposition = Counter()

    for assignment in split["groups"]:
        paths = groups[assignment["group_id"]]["paths"]
        observed_bytes = sum(bytes_by_path[path] for path in paths)
        assert assignment["files"] == len(paths)
        assert assignment["bytes"] == observed_bytes
        files_by_disposition[assignment["disposition"]] += len(paths)
        bytes_by_disposition[assignment["disposition"]] += observed_bytes

    summary = split["summary"]
    assert sum(files_by_disposition.values()) == summary["all_active_files"]
    assert sum(bytes_by_disposition.values()) == summary["all_active_bytes"]
    assert files_by_disposition["pending_excluded"] == summary["pending_excluded_files"]
    assert bytes_by_disposition["pending_excluded"] == summary["pending_excluded_bytes"]
    assert bytes_by_disposition["train"] == summary["train_bytes"]
    assert bytes_by_disposition["dev"] == summary["dev_bytes"]
    assert bytes_by_disposition["test"] == summary["test_bytes"]


def test_lang_v0_preserves_preexisting_overrides_for_eligible_groups():
    split = load("configs/lang-v0-split.json")
    prior = load("configs/training-pilot-v0.json")

    current = {
        item["group_id"]: item["disposition"]
        for item in split["groups"]
        if item["disposition"] in {"train", "dev", "test"}
    }
    prior_active = {
        group: disposition
        for group, disposition in prior["split_overrides"].items()
        if group in current
    }

    assert current == prior_active
    assert current == {
        "article-calculemus": "train",
        "article-bsd": "dev",
        "article-lean-trust": "train",
        "article-tex-lean": "train",
        "article-adverbs": "test",
        "unresolved-chatgpt-sessions": "train",
    }


def test_lang_v0_pending_groups_are_not_admitted_by_canonical_recipe():
    ledger = load("docs/corpus/source-notes-v0.json")
    split = load("configs/lang-v0-split.json")
    canonical = load("configs/canonical-v0.json")

    groups = {group["group"]: group for group in ledger["groups"]}
    pending = {
        item["group_id"]
        for item in split["groups"]
        if item["disposition"] == "pending_excluded"
    }

    for group_id in pending:
        origin = groups[group_id]["origin"]
        assert origin not in canonical["adapters_by_origin"]


def test_lang_v0_test_is_nonempty_and_sealed_by_policy():
    split = load("configs/lang-v0-split.json")
    assert any(item["disposition"] == "test" for item in split["groups"])
    assert any("sealed" in rule.lower() for rule in split["policy"])
