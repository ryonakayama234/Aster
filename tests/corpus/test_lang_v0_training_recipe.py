import json
import re
from pathlib import Path

import pytest

from aster.training.extract import Extracted
from aster.training.view import split_and_dedup


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_lang_v0_training_recipe_is_bound_to_frozen_split_and_canonical():
    recipe = load("configs/lang-v0-training.json")
    split = load("configs/lang-v0-split.json")
    ledger = load("docs/corpus/source-notes-v0.json")
    active = load("configs/lang-v0-corpus.json")

    assert recipe["purpose"] == "lang-v0-pretrain-pilot"
    assert recipe["require_explicit_split_groups"] is True
    assert recipe["split_contract"] == "configs/lang-v0-split.json"
    assert re.fullmatch(r"[0-9a-f]{64}", recipe["canonical_build"])

    eligible = {
        item["group_id"]: item["disposition"]
        for item in split["groups"]
        if item["disposition"] in {"train", "dev", "test"}
    }
    assert recipe["split_overrides"] == eligible

    paths_by_group = {
        group["group"]: group["paths"]
        for group in ledger["groups"]
    }
    bytes_by_path = {
        item["path"]: item["bytes"]
        for item in active["expected_files"]
    }
    raw_train_bytes = {"prose": 0, "dialogue": 0}
    for group_id, disposition in eligible.items():
        if disposition != "train":
            continue
        for path in paths_by_group[group_id]:
            domain = path.split("/", 1)[0]
            raw_train_bytes[domain] += bytes_by_path[path]

    assert recipe["train_byte_budgets"] == raw_train_bytes
    assert set(recipe["defer_paths"]) <= {
        "data/raw/pool/" + path
        for group_id, disposition in eligible.items()
        if disposition == "train"
        for path in paths_by_group[group_id]
    }


def test_lang_v0_canonical_build_is_the_observed_local_build():
    recipe = load("configs/lang-v0-training.json")
    assert recipe["canonical_build"] == (
        "239fff3a6ac9c4d796954ee23c81fae15ffff3e205a9655e7e184fea2c454f587"
    )


def test_explicit_split_mode_rejects_hash_fallback():
    candidates = [{
        "record": {
            "id": "rec_test",
            "group_id": "group-without-override",
            "transform": {"adapter": "text_document"},
        },
        "extracted": Extracted("hello", [], []),
    }]
    recipe = {
        "seed": 1,
        "split_overrides": {},
        "require_explicit_split_groups": True,
    }

    with pytest.raises(ValueError, match="Missing explicit split override"):
        split_and_dedup(candidates, recipe)
