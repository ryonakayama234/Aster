"""Frozen confirmatory protocol invariants for LEARN v0."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
    validate_confirmatory_candidate_coverage,
    validate_confirmatory_manifest,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_committed_confirmatory_manifest_is_frozen_and_valid():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)

    assert confirmatory_manifest_sha256(manifest) == EXPECTED_MANIFEST_SHA256

    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    assert len(families) == 30
    assert {
        family["correction_task"]["operation"]
        for family in families
        if isinstance(family, dict)
        and isinstance(family.get("correction_task"), dict)
    } == {"add", "subtract"}

    training = manifest["training"]
    assert isinstance(training, dict)
    assert training["seeds"] == [42, 43, 44]
    assert training["max_episode_steps"] == 8

    endpoints = manifest["endpoints"]
    assert isinstance(endpoints, dict)
    primary = endpoints["primary"]
    assert isinstance(primary, dict)
    assert primary["level"] == "L1_local_transfer"
    assert primary["metric"] == "uncorrected_sibling_teacher_prefix_accuracy"


def test_frozen_confirmatory_families_have_complete_candidate_coverage():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)

    coverage = validate_confirmatory_candidate_coverage(manifest)

    assert coverage == {
        "tasks_checked": 60,
        "decisions_checked": 240,
    }


def test_confirmatory_loader_rejects_structurally_valid_hash_drift(tmp_path):
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    changed = deepcopy(manifest)
    changed["research_question"] = "post-freeze drift"
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(changed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        load_confirmatory_manifest(path)


def test_confirmatory_validator_rejects_post_freeze_endpoint_or_training_drift():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)

    changed_endpoint = deepcopy(manifest)
    endpoints = changed_endpoint["endpoints"]
    assert isinstance(endpoints, dict)
    primary = endpoints["primary"]
    assert isinstance(primary, dict)
    primary["metric"] = "goal_verified"
    with pytest.raises(ValueError, match="Primary endpoint metric changed"):
        validate_confirmatory_manifest(changed_endpoint)

    changed_training = deepcopy(manifest)
    training = changed_training["training"]
    assert isinstance(training, dict)
    training["seeds"] = [42]
    with pytest.raises(ValueError, match="seeds"):
        validate_confirmatory_manifest(changed_training)


def test_confirmatory_validator_rejects_sibling_leakage():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    leaked = deepcopy(manifest)
    design = leaked["family_design"]
    assert isinstance(design, dict)
    design["sibling_used_for_training"] = True

    with pytest.raises(ValueError, match="never enter training"):
        validate_confirmatory_manifest(leaked)
