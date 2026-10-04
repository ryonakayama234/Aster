"""Gate 4 confirmatory aggregation and exact-inference invariants."""

from pathlib import Path

import pytest

from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
)
from aster.training.learn_confirmatory_analysis import (
    aggregate_confirmatory_units,
    exact_two_sided_sign_test_p_value,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_exact_sign_test_matches_wolfram_reference_values():
    assert exact_two_sided_sign_test_p_value(21, 9) == pytest.approx(
        0.04277394525706768
    )
    assert exact_two_sided_sign_test_p_value(20, 0) == pytest.approx(
        1.9073486328125e-6
    )
    assert exact_two_sided_sign_test_p_value(15, 15) == 1.0
    assert exact_two_sided_sign_test_p_value(0, 0) is None


def test_partial_confirmatory_campaign_exposes_progress_not_endpoint_metrics():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_ids = _family_ids(manifest)
    unit = _unit(manifest, family_ids[0], 42, "win")

    result = aggregate_confirmatory_units(manifest, [unit])

    assert result["status"] == "measurement_incomplete"
    assert result["completed_units"] == 1
    assert result["expected_units"] == 90
    assert result["primary"] is None
    assert result["family_results"] is None
    assert result["verdict"] is None


def test_confirmatory_primary_rule_supports_21_to_9_family_result():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_ids = _family_ids(manifest)
    units = []
    for index, family_id in enumerate(family_ids):
        outcome = "win" if index < 21 else "loss"
        for seed in (42, 43, 44):
            units.append(_unit(manifest, family_id, seed, outcome))

    result = aggregate_confirmatory_units(manifest, units)
    primary = result["primary"]
    verdict = result["verdict"]
    assert isinstance(primary, dict)
    assert isinstance(verdict, dict)
    assert result["status"] == "complete"
    assert primary["wins"] == 21
    assert primary["losses"] == 9
    assert primary["ties"] == 0
    assert primary["sign_test_p_value"] == pytest.approx(0.04277394525706768)
    assert primary["median_family_accuracy_delta"] == 0.25
    assert verdict["label"] == "Supported"


def test_confirmatory_primary_rule_can_falsify_correction_transfer():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_ids = _family_ids(manifest)
    units = []
    for index, family_id in enumerate(family_ids):
        outcome = "loss" if index < 21 else "win"
        for seed in (42, 43, 44):
            units.append(_unit(manifest, family_id, seed, outcome))

    result = aggregate_confirmatory_units(manifest, units)
    primary = result["primary"]
    verdict = result["verdict"]
    assert isinstance(primary, dict)
    assert isinstance(verdict, dict)
    assert primary["wins"] == 9
    assert primary["losses"] == 21
    assert verdict["label"] == "Not supported"


def test_tie_heavy_result_is_inconclusive_even_with_more_wins_than_losses():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_ids = _family_ids(manifest)
    units = []
    for index, family_id in enumerate(family_ids):
        outcome = "win" if index < 19 else "tie"
        for seed in (42, 43, 44):
            units.append(_unit(manifest, family_id, seed, outcome))

    result = aggregate_confirmatory_units(manifest, units)
    primary = result["primary"]
    verdict = result["verdict"]
    assert isinstance(primary, dict)
    assert isinstance(verdict, dict)
    assert primary["non_tied_families"] == 19
    assert verdict == {
        "label": "Inconclusive",
        "reason": "fewer_than_minimum_non_tied_families",
    }


def _family_ids(manifest):
    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    return [
        family["family_id"]
        for family in families
        if isinstance(family, dict) and isinstance(family.get("family_id"), str)
    ]


def _unit(manifest, family_id, seed, outcome):
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    if outcome == "win":
        replay_correct, correction_correct = 2, 3
    elif outcome == "loss":
        replay_correct, correction_correct = 3, 2
    elif outcome == "tie":
        replay_correct = correction_correct = 2
    else:
        raise AssertionError(outcome)

    return {
        "schema_version": "aster-learn-confirmatory-unit-0",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest_sha256,
        "family_id": family_id,
        "seed": seed,
        "measurement_git_sha": "d" * 40,
        "run_id": f"{family_id}-seed-{seed}",
        "primary": {
            "examples": 4,
            "replay_correct": replay_correct,
            "correction_correct": correction_correct,
            "replay_accuracy": replay_correct / 4,
            "correction_accuracy": correction_correct / 4,
            "accuracy_delta": (correction_correct - replay_correct) / 4,
            "replay_nll": 1.2,
            "correction_nll": 1.0,
            "nll_effect_replay_minus_correction": 0.2,
        },
        "repair": {
            arm: {"examples": 4, "accuracy": 0.5, "nll": 1.0}
            for arm in ("parent", "replay", "correction")
        },
        "sequential": {
            arm: {
                "run_id": f"{family_id}-{seed}-{arm}",
                "task_success": arm == "correction",
                "goal_verified": arm == "correction",
                "steps": 4,
            }
            for arm in ("parent", "replay", "correction")
        },
        "resources": {
            arm: {
                "optimizer_steps": 100,
                "encoded_tokens_presented": 1,
                "padded_token_positions": 1,
                "candidate_sequences_presented": 1,
                "wall_seconds": 0.1,
                "process_max_rss_before_kib": 1,
                "process_max_rss_after_kib": 1,
                "process_max_rss_increase_kib": 0,
            }
            for arm in ("replay", "correction")
        },
    }
