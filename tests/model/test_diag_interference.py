"""DIAG v0 Trial 0 invariants that do not require the local ACT artifact."""

from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.training.diag_interference import (
    EXPECTED_TRIAL_UNITS,
    TRIAL_CHECKPOINTS,
    TRIAL_CYCLE_ID,
    TRIAL_FAMILY_IDS,
    TRIAL_SEEDS,
    aggregate_diag_trial_units,
    model_parameter_l2_drift,
    summarize_diag_trial_unit,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    load_confirmatory_manifest,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_trial_definition_is_small_fixed_and_operation_balanced():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    operations = {}
    for raw in families:
        if not isinstance(raw, dict):
            continue
        family_id = raw.get("family_id")
        correction = raw.get("correction_task")
        if (
            family_id in TRIAL_FAMILY_IDS
            and isinstance(correction, dict)
            and isinstance(correction.get("operation"), str)
        ):
            operations[family_id] = correction["operation"]

    assert len(TRIAL_FAMILY_IDS) == 6
    assert TRIAL_SEEDS == (42,)
    assert TRIAL_CHECKPOINTS == (0, 10, 25, 50, 100)
    assert EXPECTED_TRIAL_UNITS == 6
    assert set(operations) == set(TRIAL_FAMILY_IDS)
    assert list(operations.values()).count("add") == 3
    assert list(operations.values()).count("subtract") == 3


def test_parameter_drift_separates_head_from_backbone():
    torch.manual_seed(7)
    parent = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=258,
                context_length=16,
                width=16,
                heads=2,
                layers=1,
            )
        )
    )
    candidate = deepcopy(parent)
    with torch.no_grad():
        candidate.head.projection.bias.add_(1.0)

    drift = model_parameter_l2_drift(parent, candidate)

    assert drift["head_l2"] > 0.0
    assert drift["backbone_l2"] == 0.0


def test_trial_aggregation_keeps_6_trajectories_distinct_from_30_checkpoints():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    operation_by_family = {
        raw["family_id"]: raw["correction_task"]["operation"]
        for raw in families
        if isinstance(raw, dict)
        and raw.get("family_id") in TRIAL_FAMILY_IDS
        and isinstance(raw.get("correction_task"), dict)
    }
    units = [
        _fake_unit(family_id, seed, operation_by_family[family_id])
        for family_id in TRIAL_FAMILY_IDS
        for seed in TRIAL_SEEDS
    ]

    result = aggregate_diag_trial_units(units)

    assert result["status"] == "complete"
    assert result["independent_training_trajectories"] == 6
    assert result["checkpoint_observations"] == 30
    assert result["checkpoint_observations_are_independent"] is False
    assert result["confirmatory_verdict"] is None


def test_unit_summary_tracks_within_trajectory_curve_not_independent_samples():
    unit = {
        "checkpoints": [
            _checkpoint(0, sibling_margin=0.5),
            _checkpoint(10, sibling_margin=0.2),
            _checkpoint(25, sibling_margin=-0.1),
            _checkpoint(50, sibling_margin=-0.3),
            _checkpoint(100, sibling_margin=-0.4),
        ]
    }

    summary = summarize_diag_trial_unit(unit)
    correction = summary["correction"]
    assert isinstance(correction, dict)
    assert correction["first_negative_sibling_margin_step"] == 25
    assert correction["minimum_sibling_margin"] == pytest.approx(-0.4)
    assert correction["repair_accuracy_delta"] == pytest.approx(0.5)
    assert correction["sibling_margin_delta"] == pytest.approx(-0.9)


def _fake_unit(family_id, seed, operation):
    arm_summary = {
        "repair_accuracy_delta": 0.25,
        "repair_nll_delta": -0.1,
        "sibling_accuracy_delta": -0.25,
        "sibling_nll_delta": 0.2,
        "sibling_margin_delta": -0.5,
        "minimum_sibling_margin": -0.5,
        "first_negative_sibling_margin_step": 25,
        "base_accuracy_delta": 0.0,
        "base_nll_delta": 0.05,
        "head_l2_step100": 0.1,
        "backbone_l2_step100": 0.2,
    }
    return {
        "cycle_id": TRIAL_CYCLE_ID,
        "source_git_sha": "a" * 40,
        "family_id": family_id,
        "operation": operation,
        "seed": seed,
        "trajectory_summary": {
            "replay": dict(arm_summary),
            "correction": dict(arm_summary),
        },
    }


def _checkpoint(step, *, sibling_margin):
    def arm(repair_accuracy):
        return {
            "repair": {
                "accuracy": repair_accuracy,
                "nll": 1.0 - repair_accuracy,
                "margin_mean": repair_accuracy,
                "margin_min": repair_accuracy - 0.1,
            },
            "sibling": {
                "accuracy": 0.5,
                "nll": 1.0,
                "margin_mean": sibling_margin,
                "margin_min": sibling_margin - 0.1,
            },
            "base_retention": {
                "accuracy": 0.75,
                "nll": 0.5,
                "margin_mean": 0.2,
                "margin_min": 0.1,
            },
            "parameter_drift": {
                "head_l2": step / 100.0,
                "backbone_l2": step / 50.0,
            },
        }

    repair_accuracy = 0.25 if step == 0 else (0.75 if step == 100 else 0.5)
    return {
        "step": step,
        "replay": arm(repair_accuracy),
        "correction": arm(repair_accuracy),
    }
