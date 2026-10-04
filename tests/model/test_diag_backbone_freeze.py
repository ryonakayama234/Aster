"""DIAG v1 backbone-freeze causal Trial invariants."""

from pathlib import Path

import pytest

from aster.training.diag_backbone_freeze import (
    EXPECTED_ARM_CHECKPOINT_RECORDS,
    EXPECTED_ARM_TRAJECTORIES,
    EXPECTED_EXPERIMENT_UNITS,
    EXPECTED_FAMILY_BLOCKS,
    EXPECTED_UNIT_CHECKPOINT_RECORDS,
    TRIAL_BACKBONE_CONDITIONS,
    TRIAL_CYCLE_ID,
    TRIAL_FAMILY_IDS,
    TRIAL_FAMILY_STRATA,
    aggregate_backbone_freeze_units,
    summarize_backbone_freeze_unit,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    load_confirmatory_manifest,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_trial_definition_is_four_paired_family_blocks():
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

    assert TRIAL_CYCLE_ID == "diag-backbone-freeze-trial-0"
    assert TRIAL_BACKBONE_CONDITIONS == (True, False)
    assert EXPECTED_FAMILY_BLOCKS == 4
    assert EXPECTED_EXPERIMENT_UNITS == 8
    assert EXPECTED_ARM_TRAJECTORIES == 16
    assert EXPECTED_UNIT_CHECKPOINT_RECORDS == 40
    assert EXPECTED_ARM_CHECKPOINT_RECORDS == 80
    assert set(operation_by_family) == set(TRIAL_FAMILY_IDS)
    assert list(operation_by_family.values()).count("add") == 2
    assert list(operation_by_family.values()).count("subtract") == 2
    assert list(TRIAL_FAMILY_STRATA.values()).count("loss") == 2
    assert list(TRIAL_FAMILY_STRATA.values()).count("control") == 2


def test_condition_summary_uses_baseline_relative_onset_and_cr_divergence():
    unit = {
        "trajectory_summary": {
            "replay": {
                "sibling_margin_delta": 0.2,
                "repair_accuracy_delta": 0.1,
            },
            "correction": {
                "sibling_margin_delta": -0.4,
                "repair_accuracy_delta": 0.8,
            },
        },
        "checkpoints": [
            _checkpoint(0, replay_margin=-0.5, correction_margin=-0.5),
            _checkpoint(10, replay_margin=-0.4, correction_margin=-0.45),
            _checkpoint(25, replay_margin=-0.35, correction_margin=-0.6),
            _checkpoint(50, replay_margin=-0.3, correction_margin=-0.8),
            _checkpoint(100, replay_margin=-0.3, correction_margin=-0.9),
        ],
    }

    summary = summarize_backbone_freeze_unit(unit)

    assert summary["sibling_margin_correction_minus_replay_delta"] == pytest.approx(
        -0.6
    )
    assert summary["repair_accuracy_correction_minus_replay_delta"] == pytest.approx(
        0.7
    )
    assert summary["first_negative_correction_sibling_margin_change_step"] == 25
    assert (
        summary["first_negative_correction_minus_replay_divergence_step"] == 10
    )


def test_aggregation_pairs_conditions_by_family_and_computes_freeze_effect():
    units = []
    for family_id in TRIAL_FAMILY_IDS:
        stratum = TRIAL_FAMILY_STRATA[family_id]
        operation = "add" if family_id.endswith(("03", "15")) else "subtract"
        units.append(
            _fake_unit(
                family_id,
                operation,
                stratum,
                "full",
                sibling_d=-1.0 if stratum == "loss" else 0.1,
                repair_cr=0.6,
            )
        )
        units.append(
            _fake_unit(
                family_id,
                operation,
                stratum,
                "freeze",
                sibling_d=-0.2 if stratum == "loss" else 0.05,
                repair_cr=0.5,
            )
        )

    result = aggregate_backbone_freeze_units(units)

    assert result["status"] == "complete"
    assert result["independent_family_blocks"] == 4
    assert result["experiment_units"] == 8
    assert result["arm_training_trajectories"] == 16
    assert result["unit_checkpoint_records"] == 40
    assert result["arm_checkpoint_records"] == 80
    assert result["backbone_conditions_are_independent_samples"] is False
    assert result["checkpoint_observations_are_independent"] is False
    assert result["confirmatory_verdict"] is None

    family_summaries = result["family_summaries"]
    assert isinstance(family_summaries, list)
    loss = [
        row
        for row in family_summaries
        if isinstance(row, dict) and row.get("family_stratum") == "loss"
    ]
    assert len(loss) == 2
    assert all(row["freeze_effect"] == pytest.approx(0.8) for row in loss)


def test_aggregation_rejects_backbone_condition_that_changes_training_data():
    units = []
    for family_id in TRIAL_FAMILY_IDS:
        stratum = TRIAL_FAMILY_STRATA[family_id]
        operation = "add" if family_id.endswith(("03", "15")) else "subtract"
        full = _fake_unit(
            family_id,
            operation,
            stratum,
            "full",
            sibling_d=-0.5,
            repair_cr=0.5,
        )
        freeze = _fake_unit(
            family_id,
            operation,
            stratum,
            "freeze",
            sibling_d=-0.1,
            repair_cr=0.5,
        )
        if family_id == TRIAL_FAMILY_IDS[0]:
            freeze["training_examples_sha256"] = {
                "replay": "same-replay",
                "correction": "different-correction",
            }
        units.extend((full, freeze))

    with pytest.raises(
        ValueError, match="causal factor changed correction training examples"
    ):
        aggregate_backbone_freeze_units(units)


def _checkpoint(step, *, replay_margin, correction_margin):
    def arm(margin):
        return {
            "sibling": {
                "accuracy": 0.5,
                "nll": 1.0,
                "margin_mean": margin,
                "margin_min": margin - 0.1,
            }
        }

    return {
        "step": step,
        "replay": arm(replay_margin),
        "correction": arm(correction_margin),
    }


def _fake_unit(
    family_id,
    operation,
    stratum,
    condition,
    *,
    sibling_d,
    repair_cr,
):
    return {
        "cycle_id": TRIAL_CYCLE_ID,
        "source_git_sha": "a" * 40,
        "family_id": family_id,
        "family_stratum": stratum,
        "operation": operation,
        "backbone_condition": condition,
        "training_examples_sha256": {
            "replay": "same-replay",
            "correction": "same-correction",
        },
        "condition_summary": {
            "sibling_margin_correction_minus_replay_delta": sibling_d,
            "repair_accuracy_correction_minus_replay_delta": repair_cr,
        },
    }
