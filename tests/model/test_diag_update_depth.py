"""DIAG v2 update-depth Trial invariants."""

from copy import deepcopy

import pytest
import torch

import aster.training.decision as decision_training
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.training.decision import DecisionTrainConfig, train_decision_with_checkpoints
from aster.training.diag_update_depth import (
    EXPECTED_ARM_CHECKPOINT_RECORDS,
    EXPECTED_ARM_TRAJECTORIES,
    EXPECTED_EXPERIMENT_UNITS,
    EXPECTED_FAMILY_BLOCKS,
    EXPECTED_UNIT_CHECKPOINT_RECORDS,
    EXPECTED_UPDATE_DEPTH_CONDITIONS,
    PARAMETER_GROUPS,
    TRIAL_CYCLE_ID,
    TRIAL_FAMILY_IDS,
    TRIAL_FAMILY_STRATA,
    UPDATE_DEPTHS,
    _validate_frozen_group_drift,
    aggregate_update_depth_units,
    component_parameter_l2_drift,
    trainability_spec,
)


def _model(*, layers=2):
    return DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=32,
                context_length=16,
                width=16,
                heads=2,
                layers=layers,
            )
        )
    )


@pytest.mark.parametrize(
    ("update_depth", "expected_groups"),
    [
        ("head-only", {"decision_head"}),
        ("last-block", {"block_1", "final_norm", "decision_head"}),
        (
            "full",
            {"embeddings", "block_0", "block_1", "final_norm", "decision_head"},
        ),
    ],
)
def test_trainability_spec_is_exact(update_depth, expected_groups):
    model = _model()
    spec = trainability_spec(model, update_depth)

    assert set(spec["trainable_groups"]) == expected_groups
    assert set(spec["frozen_groups"]) == set(PARAMETER_GROUPS) - expected_groups
    assert spec["lm_head_in_decision_forward_path"] is False

    trainable_names = set(spec["trainable_parameter_names"])
    all_names = {name for name, _ in model.named_parameters()}
    assert trainable_names
    assert trainable_names | set(spec["frozen_parameter_names"]) == all_names
    assert trainable_names.isdisjoint(set(spec["frozen_parameter_names"]))
    assert not any(name.startswith("backbone.lm_head.") for name in trainable_names)


def test_update_depth_trial_is_frozen_to_two_transformer_blocks():
    with pytest.raises(ValueError, match="two-block TinyLM"):
        trainability_spec(_model(layers=1), "last-block")


def test_explicit_parameter_mask_overrides_legacy_train_backbone(monkeypatch):
    torch.manual_seed(13)
    parent = _model()
    candidate = deepcopy(parent)
    spec = trainability_spec(candidate, "head-only")
    trainable_names = tuple(spec["trainable_parameter_names"])

    def fake_loss(model, tokenizer, example):
        del tokenizer, example
        return sum(
            (parameter * parameter).sum()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    monkeypatch.setattr(decision_training, "decision_loss", fake_loss)
    _, snapshots, _ = train_decision_with_checkpoints(
        candidate,
        object(),
        (object(),),
        checkpoints=(0, 1),
        config=DecisionTrainConfig(
            steps=1,
            learning_rate=1e-2,
            train_backbone=True,
            seed=13,
        ),
        trainable_parameter_names=trainable_names,
    )

    for name, expected in parent.backbone.state_dict().items():
        torch.testing.assert_close(
            snapshots[1].backbone.state_dict()[name], expected, rtol=0, atol=0
        )
    assert any(
        not torch.equal(value, parent.head.state_dict()[name])
        for name, value in snapshots[1].head.state_dict().items()
    )


def test_component_drift_separates_parameter_groups():
    parent = _model()
    candidate = deepcopy(parent)
    with torch.no_grad():
        candidate.backbone.blocks[1].projection.weight.add_(0.25)
        candidate.head.projection.bias.add_(0.5)

    drift = component_parameter_l2_drift(parent, candidate)

    assert drift["block_1_l2"] > 0.0
    assert drift["decision_head_l2"] > 0.0
    assert drift["head_l2"] == drift["decision_head_l2"]
    assert drift["backbone_l2"] == pytest.approx(drift["block_1_l2"])
    assert drift["embeddings_l2"] == 0.0
    assert drift["block_0_l2"] == 0.0
    assert drift["final_norm_l2"] == 0.0
    assert drift["lm_head_l2"] == 0.0


def test_frozen_group_drift_is_an_exact_invariant():
    drift = {
        "embeddings_l2": 0.0,
        "block_0_l2": 0.0,
        "block_1_l2": 1.0,
        "final_norm_l2": 0.25,
        "lm_head_l2": 0.0,
        "decision_head_l2": 0.5,
        "head_l2": 0.5,
        "backbone_l2": 1.25,
    }
    unit = {
        "trainability": {
            "frozen_groups": ["embeddings", "block_0", "lm_head"],
        },
        "checkpoints": [
            {
                "step": 0,
                "replay": {"parameter_drift": {key: 0.0 for key in drift}},
                "correction": {"parameter_drift": {key: 0.0 for key in drift}},
            },
            {
                "step": 100,
                "replay": {"parameter_drift": dict(drift)},
                "correction": {"parameter_drift": dict(drift)},
            },
        ],
    }

    _validate_frozen_group_drift(unit)
    unit["checkpoints"][1]["correction"]["parameter_drift"]["block_0_l2"] = 1e-9

    with pytest.raises(RuntimeError, match="block_0"):
        _validate_frozen_group_drift(unit)


def test_trial_bookkeeping_is_four_families_by_three_conditions():
    assert TRIAL_CYCLE_ID == "diag-update-depth-trial-0"
    assert UPDATE_DEPTHS == ("head-only", "last-block", "full")
    assert EXPECTED_FAMILY_BLOCKS == 4
    assert EXPECTED_UPDATE_DEPTH_CONDITIONS == 3
    assert EXPECTED_EXPERIMENT_UNITS == 12
    assert EXPECTED_ARM_TRAJECTORIES == 24
    assert EXPECTED_UNIT_CHECKPOINT_RECORDS == 60
    assert EXPECTED_ARM_CHECKPOINT_RECORDS == 120
    assert list(TRIAL_FAMILY_STRATA.values()).count("loss") == 2
    assert list(TRIAL_FAMILY_STRATA.values()).count("control") == 2


def test_aggregation_computes_paired_stability_and_repair_contrasts():
    units = []
    for family_id in TRIAL_FAMILY_IDS:
        stratum = TRIAL_FAMILY_STRATA[family_id]
        operation = "add" if family_id.endswith(("03", "15")) else "subtract"
        units.extend(
            [
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    "head-only",
                    sibling_d=-0.1,
                    repair_cr=0.2,
                ),
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    "last-block",
                    sibling_d=-0.4,
                    repair_cr=0.5,
                ),
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    "full",
                    sibling_d=-1.0,
                    repair_cr=0.6,
                ),
            ]
        )

    result = aggregate_update_depth_units(units)

    assert result["status"] == "complete"
    assert result["experiment_units"] == 12
    assert result["arm_training_trajectories"] == 24
    assert result["unit_checkpoint_records"] == 60
    assert result["arm_checkpoint_records"] == 120
    assert result["update_depth_conditions_are_independent_samples"] is False
    assert result["checkpoint_observations_are_independent"] is False
    assert result["confirmatory_verdict"] is None

    rows = result["family_summaries"]
    assert isinstance(rows, list)
    first = rows[0]
    assert first["contrasts"]["stability_gain_last_vs_full"] == pytest.approx(0.6)
    assert first["contrasts"]["stability_cost_last_vs_head"] == pytest.approx(-0.3)
    assert first["contrasts"]["repair_gain_last_vs_head"] == pytest.approx(0.3)
    assert first["contrasts"]["repair_gap_last_to_full"] == pytest.approx(-0.1)


def test_aggregation_rejects_condition_that_changes_training_data():
    units = []
    for family_id in TRIAL_FAMILY_IDS:
        stratum = TRIAL_FAMILY_STRATA[family_id]
        operation = "add" if family_id.endswith(("03", "15")) else "subtract"
        for update_depth in UPDATE_DEPTHS:
            units.append(
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    update_depth,
                    sibling_d=-0.1,
                    repair_cr=0.2,
                )
            )
    units[1]["training_examples_sha256"] = {
        "replay": "same-replay",
        "correction": "changed-correction",
    }

    with pytest.raises(ValueError, match="changed correction training examples"):
        aggregate_update_depth_units(units)


def _fake_unit(
    family_id,
    operation,
    stratum,
    update_depth,
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
        "update_depth": update_depth,
        "training_examples_sha256": {
            "replay": "same-replay",
            "correction": "same-correction",
        },
        "condition_summary": {
            "sibling_margin_correction_minus_replay_delta": sibling_d,
            "repair_accuracy_correction_minus_replay_delta": repair_cr,
        },
    }
