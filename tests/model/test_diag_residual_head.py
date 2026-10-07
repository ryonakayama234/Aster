"""DIAG v4 zero-init residual-head Trial invariants."""

from copy import deepcopy

import pytest
import torch

from aster.model.decision_head import DecisionModel, ResidualDecisionHead
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.training.diag_residual_head import (
    ARCHITECTURES,
    EXPECTED_ARCHITECTURE_CONDITIONS,
    EXPECTED_ARM_CHECKPOINT_RECORDS,
    EXPECTED_ARM_TRAJECTORIES,
    EXPECTED_EXPERIMENT_UNITS,
    EXPECTED_FAMILY_BLOCKS,
    EXPECTED_RESIDUAL_BRANCH_PARAMS,
    EXPECTED_UNIT_CHECKPOINT_RECORDS,
    TRIAL_CYCLE_ID,
    TRIAL_FAMILY_IDS,
    TRIAL_FAMILY_STRATA,
    aggregate_residual_head_units,
    assert_step0_equivalence,
    build_architecture_model,
)


def _model():
    return DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=32,
                context_length=16,
                width=16,
                heads=2,
                layers=1,
            )
        )
    )


def test_residual_head_copies_linear_scores_exactly_at_step0():
    torch.manual_seed(7)
    parent = _model()
    treatment = build_architecture_model(parent, "residual-head")
    token_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ]
    )

    parent.eval()
    treatment.eval()
    with torch.no_grad():
        parent_scores = parent(token_ids)
        treatment_scores = treatment(token_ids)

    assert torch.equal(parent_scores, treatment_scores)
    assert torch.equal(torch.argmax(parent_scores), torch.argmax(treatment_scores))
    assert isinstance(treatment.head, ResidualDecisionHead)
    assert torch.count_nonzero(treatment.head.residual_output.weight) == 0
    assert torch.count_nonzero(treatment.head.residual_output.bias) == 0
    assert torch.count_nonzero(treatment.head.residual_input.weight) > 0


def test_residual_branch_adds_exactly_289_parameters():
    parent = _model()
    treatment = build_architecture_model(parent, "residual-head")

    assert isinstance(treatment.head, ResidualDecisionHead)
    residual_params = (
        sum(parameter.numel() for parameter in treatment.head.parameters())
        - sum(parameter.numel() for parameter in parent.head.parameters())
    )
    assert residual_params == EXPECTED_RESIDUAL_BRANCH_PARAMS == 289


def test_architecture_builder_preserves_parent_and_caller_rng():
    torch.manual_seed(13)
    parent = _model()
    parent_state = deepcopy(parent.state_dict())
    before = torch.random.get_rng_state().clone()

    first = build_architecture_model(parent, "residual-head")
    after = torch.random.get_rng_state().clone()
    second = build_architecture_model(parent, "residual-head")

    assert torch.equal(before, after)
    for name, tensor in parent_state.items():
        torch.testing.assert_close(parent.state_dict()[name], tensor, rtol=0, atol=0)
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], tensor, rtol=0, atol=0)


def test_step0_equivalence_helper_checks_every_candidate(monkeypatch):
    parent = _model()
    treatment = build_architecture_model(parent, "residual-head")

    class Example:
        state = {}
        trajectory = ()
        candidates = (object(), object(), object())

    calls = []

    def fake_score(model, tokenizer, state, trajectory, candidates):
        del tokenizer, state, trajectory
        calls.append(model)
        assert len(candidates) == 3
        return torch.tensor([0.1, 0.2, 0.3])

    monkeypatch.setattr(
        "aster.training.diag_residual_head.score_candidates",
        fake_score,
    )
    result = assert_step0_equivalence(
        parent,
        treatment,
        object(),
        (Example(), Example()),
    )

    assert result["exact_candidate_score_identity"] is True
    assert result["exact_argmax_identity"] is True
    assert result["decision_examples_checked"] == 2
    assert result["candidate_scores_checked"] == 6
    assert result["max_abs_score_difference"] == 0.0
    assert len(calls) == 4


def test_trial_bookkeeping_is_four_families_by_two_architectures():
    assert TRIAL_CYCLE_ID == "diag-residual-head-trial-0"
    assert ARCHITECTURES == ("linear-head", "residual-head")
    assert EXPECTED_FAMILY_BLOCKS == 4
    assert EXPECTED_ARCHITECTURE_CONDITIONS == 2
    assert EXPECTED_EXPERIMENT_UNITS == 8
    assert EXPECTED_ARM_TRAJECTORIES == 16
    assert EXPECTED_UNIT_CHECKPOINT_RECORDS == 40
    assert EXPECTED_ARM_CHECKPOINT_RECORDS == 80
    assert list(TRIAL_FAMILY_STRATA.values()).count("loss") == 2
    assert list(TRIAL_FAMILY_STRATA.values()).count("control") == 2


def test_aggregation_computes_residual_stability_and_repair_contrasts():
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
                    "linear-head",
                    sibling_d=-1.0,
                    repair_cr=0.6,
                ),
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    "residual-head",
                    sibling_d=-0.4,
                    repair_cr=0.8,
                ),
            ]
        )

    result = aggregate_residual_head_units(units)

    assert result["status"] == "complete"
    assert result["experiment_units"] == 8
    assert result["arm_training_trajectories"] == 16
    assert result["unit_checkpoint_records"] == 40
    assert result["arm_checkpoint_records"] == 80
    assert result["architecture_conditions_are_independent_samples"] is False
    assert result["checkpoint_observations_are_independent"] is False
    assert result["confirmatory_verdict"] is None
    assert result["diagnostic_classification"] is None
    assert result["stop_rule_satisfied"] is True

    rows = result["family_summaries"]
    assert isinstance(rows, list)
    first = rows[0]
    assert first["contrasts"]["stability_gain_residual"] == pytest.approx(0.6)
    assert first["contrasts"]["repair_gain_residual"] == pytest.approx(0.2)


def test_aggregation_rejects_architecture_that_changes_training_data():
    units = []
    for family_id in TRIAL_FAMILY_IDS:
        stratum = TRIAL_FAMILY_STRATA[family_id]
        operation = "add" if family_id.endswith(("03", "15")) else "subtract"
        for architecture in ARCHITECTURES:
            units.append(
                _fake_unit(
                    family_id,
                    operation,
                    stratum,
                    architecture,
                    sibling_d=-0.1,
                    repair_cr=0.2,
                )
            )
    units[1]["training_examples_sha256"] = {
        "replay": "same-replay",
        "correction": "changed-correction",
    }

    with pytest.raises(ValueError, match="changed correction training examples"):
        aggregate_residual_head_units(units)


def _fake_unit(
    family_id,
    operation,
    stratum,
    architecture,
    *,
    sibling_d,
    repair_cr,
):
    return {
        "cycle_id": TRIAL_CYCLE_ID,
        "source_git_sha": "a" * 40,
        "parent_artifact_id": "decision_model:" + "b" * 64,
        "family_id": family_id,
        "family_stratum": stratum,
        "operation": operation,
        "architecture": architecture,
        "training_examples_sha256": {
            "replay": "same-replay",
            "correction": "same-correction",
        },
        "condition_summary": {
            "sibling_margin_correction_minus_replay_delta": sibling_d,
            "repair_accuracy_correction_minus_replay_delta": repair_cr,
        },
    }
