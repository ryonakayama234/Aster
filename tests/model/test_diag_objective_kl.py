"""DIAG v5 paired objective Trial: mathematical and evidence contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.training.decision import DecisionTrainConfig, train_decision_with_checkpoints
from aster.training.diag_objective_kl import (
    EXPECTED_ARM_CHECKPOINT_RECORDS,
    EXPECTED_ARM_TRAJECTORIES,
    EXPECTED_EXPERIMENT_UNITS,
    EXPECTED_FAMILY_BLOCKS,
    EXPECTED_OBJECTIVE_CONDITIONS,
    EXPECTED_UNIT_CHECKPOINT_RECORDS,
    OBJECTIVES,
    TRIAL_CYCLE_ID,
    TRIAL_FAMILY_IDS,
    TRIAL_FAMILY_STRATA,
    aggregate_objective_kl_units,
    assert_initial_objective_equivalence,
    assert_step0_equivalence,
    candidate_objective_components,
    objective_spec,
)


def _model():
    return DecisionModel(
        TinyLM(ModelConfig(vocab_size=32, context_length=16, width=16, heads=2, layers=1))
    )


def test_forward_kl_agrees_with_cross_entropy_at_parent_step_zero():
    parent = torch.tensor([0.3, -0.5, 0.2], requires_grad=True)
    student = parent.detach().clone().requires_grad_(True)

    total, ce, kl = candidate_objective_components(
        student, parent, 1, coefficient=1.0
    )
    baseline, _, _ = candidate_objective_components(
        student, parent, 1, coefficient=0.0
    )

    assert float(kl.item()) == pytest.approx(0.0, abs=1e-7)
    torch.testing.assert_close(total, ce, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(total, student, retain_graph=True)[0],
        torch.autograd.grad(baseline, student)[0],
        rtol=0.0,
        atol=1e-7,
    )
    assert parent.grad is None


def test_forward_kl_gradient_and_frozen_parent_for_changed_student():
    parent = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
    student = torch.tensor([-0.3, 0.4, -0.1], requires_grad=True)
    total, ce, kl = candidate_objective_components(
        student, parent, 1, coefficient=1.0
    )
    actual = torch.autograd.grad(total, student)[0]
    q = torch.softmax(parent.detach(), dim=0)
    p = torch.softmax(student.detach(), dim=0)
    target = torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(actual, 2 * p - q - target, atol=1e-6, rtol=0)
    assert float(kl.item()) >= -1e-6
    assert parent.grad is None
    assert ce.item() > 0


def test_candidate_loss_rejects_wrong_shape_and_parameter_sweep():
    scores = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="count mismatch"):
        candidate_objective_components(scores, torch.tensor([1.0]), 1, coefficient=1.0)
    with pytest.raises(ValueError, match="fixed coefficients"):
        candidate_objective_components(scores, scores, 1, coefficient=0.5)
    with pytest.raises(ValueError, match="out of range"):
        candidate_objective_components(scores, scores, 2, coefficient=1.0)


def test_objective_only_has_exact_parent_parameter_count():
    parent = _model()
    for objective in OBJECTIVES:
        candidate = deepcopy(parent)
        spec = objective_spec(parent, candidate, objective)
        assert spec["parent_decision_path_parameter_count"] == spec["decision_path_parameter_count"]
        assert spec["head_parameter_count"] == 17
        assert spec["trainability"]["trainable_parameter_names"] == [
            "head.projection.weight", "head.projection.bias"
        ] or set(spec["trainability"]["trainable_parameter_names"]) == {
            "head.projection.weight", "head.projection.bias"
        }


def test_step0_candidate_and_gradient_gates(monkeypatch):
    torch.manual_seed(13)
    parent = _model()
    examples = (
        SimpleNamespace(state={}, trajectory=(), candidates=("a", "b"), target_index=1),
    )

    def score(model, tokenizer, state, trajectory, candidates):
        del tokenizer, state, trajectory, candidates
        token_ids = torch.tensor([[1, 2], [2, 1]])
        return model(token_ids)

    monkeypatch.setattr("aster.training.diag_objective_kl.score_candidates", score)
    equality = assert_step0_equivalence(parent, deepcopy(parent), object(), examples)
    gradient = assert_initial_objective_equivalence(parent, object(), examples)
    assert equality["decision_examples_checked"] == 1
    assert equality["exact_candidate_score_identity"] is True
    assert gradient["gradient_identity_within_1e_6"] is True


def test_existing_checkpoint_trainer_accepts_opt_in_loss_callback():
    parent = _model()
    called = []

    def loss_fn(model, tokenizer, example):
        del tokenizer
        called.append(example)
        return (model.head.projection.weight ** 2).sum()

    history, snapshots, elapsed = train_decision_with_checkpoints(
        parent,
        object(),
        [object()],
        checkpoints=(0, 2),
        config=DecisionTrainConfig(steps=2, train_backbone=False),
        trainable_parameter_names=("head.projection.weight", "head.projection.bias"),
        loss_fn=loss_fn,
    )
    assert len(called) == len(history) == 2
    assert tuple(sorted(snapshots)) == (0, 2)
    assert tuple(sorted(elapsed)) == (0, 2)


def _unit(family_id, objective, *, q_correction="parent-c"):
    return {
        "cycle_id": TRIAL_CYCLE_ID,
        "source_git_sha": "a" * 40,
        "parent_artifact_id": "decision_model:" + "b" * 64,
        "family_id": family_id,
        "family_stratum": TRIAL_FAMILY_STRATA[family_id],
        "operation": "add" if family_id.endswith(("03", "15")) else "subtract",
        "objective": objective,
        "training_examples_sha256": {
            "replay": "same-r",
            "correction": "same-c",
        },
        "parent_q_sha256": {
            "replay": "parent-r",
            "correction": q_correction,
        },
        "condition_summary": {
            "sibling_margin_correction_minus_replay_delta": -0.5 if objective == "ce" else -0.2,
            "repair_accuracy_correction_minus_replay_delta": 0.25 if objective == "ce" else 0.25,
        },
    }


def test_trial_unit_counts_and_pairwise_aggregation():
    assert EXPECTED_FAMILY_BLOCKS == 4
    assert EXPECTED_OBJECTIVE_CONDITIONS == 2
    assert EXPECTED_EXPERIMENT_UNITS == 8
    assert EXPECTED_ARM_TRAJECTORIES == 16
    assert EXPECTED_UNIT_CHECKPOINT_RECORDS == 40
    assert EXPECTED_ARM_CHECKPOINT_RECORDS == 80
    units = [_unit(f, o) for f in TRIAL_FAMILY_IDS for o in OBJECTIVES]
    result = aggregate_objective_kl_units(units)
    assert result["status"] == "complete"
    assert result["causal_claim"] is False
    assert result["confirmatory_verdict"] is None
    assert result["diagnostic_classification"] is None
    assert result["family_summaries"][0]["contrasts"]["stability_gain_kl"] == pytest.approx(0.3)
    assert result["family_summaries"][0]["contrasts"]["repair_gain_kl"] == 0.0


def test_pairing_rejects_teacher_change_even_when_training_data_same():
    units = [_unit(f, o) for f in TRIAL_FAMILY_IDS for o in OBJECTIVES]
    units[1]["parent_q_sha256"]["correction"] = "changed"
    with pytest.raises(ValueError, match="changed correction parent Q"):
        aggregate_objective_kl_units(units)


def test_pairing_rejects_missing_or_extra_units():
    units = [_unit(f, o) for f in TRIAL_FAMILY_IDS for o in OBJECTIVES]
    assert aggregate_objective_kl_units(units[:-1])["status"] == "trial_incomplete"
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_objective_kl_units(units[:-1] + [units[0]])
