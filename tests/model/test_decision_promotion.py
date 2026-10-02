"""Fit checkpoints can be promoted for model-only ACT without fabricating calibration."""

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.inference.decide import score_candidates
from aster.model.decision_artifact import load_decision_artifact
from aster.model.decision_promotion import promote_decision_fit_checkpoint
from aster.training.decision_fit import (
    DecisionFitArm,
    DecisionFitStudyConfig,
    load_decision_fit_checkpoint,
    run_logged_decision_fit_study,
)


def test_promote_fit_checkpoint_preserves_scores_and_marks_uncalibrated(tmp_path):
    suite = build_calculate_and_store_suite()
    run_path = run_logged_decision_fit_study(
        tmp_path / "fit-root",
        suite,
        config=DecisionFitStudyConfig(
            epochs=1,
            learning_rate=1e-3,
            train_backbone=True,
            seed=31,
            target_vocab_size=320,
            min_pair_frequency=1,
            context_length=2048,
            width=8,
            heads=2,
            layers=1,
            stability_window=1,
        ),
        arms=(
            DecisionFitArm(
                "one-fixed",
                ("train-add-small/step-0",),
                "fixed",
            ),
        ),
        source_git_sha="promotion-test",
    )
    checkpoint = run_path / "arms" / "one-fixed" / "checkpoint"
    fit_model, fit_tokenizer, fit_manifest = load_decision_fit_checkpoint(checkpoint)

    promoted_dir = tmp_path / "promoted"
    promoted_manifest = promote_decision_fit_checkpoint(checkpoint, promoted_dir)
    model, tokenizer, manifest = load_decision_artifact(promoted_dir)

    case = suite.cases_for("dev")[0]
    with torch.no_grad():
        expected = score_candidates(
            fit_model,
            fit_tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()
        actual = score_candidates(
            model,
            tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert manifest == promoted_manifest
    assert manifest["source_checkpoint_id"] == fit_manifest["checkpoint_id"]
    assert manifest["model_id"] == fit_manifest["checkpoint_id"]
    assert manifest["candidate_builder_id"] == "calculate-and-store-v0"
    assert manifest["serializer_id"] == "aster-decision-input-0"
    assert manifest["calibration"] == {"status": "not_run"}
    assert manifest["routing"] == {"status": "not_configured"}
    assert manifest["initialization"] == "promoted_fit_checkpoint"
    assert manifest["train_config"]["source"] == "decision_fit_checkpoint"
    assert manifest["resume_supported"] is False


def test_uncalibrated_artifact_rejects_routing_thresholds(tmp_path):
    suite = build_calculate_and_store_suite()
    run_path = run_logged_decision_fit_study(
        tmp_path / "fit-root",
        suite,
        config=DecisionFitStudyConfig(
            epochs=1,
            seed=32,
            target_vocab_size=320,
            context_length=2048,
            width=8,
            heads=2,
            layers=1,
            stability_window=1,
        ),
        arms=(DecisionFitArm("one-fixed", ("train-add-small/step-0",), "fixed"),),
    )
    checkpoint = run_path / "arms" / "one-fixed" / "checkpoint"
    model, tokenizer, manifest = load_decision_fit_checkpoint(checkpoint)

    from aster.model.decision_artifact import save_decision_artifact

    with pytest.raises(ValueError, match="must not configure routing thresholds"):
        save_decision_artifact(
            tmp_path / "bad",
            model,
            tokenizer,
            model_id=manifest["checkpoint_id"],
            candidate_builder_id="calculate-and-store-v0",
            suite_id=manifest["suite_id"],
            suite_sha256=manifest["suite_sha256"],
            train_config={},
            temperature=None,
            autonomous_threshold=0.8,
            fallback_threshold=0.6,
        )
