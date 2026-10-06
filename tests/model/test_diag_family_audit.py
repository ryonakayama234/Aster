"""DIAG v3 read-only evidence/extraction wiring tests."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import aster.training.diag_family_audit as audit
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import DecisionExample
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.diag_family_features import build_frozen_feature_schema


class _FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, 256)
        if add_eos:
            ids.append(257)
        return ids


def _example() -> DecisionExample:
    return DecisionExample(
        state=RuntimeState(
            task={
                "task_id": "test-task",
                "kind": "calculate_and_store",
                "operation": "add",
                "left": 2,
                "right": 3,
                "store_as": "sum",
            },
            memory={},
            step=0,
        ),
        trajectory=Trajectory(()),
        candidates=(
            Action.tool("calculator", operation="add", left=2, right=3),
            Action.stop(),
        ),
        target_index=0,
        teacher="test",
    )


def _measurement() -> dict[str, object]:
    return {
        "serialized_input_char_length": 100,
        "encoded_input_token_length": 80,
        "_input_tokens": [1, 2, 3],
        "target_raw_score": 0.5,
        "best_non_target_raw_score": 1.0,
        "target_margin": -0.5,
        "target_rank": 2,
        "parent_correct": 0,
        "raw_target_nll": 0.9,
        "raw_score_spread": 0.5,
        "raw_score_entropy": 0.6,
        "target_action_type": "tool:calculator",
        "top_competitor_action_type": "stop",
        "candidate_set_size": 2,
        "target_candidate_token_length": 30,
        "top_competitor_token_length": 10,
        "target_minus_competitor_token_length_difference": 20,
    }


def test_measure_decision_records_raw_parent_and_action_geometry(monkeypatch):
    model = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=300,
                context_length=512,
                width=8,
                heads=2,
                layers=1,
            )
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    monkeypatch.setattr(
        audit,
        "score_candidates",
        lambda model, tokenizer, state, trajectory, candidates: torch.tensor([0.5, 1.0]),
    )
    tokenizer = cast(AsterTokenizer, _FakeTokenizer())
    before = audit._model_state_digest(model)
    with torch.inference_mode():
        measured = audit._measure_decision(_example(), model, tokenizer)
    after = audit._model_state_digest(model)

    assert measured["target_raw_score"] == pytest.approx(0.5)
    assert measured["best_non_target_raw_score"] == pytest.approx(1.0)
    assert measured["target_margin"] == pytest.approx(-0.5)
    assert measured["target_rank"] == 2
    assert measured["parent_correct"] == 0
    assert measured["target_action_type"] == "tool:calculator"
    assert measured["top_competitor_action_type"] == "stop"
    assert measured["target_candidate_token_length"] != measured["encoded_input_token_length"]
    assert before == after


def test_extract_family_rows_match_frozen_schema_without_outcomes(monkeypatch):
    families = []
    for index in range(1, 31):
        operation = "add" if index <= 15 else "subtract"
        left = index + 10
        right = index if operation == "add" else index - 1
        families.append(
            {
                "family_id": f"family-{index:02d}",
                "correction_task": {
                    "task_id": f"c-{index:02d}",
                    "kind": "calculate_and_store",
                    "operation": operation,
                    "left": left,
                    "right": right,
                    "store_as": f"c{index}",
                },
                "sibling_task": {
                    "task_id": f"s-{index:02d}",
                    "kind": "calculate_and_store",
                    "operation": operation,
                    "left": left,
                    "right": right,
                    "store_as": f"s{index}",
                },
            }
        )
    manifest: dict[str, object] = {"family_design": {"families": families}}
    example = _example()
    monkeypatch.setattr(audit, "_teacher_examples", lambda task: [example, example])
    monkeypatch.setattr(
        audit,
        "_measure_decision",
        lambda example, model, tokenizer: _measurement(),
    )

    model = cast(DecisionModel, object())
    tokenizer = cast(AsterTokenizer, object())
    first = audit.extract_family_feature_rows(manifest, model, tokenizer)
    second = audit.extract_family_feature_rows(manifest, model, tokenizer)

    assert first == second
    assert len(first) == 30
    expected = set(cast(list[str], build_frozen_feature_schema()["output_names"]))
    assert set(first[0]) == expected | {"family_id"}
    assert first[0]["stratum_index"] == 1
    assert first[14]["stratum_index"] == 15
    assert first[15]["stratum_index"] == 1
    assert "primary_accuracy_delta" not in first[0]
    assert "outcome" not in first[0]


def test_reconstruct_confirmatory_evidence_requires_exact_canonical_match(monkeypatch):
    result = {
        "schema_version": "aster-learn-confirmatory-result-0",
        "protocol_id": audit.PROTOCOL_ID,
        "manifest_sha256": audit.EXPECTED_MANIFEST_SHA256,
        "measurement_git_sha": audit.EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA,
        "status": "complete",
        "expected_units": 90,
        "completed_units": 90,
        "family_results": [
            {"family_id": f"family-{index:02d}"} for index in range(30)
        ],
        "primary": {
            "wins": 6,
            "losses": 19,
            "ties": 5,
            "median_family_accuracy_delta": -0.25,
        },
        "verdict": {
            "label": "Not supported",
            "reason": "primary_rule_not_supported",
        },
    }
    units = {
        (f"family-{family:02d}", seed): (object(), {"unit": True})
        for family in range(30)
        for seed in (42, 43, 44)
    }
    monkeypatch.setattr(
        audit,
        "discover_completed_confirmatory_units",
        lambda root, manifest: units,
    )
    monkeypatch.setattr(
        audit,
        "aggregate_confirmatory_units",
        lambda manifest, rows: result,
    )
    monkeypatch.setattr(
        audit,
        "_discover_canonical_confirmatory_result",
        lambda root, manifest: ("canonical-run", dict(result)),
    )

    reconstructed, run_id = audit.reconstruct_confirmatory_evidence(".", {})
    assert reconstructed == result
    assert run_id == "canonical-run"

    changed = dict(result)
    changed["family_results"] = [{"family_id": "different"}]
    monkeypatch.setattr(
        audit,
        "_discover_canonical_confirmatory_result",
        lambda root, manifest: ("canonical-run", changed),
    )
    with pytest.raises(RuntimeError, match="family_results"):
        audit.reconstruct_confirmatory_evidence(".", {})


def test_verify_diag_v2_run_identity_requires_exact_frozen_campaign(tmp_path):
    run_dir = tmp_path / "runs" / audit.EXPECTED_DIAG_V2_RUN_ID
    run_dir.mkdir(parents=True)
    run = {
        "schema_version": "aster-run-0",
        "run_id": audit.EXPECTED_DIAG_V2_RUN_ID,
        "kind": "diag_update_depth_trial_campaign",
        "status": "completed",
        "inputs": {
            "cycle_id": audit.DIAG_V2_CYCLE_ID,
            "source_git_sha": audit.EXPECTED_DIAG_V2_SOURCE_GIT_SHA,
            "families": list(audit.DIAG_V2_FAMILY_IDS),
            "planned_experiment_units": 12,
        },
    }
    result = {
        "schema_version": "aster-diag-update-depth-result-0",
        "cycle_id": audit.DIAG_V2_CYCLE_ID,
        "status": "complete",
        "source_git_sha": audit.EXPECTED_DIAG_V2_SOURCE_GIT_SHA,
        "independent_family_blocks": 4,
        "experiment_units": 12,
        "family_summaries": [
            {"family_id": family_id}
            for family_id in audit.DIAG_V2_FAMILY_IDS
        ],
    }
    import json

    (run_dir / "run.json").write_text(
        json.dumps(run),
        encoding="utf-8",
    )
    (run_dir / "diag-update-depth-results.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    identity = audit.verify_diag_v2_run_identity(tmp_path)
    assert identity["identity_verified"] is True
    assert identity["run_id"] == audit.EXPECTED_DIAG_V2_RUN_ID
    assert identity["family_ids"] == list(audit.DIAG_V2_FAMILY_IDS)

    result["source_git_sha"] = "0" * 40
    (run_dir / "diag-update-depth-results.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source Git SHA"):
        audit.verify_diag_v2_run_identity(tmp_path)


def test_reconstruct_confirmatory_evidence_reports_missing_units(monkeypatch):
    incomplete = {
        "status": "measurement_incomplete",
        "expected_units": 90,
        "completed_units": 88,
        "missing_units": [
            {"family_id": "family-03", "seed": 43},
            {"family_id": "family-12", "seed": 44},
        ],
    }
    units = {
        ("family-01", seed): (object(), {"unit": True})
        for seed in (42, 43)
    }
    monkeypatch.setattr(
        audit,
        "discover_completed_confirmatory_units",
        lambda root, manifest: units,
    )
    monkeypatch.setattr(
        audit,
        "aggregate_confirmatory_units",
        lambda manifest, rows: incomplete,
    )

    with pytest.raises(
        RuntimeError,
        match=r"discovered=2, completed=88, expected=90.*family-03/seed=43.*family-12/seed=44",
    ):
        audit.reconstruct_confirmatory_evidence(".", {})
