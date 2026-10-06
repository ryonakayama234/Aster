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


def test_reconstruct_confirmatory_evidence_requires_exact_canonical_match(monkeypatch, tmp_path):
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
            "sign_test_p_value": 0.01463329792022705,
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
        "validate_preserved_learn_outcome_identity",
        lambda result, manifest: None,
    )
    monkeypatch.setattr(
        audit,
        "resolve_act_parent_artifact",
        lambda root, run_id: ("decision_model:" + ("a" * 64), tmp_path / "artifact"),
    )
    monkeypatch.setattr(
        audit,
        "validate_confirmatory_unit_lineage",
        lambda units, manifest, **kwargs: None,
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



def test_verify_learn_evidence_source_marks_reproduction_without_claiming_original(
    monkeypatch,
    tmp_path,
):
    act_run_id = "5a8076571dca446fa07190cf4fc62509"
    parent_artifact_id = "decision_model:" + ("a" * 64)
    manifest = {
        "protocol_id": audit.PROTOCOL_ID,
        "lineage": {"parent_act_run_id": act_run_id},
    }
    monkeypatch.setattr(
        audit,
        "load_confirmatory_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        audit,
        "_git_identity",
        lambda root: {
            "git_sha": audit.EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA,
            "dirty": False,
            "dirty_entry_count": 0,
        },
    )
    monkeypatch.setattr(
        audit,
        "resolve_act_parent_artifact",
        lambda root, run_id: (parent_artifact_id, tmp_path / "artifact"),
    )
    marker_path = tmp_path / audit.LEARN_REPRODUCTION_MARKER
    marker_path.parent.mkdir(parents=True)
    audit._write_json(
        marker_path,
        {
            "schema_version": audit.LEARN_REPRODUCTION_MARKER_SCHEMA,
            "evidence_mode": "reproduction",
            "original_raw_artifact_status": "unavailable",
            "protocol_id": audit.PROTOCOL_ID,
            "manifest_sha256": audit.EXPECTED_MANIFEST_SHA256,
            "measurement_git_sha": audit.EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA,
            "parent_act_run_id": act_run_id,
            "parent_artifact_id": parent_artifact_id,
            "claim_boundary": "reproduction_not_original_raw_evidence",
        },
    )

    source = audit.verify_learn_evidence_source(
        tmp_path,
        manifest,
        mode="reproduction",
    )

    assert source["mode"] == "reproduction"
    assert source["original_raw_artifact_used"] is False
    assert source["claim_boundary"] == (
        "reproduction_evidence_not_original_raw_artifact"
    )
    assert source["reproduction_marker"] == str(audit.LEARN_REPRODUCTION_MARKER)
    assert source["reproduction_parent_artifact_id"] == parent_artifact_id


def test_verify_learn_evidence_source_rejects_wrong_or_dirty_reproduction(
    monkeypatch,
    tmp_path,
):
    manifest = {"protocol_id": audit.PROTOCOL_ID}
    monkeypatch.setattr(
        audit,
        "load_confirmatory_manifest",
        lambda path: manifest,
    )

    monkeypatch.setattr(
        audit,
        "_git_identity",
        lambda root: {
            "git_sha": "0" * 40,
            "dirty": False,
            "dirty_entry_count": 0,
        },
    )
    with pytest.raises(RuntimeError, match="original measurement Git SHA"):
        audit.verify_learn_evidence_source(
            tmp_path,
            manifest,
            mode="reproduction",
        )

    monkeypatch.setattr(
        audit,
        "_git_identity",
        lambda root: {
            "git_sha": audit.EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA,
            "dirty": True,
            "dirty_entry_count": 1,
        },
    )
    with pytest.raises(RuntimeError, match="clean worktree"):
        audit.verify_learn_evidence_source(
            tmp_path,
            manifest,
            mode="reproduction",
        )



def test_validate_preserved_learn_outcome_identity_checks_operation_and_selected_families():
    add_outcomes = {
        1: "win",
        3: "loss",
        5: "win",
        7: "loss",
        9: "win",
        11: "loss",
        13: "win",
        15: "win",
        17: "loss",
        19: "loss",
        21: "tie",
        23: "loss",
        25: "loss",
        27: "tie",
        29: "tie",
    }
    subtract_outcomes = {
        2: "loss",
        4: "loss",
        6: "loss",
        8: "loss",
        10: "loss",
        12: "win",
        14: "loss",
        16: "loss",
        18: "loss",
        20: "loss",
        22: "tie",
        24: "loss",
        26: "loss",
        28: "loss",
        30: "tie",
    }
    outcomes = {**add_outcomes, **subtract_outcomes}
    families = []
    family_results = []
    for index in range(1, 31):
        family_id = f"learn-confirm-keyshift-{index:02d}"
        operation = "add" if index % 2 else "subtract"
        families.append(
            {
                "family_id": family_id,
                "correction_task": {"operation": operation},
            }
        )
        family_results.append(
            {
                "family_id": family_id,
                "outcome": outcomes[index],
            }
        )

    manifest: dict[str, object] = {
        "family_design": {"families": families},
    }
    result: dict[str, object] = {
        "family_results": family_results,
    }

    audit.validate_preserved_learn_outcome_identity(result, manifest)

    family_results[2]["outcome"] = "win"
    with pytest.raises(RuntimeError, match="selected-family outcome mismatch"):
        audit.validate_preserved_learn_outcome_identity(result, manifest)



def test_verify_learn_evidence_source_rejects_external_original_root(
    monkeypatch,
    tmp_path,
):
    manifest = {"protocol_id": audit.PROTOCOL_ID}
    monkeypatch.setattr(
        audit,
        "load_confirmatory_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        audit,
        "_git_identity",
        lambda root: {
            "git_sha": "f" * 40,
            "dirty": False,
            "dirty_entry_count": 0,
        },
    )

    with pytest.raises(RuntimeError, match="active Aster root"):
        audit.verify_learn_evidence_source(
            tmp_path / "evidence",
            manifest,
            mode="original",
            active_root=tmp_path / "active",
        )


def test_validate_confirmatory_unit_lineage_rejects_wrong_parent(tmp_path):
    family_id = "learn-confirm-keyshift-03"
    seed = 42
    expected_act_run_id = "5a8076571dca446fa07190cf4fc62509"
    expected_parent_artifact_id = "decision_model:" + ("a" * 64)
    correction_task = {
        "task_id": "correction",
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 2,
        "right": 3,
        "store_as": "a",
    }
    sibling_task = {
        "task_id": "sibling",
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 2,
        "right": 3,
        "store_as": "b",
    }
    manifest: dict[str, object] = {
        "family_design": {
            "families": [
                {
                    "family_id": family_id,
                    "correction_task": correction_task,
                    "sibling_task": sibling_task,
                }
            ]
        }
    }
    provenance = {
        "family_id": family_id,
        "seed": seed,
        "act_run_id": expected_act_run_id,
        "parent_artifact_id": expected_parent_artifact_id,
        "correction_task": correction_task,
        "uncorrected_sibling_task": sibling_task,
    }
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    audit._write_json(
        run_dir / "run.json",
        {"inputs": {"provenance": provenance}},
    )
    audit._write_json(
        run_dir / "correction-transfer.json",
        {"provenance": provenance},
    )
    units = {
        (family_id, seed): (
            run_dir,
            {"family_id": family_id, "seed": seed},
        )
    }

    audit.validate_confirmatory_unit_lineage(
        units,
        manifest,
        expected_act_run_id=expected_act_run_id,
        expected_parent_artifact_id=expected_parent_artifact_id,
    )

    bad = dict(provenance)
    bad["parent_artifact_id"] = "decision_model:" + ("b" * 64)
    audit._write_json(
        run_dir / "correction-transfer.json",
        {"provenance": bad},
    )
    with pytest.raises(RuntimeError, match="parent_artifact_id"):
        audit.validate_confirmatory_unit_lineage(
            units,
            manifest,
            expected_act_run_id=expected_act_run_id,
            expected_parent_artifact_id=expected_parent_artifact_id,
        )
