"""DIAG v3 pre-outcome confounding/collinearity materialization tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import aster.training.diag_family_preoutcome as preoutcome
from aster.training.diag_family_features import (
    build_frozen_feature_schema,
    feature_schema_sha256,
)
from aster.training.diag_family_preoutcome import (
    build_preoutcome_audit,
    run_preoutcome_audit,
)


def _feature_payload() -> tuple[dict[str, object], dict[str, object]]:
    schema = build_frozen_feature_schema()
    raw_features = cast(list[dict[str, object]], schema["features"])
    rows: list[dict[str, object]] = []

    for family_index in range(1, 31):
        operation = "add" if family_index % 2 else "subtract"
        stratum_index = (family_index + 1) // 2
        row: dict[str, object] = {"family_id": f"family-{family_index:02d}"}

        for feature_index, feature in enumerate(raw_features):
            kind = cast(str, feature["kind"])
            outputs = cast(list[str], feature["output_names"])
            primitive = cast(str, feature["name"])
            for output in outputs:
                if output == "family_index":
                    row[output] = family_index
                elif output == "stratum_index":
                    row[output] = stratum_index
                elif output == "operation":
                    row[output] = operation
                elif kind == "continuous":
                    if primitive == "left_digits":
                        row[output] = 2
                    elif primitive in {"left_operand", "right_operand", "result"}:
                        row[output] = stratum_index * 10
                    elif primitive == "absolute_operand_gap":
                        row[output] = (
                            stratum_index * 10 if operation == "subtract" else 1
                        )
                    else:
                        row[output] = (
                            stratum_index * (feature_index + 1)
                            + (0 if operation == "add" else 0.25)
                        )
                elif output.endswith("_counts"):
                    row[output] = {"x": 1}
                elif output.endswith("_mode"):
                    row[output] = "x"
                else:
                    row[output] = "x"

        rows.append(row)

    payload: dict[str, object] = {
        "schema_version": "aster-diag-family-feature-extraction-0",
        "feature_schema_sha256": feature_schema_sha256(schema),
        "manifest_sha256": "manifest",
        "parent_artifact_id": "decision_model:" + ("a" * 64),
        "families": rows,
        "outcome_columns_present": False,
    }
    return payload, schema


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_preoutcome_audit_is_outcome_blind_and_marks_design_confounding():
    payload, schema = _feature_payload()

    audit = build_preoutcome_audit(payload, schema)

    assert audit["status"] == "preoutcome_audit_complete"
    assert audit["outcome_joined"] is False
    assert audit["outcome_columns_present"] is False
    populations = cast(dict[str, dict[str, object]], audit["populations"])
    assert populations["all30"]["n"] == 30
    assert populations["add"]["n"] == 15
    assert populations["subtract"]["n"] == 15

    design = cast(dict[str, dict[str, object]], audit["design_order"])
    assert design["left_operand"]["design_order_confounded"] is True
    assert design["left_operand"]["confounded_operations"] == ["add", "subtract"]

    left_digits_add = cast(dict[str, object], design["left_digits"]["add"])
    left_digits_subtract = cast(dict[str, object], design["left_digits"]["subtract"])
    assert left_digits_add == {
        "rho": None,
        "degenerate": True,
        "n": 15,
        "left_distinct": 1,
        "right_distinct": 15,
    }
    assert left_digits_subtract["rho"] is None
    assert design["left_digits"]["design_order_confounded"] is False

    subtract_gap = cast(dict[str, object], design["absolute_operand_gap"]["subtract"])
    assert subtract_gap["rho"] == pytest.approx(1.0)
    assert design["absolute_operand_gap"]["design_order_confounded"] is True


def test_preoutcome_audit_rejects_outcome_joined_source():
    payload, schema = _feature_payload()
    payload["outcome_columns_present"] = True

    with pytest.raises(RuntimeError, match="outcome-free"):
        build_preoutcome_audit(payload, schema)


def test_preoutcome_audit_rejects_wrong_schema_hash():
    payload, schema = _feature_payload()
    payload["feature_schema_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="feature-schema identity"):
        build_preoutcome_audit(payload, schema)


def test_preoutcome_audit_rejects_persisted_schema_drift():
    payload, schema = _feature_payload()
    features = cast(list[dict[str, object]], schema["features"])
    features[0]["screening_eligible"] = True
    payload["feature_schema_sha256"] = feature_schema_sha256(schema)

    with pytest.raises(RuntimeError, match="differs from frozen schema"):
        build_preoutcome_audit(payload, schema)


def test_run_preoutcome_audit_materializes_separate_completed_run(monkeypatch, tmp_path):
    payload, schema = _feature_payload()
    source_id = "source-feature-run"
    source = tmp_path / "runs" / source_id
    source.mkdir(parents=True)

    _write_json(
        source / "run.json",
        {
            "schema_version": "aster-run-0",
            "run_id": source_id,
            "kind": "diag_family_feature_extraction",
            "status": "completed",
        },
    )
    extraction = {
        "status": "feature_extraction_complete",
        "feature_schema_sha256": payload["feature_schema_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "parent_artifact_id": payload["parent_artifact_id"],
        "outcome_joined": False,
    }
    _write_json(source / "diag-family-feature-extraction.json", extraction)
    _write_json(source / "diag-family-feature-schema.json", schema)
    _write_json(source / "diag-family-features.json", payload)

    rows = cast(list[dict[str, object]], payload["families"])
    families = [
        {
            "family_id": row["family_id"],
            "correction_task": {"operation": row["operation"]},
        }
        for row in rows
    ]
    manifest: dict[str, object] = {"family_design": {"families": families}}
    monkeypatch.setattr(preoutcome, "load_confirmatory_manifest", lambda path: manifest)
    monkeypatch.setattr(
        preoutcome,
        "confirmatory_manifest_sha256",
        lambda value: cast(str, payload["manifest_sha256"]),
    )

    run_path = run_preoutcome_audit(tmp_path, source_id)

    assert run_path != source
    result = json.loads(
        (run_path / "diag-family-preoutcome-audit.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "preoutcome_audit_complete"
    assert result["outcome_joined"] is False
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert run["kind"] == "diag_family_preoutcome_audit"
    assert run["status"] == "completed"
    assert run["source_feature_run_id"] == source_id



def test_validate_rows_against_manifest_rejects_family_order_mismatch():
    payload, _ = _feature_payload()
    rows = cast(list[dict[str, object]], payload["families"])
    families = [
        {
            "family_id": row["family_id"],
            "correction_task": {"operation": row["operation"]},
        }
        for row in rows
    ]
    manifest: dict[str, object] = {"family_design": {"families": families}}

    preoutcome._validate_rows_against_manifest(rows, manifest)

    families[0]["family_id"] = "different-family"
    with pytest.raises(RuntimeError, match="frozen family/order manifest"):
        preoutcome._validate_rows_against_manifest(rows, manifest)
