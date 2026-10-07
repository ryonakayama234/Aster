"""DIAG v3 Gate 2A frozen outcome screening tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import aster.training.diag_family_outcome_screening as screening
from aster.corpus.pipeline import digest, json_bytes
from aster.training.diag_family_features import (
    build_frozen_feature_schema,
    feature_schema_sha256,
)
from aster.training.diag_family_outcome_screening import (
    build_outcome_screening,
    run_outcome_screening,
)
from aster.training.diag_family_preoutcome import build_preoutcome_audit


def _feature_payload() -> tuple[dict[str, object], dict[str, object]]:
    schema = build_frozen_feature_schema()
    features = cast(list[dict[str, object]], schema["features"])
    rows: list[dict[str, object]] = []

    for family_index in range(1, 31):
        operation = "add" if family_index % 2 else "subtract"
        stratum_index = (family_index + 1) // 2
        signal = 1 if stratum_index <= 5 else 0
        row: dict[str, object] = {"family_id": f"family-{family_index:02d}"}

        for feature_index, feature in enumerate(features):
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
                    if output == "correction_target_margin_mean":
                        row[output] = signal
                    elif primitive == "left_digits":
                        row[output] = 2
                    elif primitive in {"left_operand", "right_operand", "result"}:
                        row[output] = stratum_index * 10
                    elif primitive == "absolute_operand_gap":
                        row[output] = (
                            stratum_index * 10 if operation == "subtract" else 1
                        )
                    else:
                        row[output] = (
                            (feature_index + 1) * stratum_index
                            + (0.125 if operation == "subtract" else 0.0)
                        )
                elif output.endswith("_counts"):
                    row[output] = {"x": 1}
                elif primitive == "top_competitor_type_matches_pair":
                    row[output] = bool(signal)
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


def _family_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for family_index in range(1, 31):
        stratum_index = (family_index + 1) // 2
        delta = 1.0 if stratum_index <= 5 else 0.0
        results.append(
            {
                "family_id": f"family-{family_index:02d}",
                "primary_accuracy_delta": delta,
                "primary_accuracy_delta_exact": "1/1" if delta else "0/1",
                "outcome": "win" if delta else "tie",
                "secondary_nll_effect_replay_minus_correction": delta / 10.0,
                "repair_accuracy": {
                    "parent": 0.25,
                    "replay": 0.50,
                    "correction": 0.75,
                },
            }
        )
    return results


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_outcome_screening_joins_after_preoutcome_and_emits_no_p_values():
    payload, schema = _feature_payload()
    preoutcome = build_preoutcome_audit(payload, schema)

    result = build_outcome_screening(
        payload,
        preoutcome,
        _family_results(),
        schema=schema,
    )

    assert result["status"] == "outcome_screening_complete"
    assert result["outcome_joined"] is True
    assert result["diag_v2_overlay_joined"] is False
    assert result["classification"] is None
    assert result["p_values_emitted"] is False

    continuous = cast(dict[str, dict[str, object]], result["continuous_screening"])
    margin = continuous["correction_target_margin_mean"]
    lead = cast(dict[str, object], margin["lead"])
    assert lead["lead_kind"] == "global_descriptive_lead"
    assert lead["qualifies_unconfounded_group"] is True
    assert margin["feature_group"] == "parent_geometry"

    left = continuous["left_operand"]
    assert left["design_order_confounded"] is True

    categorical = cast(dict[str, dict[str, object]], result["categorical_screening"])
    operation = categorical["operation"]
    categories = cast(list[dict[str, object]], operation["categories"])
    assert [item["family_count"] for item in categories] == [15, 15]

    serialized = json.dumps(result, sort_keys=True)
    assert "p_value" not in serialized
    assert "sign_test" not in serialized


def test_outcome_screening_rejects_family_order_mismatch():
    payload, schema = _feature_payload()
    preoutcome = build_preoutcome_audit(payload, schema)
    outcomes = _family_results()
    outcomes[0], outcomes[1] = outcomes[1], outcomes[0]

    with pytest.raises(RuntimeError, match="family/order identity"):
        build_outcome_screening(
            payload,
            preoutcome,
            outcomes,
            schema=schema,
        )


def test_outcome_screening_rejects_schema_drift():
    payload, schema = _feature_payload()
    preoutcome = build_preoutcome_audit(payload, schema)
    features = cast(list[dict[str, object]], schema["features"])
    features[0]["screening_eligible"] = True
    payload["feature_schema_sha256"] = feature_schema_sha256(schema)

    with pytest.raises(RuntimeError, match="persisted schema differs"):
        build_outcome_screening(
            payload,
            preoutcome,
            _family_results(),
            schema=schema,
        )


def test_run_outcome_screening_materializes_separate_gate_2a_run(
    monkeypatch,
    tmp_path,
):
    payload, schema = _feature_payload()
    preoutcome = build_preoutcome_audit(payload, schema)
    feature_run_id = "feature-run"
    preoutcome_run_id = "preoutcome-run"

    feature_path = tmp_path / "runs" / feature_run_id
    feature_path.mkdir(parents=True)
    _write_json(
        feature_path / "run.json",
        {
            "run_id": feature_run_id,
            "kind": "diag_family_feature_extraction",
            "status": "completed",
        },
    )
    _write_json(feature_path / "diag-family-features.json", payload)
    _write_json(feature_path / "diag-family-feature-schema.json", schema)

    preoutcome["source_feature_run_id"] = feature_run_id
    preoutcome["source_features_sha256"] = digest(json_bytes(payload))
    preoutcome_path = tmp_path / "runs" / preoutcome_run_id
    preoutcome_path.mkdir(parents=True)
    _write_json(
        preoutcome_path / "run.json",
        {
            "run_id": preoutcome_run_id,
            "kind": "diag_family_preoutcome_audit",
            "status": "completed",
        },
    )
    _write_json(
        preoutcome_path / "diag-family-preoutcome-audit.json",
        preoutcome,
    )

    manifest: dict[str, object] = {"protocol_id": "frozen"}
    monkeypatch.setattr(screening, "load_confirmatory_manifest", lambda path: manifest)
    monkeypatch.setattr(
        screening,
        "confirmatory_manifest_sha256",
        lambda value: "manifest",
    )
    monkeypatch.setattr(
        screening,
        "verify_learn_evidence_source",
        lambda root, current_manifest, **kwargs: {
            "mode": "reproduction",
            "original_raw_artifact_used": False,
        },
    )
    monkeypatch.setattr(
        screening,
        "reconstruct_confirmatory_evidence",
        lambda root, current_manifest: (
            {
                "family_results": _family_results(),
                "verdict": {"label": "Not supported"},
            },
            "campaign-run",
        ),
    )

    run_path = run_outcome_screening(
        tmp_path,
        preoutcome_run_id=preoutcome_run_id,
        learn_evidence_root=tmp_path / "evidence",
        learn_evidence_mode="reproduction",
    )

    assert run_path != feature_path
    assert run_path != preoutcome_path
    result = json.loads(
        (run_path / "diag-family-outcome-screening.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "outcome_screening_complete"
    assert result["learn_evidence_campaign_run_id"] == "campaign-run"
    assert result["diag_v2_overlay_joined"] is False
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert run["kind"] == "diag_family_outcome_screening"
    assert run["status"] == "completed"
    assert run["classification"] is None
