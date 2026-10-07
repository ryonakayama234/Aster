"""Outcome-blind confounding and collinearity materialization for DIAG v3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import cast

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.training.diag_family_features import (
    COLLINEARITY_RHO_THRESHOLD,
    DESIGN_ORDER_RHO_THRESHOLD,
    build_frozen_feature_schema,
    collinearity_clusters,
    feature_schema_sha256,
    spearman_rho,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
)


PREOUTCOME_SCHEMA_VERSION = "aster-diag-family-preoutcome-audit-0"
_SOURCE_KIND = "diag_family_feature_extraction"
_FORBIDDEN_OUTCOME_COLUMNS = {
    "outcome",
    "primary_accuracy_delta",
    "primary_accuracy_delta_exact",
    "secondary_nll_effect_replay_minus_correction",
    "repair_accuracy",
    "repair_cr",
    "win",
    "loss",
    "tie",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _continuous_screening_columns(
    schema: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    raw_features = schema.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("DIAG v3 feature schema is missing features")

    columns: dict[str, dict[str, str]] = {}
    for raw in raw_features:
        if not isinstance(raw, dict):
            raise ValueError("DIAG v3 feature schema contains an invalid feature")
        feature = cast(dict[str, object], raw)
        if feature.get("kind") != "continuous" or feature.get("screening_eligible") is not True:
            continue
        group = feature.get("group")
        primitive = feature.get("name")
        outputs = feature.get("output_names")
        if not isinstance(group, str) or not isinstance(primitive, str):
            raise ValueError("DIAG v3 continuous feature ownership is invalid")
        if not isinstance(outputs, list) or not all(isinstance(name, str) for name in outputs):
            raise ValueError("DIAG v3 continuous feature outputs are invalid")
        for output_name in cast(list[str], outputs):
            if output_name in columns:
                raise ValueError(f"Duplicate screening column: {output_name}")
            columns[output_name] = {
                "feature_group": group,
                "primitive_feature": primitive,
            }
    return columns


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"DIAG v3 numeric feature is not numeric: {field}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"DIAG v3 numeric feature is not finite: {field}")
    return numeric


def _rho_record(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    rho = spearman_rho(left, right)
    return {
        "rho": rho,
        "degenerate": rho is None,
        "n": len(left),
        "left_distinct": len(set(left)),
        "right_distinct": len(set(right)),
    }


def _population_audit(
    rows: Sequence[dict[str, object]],
    columns: Mapping[str, dict[str, str]],
) -> dict[str, object]:
    values = {
        name: [_finite_number(row[name], field=name) for row in rows]
        for name in sorted(columns)
    }
    stratum = [_finite_number(row["stratum_index"], field="stratum_index") for row in rows]

    feature_statistics = {
        name: {
            "feature_group": columns[name]["feature_group"],
            "primitive_feature": columns[name]["primitive_feature"],
            "distinct_values": len(set(feature_values)),
            "vs_stratum_index": _rho_record(feature_values, stratum),
        }
        for name, feature_values in values.items()
    }

    pairwise: list[dict[str, object]] = []
    names = sorted(values)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left_name,
                    "right": right_name,
                    **_rho_record(values[left_name], values[right_name]),
                }
            )

    return {
        "n": len(rows),
        "continuous_screening_columns": len(columns),
        "feature_statistics": feature_statistics,
        "pairwise_spearman": pairwise,
        "collinearity_clusters": [
            list(cluster) for cluster in collinearity_clusters(values)
        ],
    }


def build_preoutcome_audit(
    feature_payload: Mapping[str, object],
    schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the frozen pre-outcome audit without reading any family outcomes."""
    canonical_schema = build_frozen_feature_schema()
    canonical_schema_sha = feature_schema_sha256(canonical_schema)
    frozen_schema = (
        canonical_schema
        if schema is None
        else cast(dict[str, object], dict(schema))
    )
    schema_sha = feature_schema_sha256(frozen_schema)
    if schema_sha != canonical_schema_sha:
        raise RuntimeError(
            "DIAG v3 pre-outcome persisted feature schema differs from frozen schema"
        )
    if feature_payload.get("feature_schema_sha256") != canonical_schema_sha:
        raise RuntimeError("DIAG v3 pre-outcome feature-schema identity mismatch")
    if feature_payload.get("outcome_columns_present") is not False:
        raise RuntimeError("DIAG v3 pre-outcome source must be outcome-free")

    raw_rows = feature_payload.get("families")
    if not isinstance(raw_rows, list) or len(raw_rows) != 30:
        raise RuntimeError("DIAG v3 pre-outcome audit requires exactly 30 family rows")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise RuntimeError("DIAG v3 pre-outcome family row is invalid")
        rows.append(cast(dict[str, object], raw))

    raw_outputs = frozen_schema.get("output_names")
    if not isinstance(raw_outputs, list) or not all(isinstance(name, str) for name in raw_outputs):
        raise RuntimeError("DIAG v3 frozen output schema is invalid")
    expected_keys = set(cast(list[str], raw_outputs)) | {"family_id"}
    family_ids: set[str] = set()
    for row in rows:
        if set(row) != expected_keys:
            raise RuntimeError("DIAG v3 pre-outcome row differs from frozen feature schema")
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise RuntimeError("DIAG v3 pre-outcome family_id is invalid")
        if family_id in family_ids:
            raise RuntimeError(f"Duplicate DIAG v3 family row: {family_id}")
        family_ids.add(family_id)
        if set(row) & _FORBIDDEN_OUTCOME_COLUMNS:
            raise RuntimeError("DIAG v3 pre-outcome source contains an outcome column")

    operation_counts = Counter(row.get("operation") for row in rows)
    if operation_counts != Counter({"add": 15, "subtract": 15}):
        raise RuntimeError("DIAG v3 pre-outcome operation strata must be 15 add / 15 subtract")
    for operation in ("add", "subtract"):
        operation_rows = [row for row in rows if row["operation"] == operation]
        indices = sorted(
            _finite_number(row["stratum_index"], field="stratum_index")
            for row in operation_rows
        )
        if indices != [float(index) for index in range(1, 16)]:
            raise RuntimeError(
                f"DIAG v3 {operation} stratum_index must be exactly 1..15"
            )

    columns = _continuous_screening_columns(frozen_schema)
    populations = {
        "all30": rows,
        "add": [row for row in rows if row["operation"] == "add"],
        "subtract": [row for row in rows if row["operation"] == "subtract"],
    }
    population_audits = {
        name: _population_audit(population_rows, columns)
        for name, population_rows in populations.items()
    }

    design_order: dict[str, object] = {}
    add_stats = cast(
        dict[str, object],
        population_audits["add"]["feature_statistics"],
    )
    subtract_stats = cast(
        dict[str, object],
        population_audits["subtract"]["feature_statistics"],
    )
    for name in sorted(columns):
        add_record = cast(dict[str, object], add_stats[name])["vs_stratum_index"]
        subtract_record = cast(dict[str, object], subtract_stats[name])["vs_stratum_index"]
        add_rho = cast(dict[str, object], add_record)["rho"]
        subtract_rho = cast(dict[str, object], subtract_record)["rho"]
        confounded_operations = []
        if isinstance(add_rho, (int, float)) and abs(float(add_rho)) >= DESIGN_ORDER_RHO_THRESHOLD:
            confounded_operations.append("add")
        if isinstance(subtract_rho, (int, float)) and abs(float(subtract_rho)) >= DESIGN_ORDER_RHO_THRESHOLD:
            confounded_operations.append("subtract")
        design_order[name] = {
            "feature_group": columns[name]["feature_group"],
            "primitive_feature": columns[name]["primitive_feature"],
            "add": add_record,
            "subtract": subtract_record,
            "design_order_confounded": bool(confounded_operations),
            "confounded_operations": confounded_operations,
        }

    return {
        "schema_version": PREOUTCOME_SCHEMA_VERSION,
        "status": "preoutcome_audit_complete",
        "feature_schema_sha256": canonical_schema_sha,
        "manifest_sha256": feature_payload.get("manifest_sha256"),
        "parent_artifact_id": feature_payload.get("parent_artifact_id"),
        "families": 30,
        "outcome_joined": False,
        "outcome_columns_present": False,
        "causal_claim": False,
        "thresholds": {
            "design_order_abs_spearman": DESIGN_ORDER_RHO_THRESHOLD,
            "collinearity_abs_spearman": COLLINEARITY_RHO_THRESHOLD,
        },
        "populations": population_audits,
        "design_order": design_order,
        "next_gate": "join frozen family outcomes only after this artifact is persisted",
    }


def _validate_rows_against_manifest(
    rows: Sequence[dict[str, object]],
    manifest: Mapping[str, object],
) -> None:
    design = manifest.get("family_design")
    if not isinstance(design, dict):
        raise RuntimeError("DIAG v3 frozen manifest is missing family_design")
    raw_families = cast(dict[str, object], design).get("families")
    if not isinstance(raw_families, list) or len(raw_families) != 30:
        raise RuntimeError("DIAG v3 frozen manifest must contain exactly 30 families")

    expected: list[tuple[str, str, int]] = []
    counters = {"add": 0, "subtract": 0}
    for raw in raw_families:
        if not isinstance(raw, dict):
            raise RuntimeError("DIAG v3 frozen family definition is invalid")
        family = cast(dict[str, object], raw)
        family_id = family.get("family_id")
        correction = family.get("correction_task")
        if not isinstance(family_id, str) or not isinstance(correction, dict):
            raise RuntimeError("DIAG v3 frozen family identity is invalid")
        operation = cast(dict[str, object], correction).get("operation")
        if operation not in {"add", "subtract"}:
            raise RuntimeError("DIAG v3 frozen family operation is invalid")
        operation_name = cast(str, operation)
        counters[operation_name] += 1
        expected.append((family_id, operation_name, counters[operation_name]))

    actual = [
        (
            cast(str, row["family_id"]),
            cast(str, row["operation"]),
            _finite_number(row["stratum_index"], field="stratum_index"),
        )
        for row in rows
    ]
    expected_normalized = [
        (family_id, operation, float(stratum))
        for family_id, operation, stratum in expected
    ]
    if actual != expected_normalized:
        raise RuntimeError(
            "DIAG v3 feature rows differ from frozen family/order manifest"
        )


def run_preoutcome_audit(root: str | Path, feature_run_id: str) -> Path:
    """Materialize the outcome-blind Gate 1.5 audit from one completed feature Run."""
    root_path = Path(root).resolve()
    source_path = root_path / "runs" / feature_run_id
    if not source_path.is_dir() or source_path.is_symlink():
        raise RuntimeError("DIAG v3 source feature Run directory is unavailable")

    source_run = _read_json(source_path / "run.json")
    if source_run.get("run_id") != feature_run_id:
        raise RuntimeError("DIAG v3 source feature Run ID mismatch")
    if source_run.get("kind") != _SOURCE_KIND or source_run.get("status") != "completed":
        raise RuntimeError("DIAG v3 source feature Run must be completed feature extraction")

    extraction = _read_json(source_path / "diag-family-feature-extraction.json")
    if extraction.get("status") != "feature_extraction_complete":
        raise RuntimeError("DIAG v3 source feature extraction is incomplete")
    if extraction.get("outcome_joined") is not False:
        raise RuntimeError("DIAG v3 source feature extraction already joined outcomes")

    schema = _read_json(source_path / "diag-family-feature-schema.json")
    feature_payload = _read_json(source_path / "diag-family-features.json")
    audit = build_preoutcome_audit(feature_payload, schema)

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    manifest_sha = confirmatory_manifest_sha256(manifest)
    if manifest_sha != audit["manifest_sha256"]:
        raise RuntimeError("DIAG v3 source feature manifest identity mismatch")
    raw_rows = feature_payload.get("families")
    if not isinstance(raw_rows, list):
        raise RuntimeError("DIAG v3 source feature rows are unavailable")
    _validate_rows_against_manifest(
        [cast(dict[str, object], row) for row in raw_rows],
        manifest,
    )
    source_features_sha256 = digest(json_bytes(feature_payload))

    if extraction.get("feature_schema_sha256") != audit["feature_schema_sha256"]:
        raise RuntimeError("DIAG v3 source extraction/schema hash mismatch")
    if extraction.get("manifest_sha256") != audit["manifest_sha256"]:
        raise RuntimeError("DIAG v3 source extraction/manifest mismatch")
    if extraction.get("parent_artifact_id") != audit["parent_artifact_id"]:
        raise RuntimeError("DIAG v3 source extraction/parent artifact mismatch")

    run = RunLog(
        root_path,
        "diag_family_preoutcome_audit",
        {
            "schema_version": PREOUTCOME_SCHEMA_VERSION,
            "source_feature_run_id": feature_run_id,
            "feature_schema_sha256": audit["feature_schema_sha256"],
            "manifest_sha256": audit["manifest_sha256"],
            "parent_artifact_id": audit["parent_artifact_id"],
            "source_features_sha256": source_features_sha256,
            "outcome_joined": False,
            "causal_claim": False,
        },
        producer="diagnostic_audit",
    )
    try:
        audit["source_feature_run_id"] = feature_run_id
        audit["source_features_sha256"] = source_features_sha256
        _write_json(run.path / "diag-family-preoutcome-audit.json", audit)
        run.finish(
            "completed",
            result="diag-family-preoutcome-audit.json",
            source_feature_run_id=feature_run_id,
            outcome_joined=False,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
