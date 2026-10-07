"""Frozen outcome join and 30-family descriptive screening for DIAG v3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import cast

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.training.diag_family_audit import (
    reconstruct_confirmatory_evidence,
    verify_learn_evidence_source,
)
from aster.training.diag_family_features import (
    build_frozen_feature_schema,
    feature_schema_sha256,
    operational_lead,
    spearman_rho,
)
from aster.training.diag_family_preoutcome import (
    PREOUTCOME_SCHEMA_VERSION,
    _continuous_screening_columns,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
)


OUTCOME_SCREEN_SCHEMA_VERSION = "aster-diag-family-outcome-screening-0"
_PREOUTCOME_RUN_KIND = "diag_family_preoutcome_audit"
_FEATURE_RUN_KIND = "diag_family_feature_extraction"
_OUTCOMES = ("win", "loss", "tie")


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


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"DIAG v3 numeric value is invalid: {field}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"DIAG v3 numeric value is non-finite: {field}")
    return number


def _require_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"DIAG v3 field {key!r} must be an object")
    return cast(dict[str, object], value)


def _require_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"DIAG v3 field {key!r} must be a non-empty string")
    return value


def _categorical_screening_columns(
    schema: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    raw_features = schema.get("features")
    if not isinstance(raw_features, list):
        raise RuntimeError("DIAG v3 feature schema is missing features")

    columns: dict[str, dict[str, str]] = {}
    for raw in raw_features:
        if not isinstance(raw, dict):
            raise RuntimeError("DIAG v3 feature schema contains an invalid feature")
        feature = cast(dict[str, object], raw)
        if feature.get("kind") != "categorical" or feature.get("screening_eligible") is not True:
            continue
        group = feature.get("group")
        primitive = feature.get("name")
        scope = feature.get("scope")
        outputs = feature.get("output_names")
        if (
            not isinstance(group, str)
            or not isinstance(primitive, str)
            or not isinstance(scope, str)
            or not isinstance(outputs, list)
            or not all(isinstance(name, str) for name in outputs)
        ):
            raise RuntimeError("DIAG v3 categorical feature schema is invalid")

        output_names = cast(list[str], outputs)
        if scope == "family_categorical":
            selected = output_names
        else:
            selected = [name for name in output_names if name.endswith("_mode")]
        if not selected:
            raise RuntimeError(
                f"DIAG v3 categorical feature has no deterministic mode output: {primitive}"
            )
        for output_name in selected:
            if output_name in columns:
                raise RuntimeError(f"Duplicate categorical screening column: {output_name}")
            columns[output_name] = {
                "feature_group": group,
                "primitive_feature": primitive,
            }
    return columns


def _rho_summary(rows: Sequence[dict[str, object]], feature_name: str) -> dict[str, object]:
    values = [_finite_number(row[feature_name], field=feature_name) for row in rows]
    outcomes = [
        _finite_number(row["primary_accuracy_delta"], field="primary_accuracy_delta")
        for row in rows
    ]
    rho = spearman_rho(values, outcomes)
    return {
        "n": len(rows),
        "distinct_feature_values": len(set(values)),
        "rho": rho,
        "degenerate": rho is None,
    }


def _canonical_category(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _categorical_summary(
    rows: Sequence[dict[str, object]],
    column_name: str,
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for row in rows:
        value = row[column_name]
        key = _canonical_category(value)
        bucket = buckets.setdefault(
            key,
            {
                "category": value,
                "family_ids": [],
                "primary_deltas": [],
                "outcomes": [],
            },
        )
        cast(list[str], bucket["family_ids"]).append(_require_str(row, "family_id"))
        cast(list[float], bucket["primary_deltas"]).append(
            _finite_number(
                row["primary_accuracy_delta"],
                field="primary_accuracy_delta",
            )
        )
        outcome = _require_str(row, "outcome")
        if outcome not in _OUTCOMES:
            raise RuntimeError(f"Unexpected DIAG v3 family outcome: {outcome}")
        cast(list[str], bucket["outcomes"]).append(outcome)

    summaries: list[dict[str, object]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        deltas = cast(list[float], bucket["primary_deltas"])
        outcomes = cast(list[str], bucket["outcomes"])
        counts = Counter(outcomes)
        summaries.append(
            {
                "category": bucket["category"],
                "family_count": len(deltas),
                "family_ids": cast(list[str], bucket["family_ids"]),
                "median_primary_accuracy_delta": float(median(deltas)),
                "outcome_counts": {
                    outcome: counts[outcome] for outcome in _OUTCOMES
                },
            }
        )
    return summaries


def _outcome_rows(
    feature_rows: Sequence[dict[str, object]],
    family_results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    if len(feature_rows) != 30 or len(family_results) != 30:
        raise RuntimeError("DIAG v3 Gate 2A requires exactly 30 feature/outcome rows")

    outcome_by_family: dict[str, dict[str, object]] = {}
    outcome_order: list[str] = []
    for raw in family_results:
        family_id = _require_str(raw, "family_id")
        if family_id in outcome_by_family:
            raise RuntimeError(f"Duplicate DIAG v3 outcome family: {family_id}")
        outcome_by_family[family_id] = raw
        outcome_order.append(family_id)

    feature_order = [_require_str(row, "family_id") for row in feature_rows]
    if feature_order != outcome_order:
        raise RuntimeError(
            "DIAG v3 feature rows and frozen outcome rows differ in family/order identity"
        )

    joined: list[dict[str, object]] = []
    for feature_row in feature_rows:
        family_id = _require_str(feature_row, "family_id")
        result = outcome_by_family[family_id]
        outcome = _require_str(result, "outcome")
        if outcome not in _OUTCOMES:
            raise RuntimeError(f"Unexpected DIAG v3 family outcome: {outcome}")
        repair = _require_dict(result, "repair_accuracy")
        parent_repair = _finite_number(
            repair.get("parent"),
            field="repair_accuracy.parent",
        )
        replay_repair = _finite_number(
            repair.get("replay"),
            field="repair_accuracy.replay",
        )
        correction_repair = _finite_number(
            repair.get("correction"),
            field="repair_accuracy.correction",
        )

        row = dict(feature_row)
        row.update(
            {
                "primary_accuracy_delta": _finite_number(
                    result.get("primary_accuracy_delta"),
                    field="primary_accuracy_delta",
                ),
                "primary_accuracy_delta_exact": _require_str(
                    result,
                    "primary_accuracy_delta_exact",
                ),
                "outcome": outcome,
                "secondary_nll_effect_replay_minus_correction": _finite_number(
                    result.get("secondary_nll_effect_replay_minus_correction"),
                    field="secondary_nll_effect_replay_minus_correction",
                ),
                "repair_parent_accuracy": parent_repair,
                "repair_replay_accuracy": replay_repair,
                "repair_correction_accuracy": correction_repair,
                "repair_cr": correction_repair - replay_repair,
            }
        )
        joined.append(row)
    return joined


def _cluster_memberships(
    preoutcome: Mapping[str, object],
    continuous_columns: Sequence[str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[list[str]]]]:
    raw_populations = preoutcome.get("populations")
    if not isinstance(raw_populations, dict):
        raise RuntimeError("DIAG v3 pre-outcome populations are missing")

    memberships: dict[str, dict[str, str]] = {
        name: {} for name in continuous_columns
    }
    clusters_by_population: dict[str, list[list[str]]] = {}
    expected = set(continuous_columns)

    for population in ("all30", "add", "subtract"):
        raw_population = cast(dict[str, object], raw_populations).get(population)
        if not isinstance(raw_population, dict):
            raise RuntimeError(f"DIAG v3 pre-outcome population is missing: {population}")
        raw_clusters = cast(dict[str, object], raw_population).get(
            "collinearity_clusters"
        )
        if not isinstance(raw_clusters, list):
            raise RuntimeError(
                f"DIAG v3 pre-outcome collinearity clusters are missing: {population}"
            )

        seen: set[str] = set()
        clusters: list[list[str]] = []
        for index, raw_cluster in enumerate(raw_clusters, 1):
            if (
                not isinstance(raw_cluster, list)
                or not raw_cluster
                or not all(isinstance(name, str) for name in raw_cluster)
            ):
                raise RuntimeError("DIAG v3 pre-outcome collinearity cluster is invalid")
            cluster = cast(list[str], raw_cluster)
            cluster_id = f"{population}-cluster-{index:03d}"
            for name in cluster:
                if name not in expected or name in seen:
                    raise RuntimeError(
                        "DIAG v3 pre-outcome collinearity membership mismatch"
                    )
                seen.add(name)
                memberships[name][population] = cluster_id
            clusters.append(cluster)
        if seen != expected:
            raise RuntimeError(
                f"DIAG v3 {population} clusters do not cover all screening columns"
            )
        clusters_by_population[population] = clusters

    return memberships, clusters_by_population


def build_outcome_screening(
    feature_payload: Mapping[str, object],
    preoutcome: Mapping[str, object],
    family_results: Sequence[dict[str, object]],
    *,
    schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Join frozen outcomes and compute the 30-family descriptive screen."""
    canonical_schema = build_frozen_feature_schema()
    canonical_sha = feature_schema_sha256(canonical_schema)
    frozen_schema = canonical_schema if schema is None else dict(schema)
    if feature_schema_sha256(cast(dict[str, object], frozen_schema)) != canonical_sha:
        raise RuntimeError("DIAG v3 Gate 2A persisted schema differs from frozen schema")
    if feature_payload.get("feature_schema_sha256") != canonical_sha:
        raise RuntimeError("DIAG v3 Gate 2A feature-schema identity mismatch")
    if preoutcome.get("feature_schema_sha256") != canonical_sha:
        raise RuntimeError("DIAG v3 Gate 2A pre-outcome schema identity mismatch")
    if preoutcome.get("status") != "preoutcome_audit_complete":
        raise RuntimeError("DIAG v3 Gate 2A requires a completed pre-outcome audit")
    if preoutcome.get("outcome_joined") is not False:
        raise RuntimeError("DIAG v3 Gate 2A pre-outcome audit already joined outcomes")

    raw_feature_rows = feature_payload.get("families")
    if not isinstance(raw_feature_rows, list) or not all(
        isinstance(row, dict) for row in raw_feature_rows
    ):
        raise RuntimeError("DIAG v3 Gate 2A feature rows are unavailable")
    feature_rows = [
        cast(dict[str, object], row) for row in cast(list[object], raw_feature_rows)
    ]
    joined = _outcome_rows(feature_rows, family_results)

    continuous_columns = _continuous_screening_columns(frozen_schema)
    categorical_columns = _categorical_screening_columns(frozen_schema)
    memberships, clusters_by_population = _cluster_memberships(
        preoutcome,
        sorted(continuous_columns),
    )
    raw_design = preoutcome.get("design_order")
    if not isinstance(raw_design, dict):
        raise RuntimeError("DIAG v3 pre-outcome design-order audit is missing")
    design_order = cast(dict[str, object], raw_design)

    add_rows = [row for row in joined if row.get("operation") == "add"]
    subtract_rows = [row for row in joined if row.get("operation") == "subtract"]
    if len(add_rows) != 15 or len(subtract_rows) != 15:
        raise RuntimeError("DIAG v3 Gate 2A operation strata must remain 15 / 15")

    continuous_screening: dict[str, object] = {}
    for name in sorted(continuous_columns):
        all30 = _rho_summary(joined, name)
        add = _rho_summary(add_rows, name)
        subtract = _rho_summary(subtract_rows, name)
        design_record = design_order.get(name)
        if not isinstance(design_record, dict):
            raise RuntimeError(
                f"DIAG v3 pre-outcome design-order record is missing: {name}"
            )
        confounded = design_record.get("design_order_confounded")
        if type(confounded) is not bool:
            raise RuntimeError("DIAG v3 design-order confounding flag is invalid")

        rho_all30 = cast(float | None, all30["rho"])
        rho_add = cast(float | None, add["rho"])
        rho_subtract = cast(float | None, subtract["rho"])
        lead = operational_lead(
            rho_all30=rho_all30,
            rho_add=rho_add,
            rho_subtract=rho_subtract,
            design_order_confounded_flag=cast(bool, confounded),
        )
        same_sign: bool | None = None
        if rho_add is not None and rho_subtract is not None:
            same_sign = rho_add * rho_subtract > 0

        continuous_screening[name] = {
            "feature_group": continuous_columns[name]["feature_group"],
            "primitive_feature": continuous_columns[name]["primitive_feature"],
            "all30": all30,
            "add": add,
            "subtract": subtract,
            "same_sign_across_strata": same_sign,
            "design_order_confounded": confounded,
            "confounded_operations": design_record.get("confounded_operations"),
            "collinearity_membership": memberships[name],
            "lead": lead,
            "value_pairs": [
                {
                    "family_id": _require_str(row, "family_id"),
                    "operation": _require_str(row, "operation"),
                    "feature_value": _finite_number(row[name], field=name),
                    "primary_accuracy_delta": _finite_number(
                        row["primary_accuracy_delta"],
                        field="primary_accuracy_delta",
                    ),
                }
                for row in joined
            ],
        }

    cluster_summaries: dict[str, object] = {}
    for population, clusters in clusters_by_population.items():
        summaries: list[dict[str, object]] = []
        for index, members in enumerate(clusters, 1):
            qualifying = [
                name
                for name in members
                if cast(dict[str, object], continuous_screening[name])["lead"]
                and cast(
                    dict[str, object],
                    cast(dict[str, object], continuous_screening[name])["lead"],
                ).get("lead_kind")
                is not None
            ]
            unconfounded = [
                name
                for name in qualifying
                if cast(
                    dict[str, object],
                    cast(dict[str, object], continuous_screening[name])["lead"],
                ).get("qualifies_unconfounded_group")
                is True
            ]
            groups = sorted(
                {
                    continuous_columns[name]["feature_group"]
                    for name in qualifying
                }
            )
            unconfounded_groups = sorted(
                {
                    continuous_columns[name]["feature_group"]
                    for name in unconfounded
                }
            )
            lead_kinds = sorted(
                {
                    cast(
                        str,
                        cast(
                            dict[str, object],
                            cast(dict[str, object], continuous_screening[name])["lead"],
                        )["lead_kind"],
                    )
                    for name in qualifying
                }
            )
            summaries.append(
                {
                    "cluster_id": f"{population}-cluster-{index:03d}",
                    "members": members,
                    "feature_groups": sorted(
                        {continuous_columns[name]["feature_group"] for name in members}
                    ),
                    "qualifying_members": qualifying,
                    "qualifying_unconfounded_members": unconfounded,
                    "qualifying_groups": groups,
                    "qualifying_unconfounded_groups": unconfounded_groups,
                    "lead_kinds": lead_kinds,
                    "design_order_confounded_members": [
                        name
                        for name in qualifying
                        if cast(
                            dict[str, object],
                            continuous_screening[name],
                        )["design_order_confounded"]
                        is True
                    ],
                    "cross_group_cluster": len(
                        {continuous_columns[name]["feature_group"] for name in members}
                    )
                    > 1,
                }
            )
        cluster_summaries[population] = summaries

    categorical_screening = {
        name: {
            "feature_group": categorical_columns[name]["feature_group"],
            "primitive_feature": categorical_columns[name]["primitive_feature"],
            "categories": _categorical_summary(joined, name),
        }
        for name in sorted(categorical_columns)
    }

    return {
        "schema_version": OUTCOME_SCREEN_SCHEMA_VERSION,
        "status": "outcome_screening_complete",
        "families": 30,
        "feature_schema_sha256": canonical_sha,
        "manifest_sha256": feature_payload.get("manifest_sha256"),
        "parent_artifact_id": feature_payload.get("parent_artifact_id"),
        "outcome_joined": True,
        "diag_v2_overlay_joined": False,
        "classification": None,
        "causal_claim": False,
        "p_values_emitted": False,
        "joined_families": joined,
        "continuous_screening": continuous_screening,
        "categorical_screening": categorical_screening,
        "collinearity_cluster_summaries": cluster_summaries,
        "next_gate": (
            "persisted 30-family screening may now receive DIAG v2 four-family overlay "
            "and exactly-one classification"
        ),
    }


def run_outcome_screening(
    root: str | Path,
    *,
    preoutcome_run_id: str,
    learn_evidence_root: str | Path | None = None,
    learn_evidence_mode: str = "original",
) -> Path:
    """Materialize Gate 2A from one persisted Gate 1.5 Run and frozen LEARN evidence."""
    root_path = Path(root).resolve()
    evidence_root = (
        root_path
        if learn_evidence_root is None
        else Path(learn_evidence_root).resolve()
    )

    preoutcome_path = root_path / "runs" / preoutcome_run_id
    if not preoutcome_path.is_dir() or preoutcome_path.is_symlink():
        raise RuntimeError("DIAG v3 Gate 2A pre-outcome Run is unavailable")
    preoutcome_run = _read_json(preoutcome_path / "run.json")
    if (
        preoutcome_run.get("run_id") != preoutcome_run_id
        or preoutcome_run.get("kind") != _PREOUTCOME_RUN_KIND
        or preoutcome_run.get("status") != "completed"
    ):
        raise RuntimeError("DIAG v3 Gate 2A requires a completed pre-outcome Run")

    preoutcome = _read_json(
        preoutcome_path / "diag-family-preoutcome-audit.json"
    )
    if preoutcome.get("schema_version") != PREOUTCOME_SCHEMA_VERSION:
        raise RuntimeError("DIAG v3 Gate 2A pre-outcome schema mismatch")
    if preoutcome.get("outcome_joined") is not False:
        raise RuntimeError("DIAG v3 Gate 2A source already joined outcomes")
    preoutcome_sha256 = digest(json_bytes(preoutcome))

    feature_run_id = _require_str(preoutcome, "source_feature_run_id")
    feature_path = root_path / "runs" / feature_run_id
    feature_run = _read_json(feature_path / "run.json")
    if (
        feature_run.get("run_id") != feature_run_id
        or feature_run.get("kind") != _FEATURE_RUN_KIND
        or feature_run.get("status") != "completed"
    ):
        raise RuntimeError("DIAG v3 Gate 2A source feature Run is invalid")
    feature_payload = _read_json(feature_path / "diag-family-features.json")
    source_features_sha256 = digest(json_bytes(feature_payload))
    if preoutcome.get("source_features_sha256") != source_features_sha256:
        raise RuntimeError("DIAG v3 Gate 2A feature payload digest mismatch")
    schema = _read_json(feature_path / "diag-family-feature-schema.json")

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    manifest_sha = confirmatory_manifest_sha256(manifest)
    if feature_payload.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("DIAG v3 Gate 2A frozen manifest identity mismatch")

    evidence_source = verify_learn_evidence_source(
        evidence_root,
        manifest,
        mode=learn_evidence_mode,
        active_root=root_path,
    )
    confirmatory, campaign_run_id = reconstruct_confirmatory_evidence(
        evidence_root,
        manifest,
    )
    raw_family_results = confirmatory.get("family_results")
    if not isinstance(raw_family_results, list) or not all(
        isinstance(row, dict) for row in raw_family_results
    ):
        raise RuntimeError("DIAG v3 Gate 2A frozen family outcomes are unavailable")
    family_results = [
        cast(dict[str, object], row)
        for row in cast(list[object], raw_family_results)
    ]

    result = build_outcome_screening(
        feature_payload,
        preoutcome,
        family_results,
        schema=schema,
    )
    result.update(
        {
            "source_preoutcome_run_id": preoutcome_run_id,
            "source_preoutcome_sha256": preoutcome_sha256,
            "source_feature_run_id": feature_run_id,
            "source_features_sha256": source_features_sha256,
            "learn_evidence_source": evidence_source,
            "learn_evidence_campaign_run_id": campaign_run_id,
            "learn_confirmatory_verdict": _require_dict(
                confirmatory,
                "verdict",
            ).get("label"),
        }
    )

    run = RunLog(
        root_path,
        "diag_family_outcome_screening",
        {
            "schema_version": OUTCOME_SCREEN_SCHEMA_VERSION,
            "source_preoutcome_run_id": preoutcome_run_id,
            "source_preoutcome_sha256": preoutcome_sha256,
            "source_feature_run_id": feature_run_id,
            "source_features_sha256": source_features_sha256,
            "learn_evidence_mode": learn_evidence_mode,
            "learn_evidence_campaign_run_id": campaign_run_id,
            "outcome_joined": True,
            "diag_v2_overlay_joined": False,
            "causal_claim": False,
        },
        producer="diagnostic_audit",
    )
    try:
        _write_json(run.path / "diag-family-outcome-screening.json", result)
        run.finish(
            "completed",
            result="diag-family-outcome-screening.json",
            source_preoutcome_run_id=preoutcome_run_id,
            outcome_joined=True,
            diag_v2_overlay_joined=False,
            classification=None,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
