"""DIAG v3 Gate 2B: frozen DIAG v2 overlay and deterministic closeout."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import platform
import resource
import time
from typing import cast

from aster.corpus.pipeline import digest, json_bytes
from aster.records.runlog import RunLog
from aster.training.diag_family_audit import (
    DIAG_V2_FAMILY_IDS,
    EXPECTED_DIAG_V2_RUN_ID,
    _git_identity,
    verify_diag_v2_run_identity,
)
from aster.training.diag_family_features import classify_diagnostic
from aster.training.diag_family_outcome_screening import OUTCOME_SCREEN_SCHEMA_VERSION


FINAL_AUDIT_SCHEMA_VERSION = "aster-diag-family-feature-audit-0"
_OUTCOME_RUN_KIND = "diag_family_outcome_screening"
_FEATURE_RUN_KIND = "diag_family_feature_extraction"
_DIAG_V2_RESULT_FILE = "diag-update-depth-results.json"
_SELECTED_PAIRS = (
    ("add", "learn-confirm-keyshift-03", "learn-confirm-keyshift-15"),
    ("subtract", "learn-confirm-keyshift-02", "learn-confirm-keyshift-12"),
)
_CLASSIFICATION_NEXT_QUESTION = {
    "parent_geometry_linked": (
        "Does one controlled change to parent Decision geometry causally change "
        "Correction-vs-Replay sibling transfer while data, serializer, Tokenizer, "
        "and optimizer budget stay fixed?"
    ),
    "serializer_tokenization_linked": (
        "Does changing exactly one frozen serialization/tokenization geometry factor "
        "causally change Correction-vs-Replay sibling transfer while the parent model, "
        "examples, and optimizer budget stay fixed?"
    ),
    "candidate_action_linked": (
        "Does changing exactly one candidate/action geometry factor causally change "
        "Correction-vs-Replay sibling transfer while the parent model, serializer, "
        "Tokenizer, examples, and optimizer budget stay fixed?"
    ),
    "task_state_linked": (
        "Does varying the implicated task/state feature in a matched family design that "
        "breaks the old ordinal coupling causally change Correction-vs-Replay sibling transfer?"
    ),
    "design_order_confounded": (
        "When family order and operand magnitude are decoupled in a matched family design, "
        "does the apparent task/state association with Correction-vs-Replay sibling transfer persist?"
    ),
    "mixed": (
        "Which single discriminating factor, changed alone while the other implicated "
        "feature groups are fixed, causally changes Correction-vs-Replay sibling transfer?"
    ),
    "no_clear_static_explanation": (
        "In a 2x2 embedding-update x block-update Trial that adds the missing embeddings+head "
        "condition, which update component causally changes Correction-vs-Replay sibling transfer?"
    ),
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"DIAG v3 Gate 2B field {key!r} must be an object")
    return cast(dict[str, object], value)


def _require_list(data: Mapping[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"DIAG v3 Gate 2B field {key!r} must be a list")
    return cast(list[object], value)


def _require_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"DIAG v3 Gate 2B field {key!r} must be a non-empty string")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"DIAG v3 Gate 2B numeric value is invalid: {field}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"DIAG v3 Gate 2B numeric value is non-finite: {field}")
    return number


def _contains_forbidden_inference_key(value: object) -> bool:
    forbidden = {"p_value", "sign_test", "verdict"}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in forbidden:
                return True
            if _contains_forbidden_inference_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_inference_key(item) for item in value)
    return False


def _load_outcome_screening(
    root: Path,
    run_id: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    run_dir = root / "runs" / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("DIAG v3 Gate 2B outcome-screening Run is unavailable")

    run = _read_json(run_dir / "run.json")
    if (
        run.get("schema_version") != "aster-run-0"
        or run.get("run_id") != run_id
        or run.get("kind") != _OUTCOME_RUN_KIND
        or run.get("status") != "completed"
    ):
        raise RuntimeError("DIAG v3 Gate 2B requires a completed Gate 2A Run")

    result_file = run_dir / "diag-family-outcome-screening.json"
    if not result_file.is_file() or result_file.is_symlink():
        raise RuntimeError("DIAG v3 Gate 2B Gate 2A artifact is missing")
    screening = _read_json(result_file)
    if screening.get("schema_version") != OUTCOME_SCREEN_SCHEMA_VERSION:
        raise RuntimeError("DIAG v3 Gate 2B Gate 2A schema mismatch")
    if screening.get("status") != "outcome_screening_complete":
        raise RuntimeError("DIAG v3 Gate 2B source screening is incomplete")
    if screening.get("families") != 30:
        raise RuntimeError("DIAG v3 Gate 2B requires exactly 30 screened families")
    if screening.get("outcome_joined") is not True:
        raise RuntimeError("DIAG v3 Gate 2B source has not joined frozen outcomes")
    if screening.get("diag_v2_overlay_joined") is not False:
        raise RuntimeError("DIAG v3 Gate 2B source already joined DIAG v2 overlay")
    if screening.get("classification") is not None:
        raise RuntimeError("DIAG v3 Gate 2B source already contains a classification")
    if screening.get("causal_claim") is not False:
        raise RuntimeError("DIAG v3 Gate 2B source causal-claim boundary changed")
    if screening.get("inferential_statistics_emitted") is not False:
        raise RuntimeError("DIAG v3 Gate 2B source emitted inferential statistics")
    if _contains_forbidden_inference_key(screening):
        raise RuntimeError("DIAG v3 Gate 2B source leaked inferential-statistic fields")

    inputs = _require_dict(run, "inputs")
    required_identity = {
        "source_preoutcome_run_id": screening.get("source_preoutcome_run_id"),
        "source_feature_run_id": screening.get("source_feature_run_id"),
        "learn_evidence_campaign_run_id": screening.get("learn_evidence_campaign_run_id"),
        "outcome_joined": True,
        "diag_v2_overlay_joined": False,
        "causal_claim": False,
    }
    for key, expected in required_identity.items():
        if inputs.get(key) != expected:
            raise RuntimeError(
                f"DIAG v3 Gate 2B Gate 2A frozen identity mismatch for {key}"
            )

    return run, screening, digest(json_bytes(screening))


def _load_feature_summary(
    root: Path,
    screening: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    feature_run_id = _require_str(screening, "source_feature_run_id")
    feature_dir = root / "runs" / feature_run_id
    if feature_dir.is_symlink() or not feature_dir.is_dir():
        raise RuntimeError("DIAG v3 Gate 2B source feature Run is unavailable")

    run = _read_json(feature_dir / "run.json")
    if (
        run.get("schema_version") != "aster-run-0"
        or run.get("run_id") != feature_run_id
        or run.get("kind") != _FEATURE_RUN_KIND
        or run.get("status") != "completed"
    ):
        raise RuntimeError("DIAG v3 Gate 2B source feature Run identity mismatch")

    summary = _read_json(feature_dir / "diag-family-feature-extraction.json")
    if summary.get("status") != "feature_extraction_complete":
        raise RuntimeError("DIAG v3 Gate 2B source feature extraction is incomplete")
    if summary.get("feature_schema_sha256") != screening.get("feature_schema_sha256"):
        raise RuntimeError("DIAG v3 Gate 2B feature schema identity mismatch")
    if summary.get("manifest_sha256") != screening.get("manifest_sha256"):
        raise RuntimeError("DIAG v3 Gate 2B manifest identity mismatch")
    if summary.get("parent_artifact_id") != screening.get("parent_artifact_id"):
        raise RuntimeError("DIAG v3 Gate 2B parent artifact identity mismatch")
    if summary.get("outcome_joined") is not False or summary.get("causal_claim") is not False:
        raise RuntimeError("DIAG v3 Gate 2B source feature extraction boundary changed")

    invariant = _require_dict(summary, "no_update_invariant")
    expected_invariant = {
        "optimizer_constructed": False,
        "model_training": False,
        "all_parameters_requires_grad_false": True,
        "model_state_unchanged": True,
        "tokenizer_artifact_unchanged": True,
        "model_artifact_promoted": False,
    }
    for key, expected in expected_invariant.items():
        if invariant.get(key) is not expected:
            raise RuntimeError(
                f"DIAG v3 Gate 2B no-update invariant mismatch for {key}"
            )
    return summary, digest(json_bytes(summary))


def _load_diag_v2_result(root: Path) -> tuple[dict[str, object], str]:
    identity = verify_diag_v2_run_identity(root)
    if identity.get("run_id") != EXPECTED_DIAG_V2_RUN_ID:
        raise RuntimeError("DIAG v3 Gate 2B DIAG v2 Run identity mismatch")
    result_path = root / "runs" / EXPECTED_DIAG_V2_RUN_ID / _DIAG_V2_RESULT_FILE
    result = _read_json(result_path)
    if result.get("confirmatory_verdict") is not None:
        raise RuntimeError("DIAG v3 Gate 2B DIAG v2 result must remain diagnostic")
    return result, digest(json_bytes(result))


def _family_rows_by_id(screening: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = _require_list(screening, "joined_families")
    if len(rows) != 30 or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("DIAG v3 Gate 2B joined family table must contain 30 rows")
    indexed: dict[str, dict[str, object]] = {}
    for raw in rows:
        row = cast(dict[str, object], raw)
        family_id = _require_str(row, "family_id")
        if family_id in indexed:
            raise RuntimeError(f"DIAG v3 Gate 2B duplicate family row: {family_id}")
        indexed[family_id] = row
    for family_id in DIAG_V2_FAMILY_IDS:
        if family_id not in indexed:
            raise RuntimeError(f"DIAG v3 Gate 2B selected family is missing: {family_id}")
    return indexed


def _diag_v2_by_family(result: Mapping[str, object]) -> dict[str, dict[str, object]]:
    summaries = _require_list(result, "family_summaries")
    if len(summaries) != len(DIAG_V2_FAMILY_IDS):
        raise RuntimeError("DIAG v3 Gate 2B DIAG v2 family-summary count mismatch")
    indexed: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for raw in summaries:
        if not isinstance(raw, dict):
            raise RuntimeError("DIAG v3 Gate 2B DIAG v2 family summary is invalid")
        summary = cast(dict[str, object], raw)
        family_id = _require_str(summary, "family_id")
        if family_id in indexed:
            raise RuntimeError(f"DIAG v3 Gate 2B duplicate DIAG v2 family: {family_id}")
        indexed[family_id] = summary
        order.append(family_id)
    if order != list(DIAG_V2_FAMILY_IDS):
        raise RuntimeError("DIAG v3 Gate 2B DIAG v2 family order mismatch")
    return indexed


def _classification_inputs(screening: Mapping[str, object]) -> dict[str, object]:
    continuous = _require_dict(screening, "continuous_screening")
    qualifying_groups: set[str] = set()
    add_stratum_groups: set[str] = set()
    subtract_stratum_groups: set[str] = set()
    task_state_leads: list[dict[str, object]] = []
    lead_columns: list[str] = []

    for name in sorted(continuous):
        raw_record = continuous[name]
        if not isinstance(raw_record, dict):
            raise RuntimeError("DIAG v3 Gate 2B continuous screening record is invalid")
        record = cast(dict[str, object], raw_record)
        group = _require_str(record, "feature_group")
        lead = _require_dict(record, "lead")
        lead_kind = lead.get("lead_kind")
        if lead_kind is None:
            continue
        if lead_kind not in {"global_descriptive_lead", "stratum_specific_lead"}:
            raise RuntimeError(f"DIAG v3 Gate 2B unknown lead kind: {lead_kind}")
        lead_columns.append(name)
        if group == "task_state":
            task_state_leads.append(lead)

        if lead.get("qualifies_unconfounded_group") is True:
            qualifying_groups.add(group)
            if lead.get("add_stratum_lead") is True:
                add_stratum_groups.add(group)
            if lead.get("subtract_stratum_lead") is True:
                subtract_stratum_groups.add(group)

    task_state_lead_exists = bool(task_state_leads)
    all_task_state_leads_design_order_confounded = (
        task_state_lead_exists
        and all(lead.get("design_order_confounded") is True for lead in task_state_leads)
    )

    return {
        "lead_columns": lead_columns,
        "qualifying_unconfounded_groups": sorted(qualifying_groups),
        "add_stratum_groups": sorted(add_stratum_groups),
        "subtract_stratum_groups": sorted(subtract_stratum_groups),
        "task_state_lead_exists": task_state_lead_exists,
        "all_task_state_leads_design_order_confounded": (
            all_task_state_leads_design_order_confounded
        ),
    }


def _feature_comparison(
    left: Mapping[str, object],
    right: Mapping[str, object],
    screening: Mapping[str, object],
) -> list[dict[str, object]]:
    continuous = _require_dict(screening, "continuous_screening")
    categorical = _require_dict(screening, "categorical_screening")
    rows: list[dict[str, object]] = []

    for name in sorted(continuous):
        record = cast(dict[str, object], continuous[name])
        left_value = _finite_number(left.get(name), field=f"{name}.left")
        right_value = _finite_number(right.get(name), field=f"{name}.right")
        rows.append(
            {
                "feature_name": name,
                "feature_group": _require_str(record, "feature_group"),
                "kind": "continuous",
                "loss_family_value": left_value,
                "control_family_value": right_value,
                "control_minus_loss": right_value - left_value,
                "lead": record.get("lead"),
                "design_order_confounded": record.get("design_order_confounded"),
            }
        )

    for name in sorted(categorical):
        record = cast(dict[str, object], categorical[name])
        rows.append(
            {
                "feature_name": name,
                "feature_group": _require_str(record, "feature_group"),
                "kind": "categorical",
                "loss_family_value": left.get(name),
                "control_family_value": right.get(name),
                "control_minus_loss": None,
            }
        )
    return rows


def build_feature_audit(
    screening: Mapping[str, object],
    diag_v2_result: Mapping[str, object],
    *,
    source_outcome_screening_run_id: str,
    source_outcome_screening_sha256: str,
    source_feature_summary_sha256: str,
    source_diag_v2_sha256: str,
    no_update_invariant: Mapping[str, object],
    closeout_source: Mapping[str, object],
) -> dict[str, object]:
    """Build final read-only DIAG v3 closeout from frozen Gate 2A + DIAG v2."""
    if screening.get("status") != "outcome_screening_complete":
        raise RuntimeError("DIAG v3 Gate 2B requires completed Gate 2A screening")
    if screening.get("classification") is not None:
        raise RuntimeError("DIAG v3 Gate 2B must classify exactly once from null")
    if screening.get("diag_v2_overlay_joined") is not False:
        raise RuntimeError("DIAG v3 Gate 2B source overlay boundary changed")

    family_rows = _family_rows_by_id(screening)
    diag_rows = _diag_v2_by_family(diag_v2_result)
    classification_inputs = _classification_inputs(screening)
    classification = classify_diagnostic(
        cast(list[str], classification_inputs["qualifying_unconfounded_groups"]),
        task_state_lead_exists=cast(
            bool, classification_inputs["task_state_lead_exists"]
        ),
        all_task_state_leads_design_order_confounded=cast(
            bool,
            classification_inputs["all_task_state_leads_design_order_confounded"],
        ),
        add_stratum_groups=cast(
            list[str], classification_inputs["add_stratum_groups"]
        ),
        subtract_stratum_groups=cast(
            list[str], classification_inputs["subtract_stratum_groups"]
        ),
    )

    pair_inspection: list[dict[str, object]] = []
    for operation, loss_family_id, control_family_id in _SELECTED_PAIRS:
        loss_row = family_rows[loss_family_id]
        control_row = family_rows[control_family_id]
        if loss_row.get("operation") != operation or control_row.get("operation") != operation:
            raise RuntimeError("DIAG v3 Gate 2B selected-pair operation mismatch")
        loss_overlay = diag_rows[loss_family_id]
        control_overlay = diag_rows[control_family_id]
        if (
            loss_overlay.get("operation") != operation
            or control_overlay.get("operation") != operation
        ):
            raise RuntimeError("DIAG v3 Gate 2B DIAG v2 pair operation mismatch")

        pair_inspection.append(
            {
                "operation": operation,
                "loss_family_id": loss_family_id,
                "control_family_id": control_family_id,
                "feature_comparison": _feature_comparison(
                    loss_row, control_row, screening
                ),
                "learn_outcome_context": {
                    "loss_family": {
                        "outcome": loss_row.get("outcome"),
                        "primary_accuracy_delta": loss_row.get("primary_accuracy_delta"),
                        "repair_cr": loss_row.get("repair_cr"),
                    },
                    "control_family": {
                        "outcome": control_row.get("outcome"),
                        "primary_accuracy_delta": control_row.get("primary_accuracy_delta"),
                        "repair_cr": control_row.get("repair_cr"),
                    },
                },
                "diag_v2_overlay": {
                    "loss_family": loss_overlay,
                    "control_family": control_overlay,
                },
            }
        )

    return {
        "schema_version": FINAL_AUDIT_SCHEMA_VERSION,
        "status": "feature_audit_complete",
        "source_outcome_screening_run_id": source_outcome_screening_run_id,
        "source_outcome_screening_sha256": source_outcome_screening_sha256,
        "source_feature_run_id": screening.get("source_feature_run_id"),
        "source_feature_summary_sha256": source_feature_summary_sha256,
        "source_preoutcome_run_id": screening.get("source_preoutcome_run_id"),
        "source_diag_v2_run_id": EXPECTED_DIAG_V2_RUN_ID,
        "source_diag_v2_sha256": source_diag_v2_sha256,
        "feature_schema_sha256": screening.get("feature_schema_sha256"),
        "manifest_sha256": screening.get("manifest_sha256"),
        "parent_artifact_id": screening.get("parent_artifact_id"),
        "learn_evidence_source": screening.get("learn_evidence_source"),
        "learn_evidence_campaign_run_id": screening.get("learn_evidence_campaign_run_id"),
        "closeout_source": dict(closeout_source),
        "families": 30,
        "outcome_joined": True,
        "diag_v2_overlay_joined": True,
        "classification": classification,
        "classification_inputs": classification_inputs,
        "classification_inputs_source": "persisted_gate2a_screening_only",
        "diag_v2_overlay_can_override_classification": False,
        "causal_claim": False,
        "confirmatory_verdict": None,
        "inferential_statistics_emitted": False,
        "no_update_invariant": dict(no_update_invariant),
        "joined_families": screening.get("joined_families"),
        "continuous_screening": screening.get("continuous_screening"),
        "categorical_screening": screening.get("categorical_screening"),
        "collinearity_cluster_summaries": screening.get(
            "collinearity_cluster_summaries"
        ),
        "pair_inspection": pair_inspection,
        "next_causal_question": _CLASSIFICATION_NEXT_QUESTION[classification],
        "stop_rule_satisfied": True,
        "new_training_arm_introduced": False,
    }


def _lead_rows_for_report(audit: Mapping[str, object]) -> list[dict[str, object]]:
    continuous = _require_dict(audit, "continuous_screening")
    inputs = _require_dict(audit, "classification_inputs")
    raw_names = inputs.get("lead_columns")
    if not isinstance(raw_names, list) or not all(
        isinstance(name, str) for name in raw_names
    ):
        raise RuntimeError("DIAG v3 Gate 2B lead-column list is invalid")

    rows: list[dict[str, object]] = []
    for name in cast(list[str], raw_names):
        record = cast(dict[str, object], continuous[name])
        lead = _require_dict(record, "lead")
        rows.append(
            {
                "name": name,
                "group": record.get("feature_group"),
                "rho_all30": _require_dict(record, "all30").get("rho"),
                "rho_add": _require_dict(record, "add").get("rho"),
                "rho_subtract": _require_dict(record, "subtract").get("rho"),
                "lead_kind": lead.get("lead_kind"),
                "design_order_confounded": record.get("design_order_confounded"),
            }
        )
    return rows


def render_markdown_report(audit: Mapping[str, object]) -> str:
    """Render compact human-readable closeout; JSON remains canonical."""
    classification = _require_str(audit, "classification")
    lines = [
        "# DIAG v3 — Family/State Feature Audit closeout",
        "",
        "## 結論",
        "",
        f"- diagnostic classification: `{classification}`",
        "- `causal_claim=false`",
        "- `confirmatory_verdict=null`",
        "- p-value / confirmatory retest: なし",
        "- new training arm: なし",
        "",
        "この分類は保存済み30-family descriptive screeningと、固定済み4-family DIAG v2 overlayを",
        "組み合わせたpost-hoc diagnostic classificationであり、因果効果の証明ではない。",
        "",
        "## 30-family operational leads",
        "",
        "| feature | group | rho all30 | rho add | rho subtract | lead | design-order confounded |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    def fmt(value: object) -> str:
        return "null" if value is None else f"{float(cast(float, value)):.6g}"

    lead_rows = _lead_rows_for_report(audit)
    for row in lead_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"`{row['group']}`",
                    fmt(row["rho_all30"]),
                    fmt(row["rho_add"]),
                    fmt(row["rho_subtract"]),
                    f"`{row['lead_kind']}`",
                    str(row["design_order_confounded"]).lower(),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Four-family pair inspection",
            "",
            "Machine-readable JSONには全screening featureのside-by-side比較を保存する。",
            "ここではoperational leadのみを要約し、その後にDIAG v2 overlayを示す。",
            "",
        ]
    )
    lead_names = {cast(str, row["name"]) for row in lead_rows}

    for raw_pair in _require_list(audit, "pair_inspection"):
        pair = cast(dict[str, object], raw_pair)
        lines.extend(
            [
                f"### {pair['loss_family_id']} vs {pair['control_family_id']} ({pair['operation']})",
                "",
                "| lead feature | loss | control | control-loss |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        raw_features = pair.get("feature_comparison")
        if not isinstance(raw_features, list):
            raise RuntimeError("DIAG v3 Gate 2B pair feature comparison is invalid")
        for raw_feature in raw_features:
            if not isinstance(raw_feature, dict):
                raise RuntimeError("DIAG v3 Gate 2B pair feature row is invalid")
            feature = cast(dict[str, object], raw_feature)
            if feature.get("feature_name") not in lead_names:
                continue
            delta = feature.get("control_minus_loss")
            if delta is None:
                continue
            lines.append(
                f"| `{feature['feature_name']}` | "
                f"{_finite_number(feature.get('loss_family_value'), field='loss'):.6g} | "
                f"{_finite_number(feature.get('control_family_value'), field='control'):.6g} | "
                f"{_finite_number(delta, field='control_minus_loss'):.6g} |"
            )

        overlay = _require_dict(pair, "diag_v2_overlay")
        lines.extend(
            [
                "",
                "DIAG v2 overlay:",
                "",
                "| family | condition | D sibling margin | Repair_CR |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for side in ("loss_family", "control_family"):
            family = _require_dict(overlay, side)
            family_id = _require_str(family, "family_id")
            conditions = _require_dict(family, "conditions")
            for condition in ("head-only", "last-block", "full"):
                values = _require_dict(conditions, condition)
                lines.append(
                    f"| `{family_id}` | `{condition}` | "
                    f"{_finite_number(values.get('D_sibling_margin'), field='D'):.6g} | "
                    f"{_finite_number(values.get('Repair_CR'), field='Repair_CR'):.6g} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Next causal question",
            "",
            str(audit["next_causal_question"]),
            "",
            "## Provenance boundary",
            "",
            f"- Gate 2A Run: `{audit['source_outcome_screening_run_id']}`",
            f"- DIAG v2 Run: `{audit['source_diag_v2_run_id']}`",
            f"- feature schema SHA-256: `{audit['feature_schema_sha256']}`",
            f"- parent artifact: `{audit['parent_artifact_id']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_feature_audit_closeout(
    root: str | Path,
    *,
    outcome_screening_run_id: str,
    write_reports: bool = True,
) -> Path:
    """Materialize Gate 2B and close DIAG v3 without new model updates."""
    started = time.perf_counter()
    root_path = Path(root).resolve()
    _, screening, screening_sha = _load_outcome_screening(
        root_path, outcome_screening_run_id
    )
    feature_summary, feature_summary_sha = _load_feature_summary(root_path, screening)
    diag_v2_result, diag_v2_sha = _load_diag_v2_result(root_path)
    source = _git_identity(root_path)
    no_update_invariant = _require_dict(feature_summary, "no_update_invariant")

    audit = build_feature_audit(
        screening,
        diag_v2_result,
        source_outcome_screening_run_id=outcome_screening_run_id,
        source_outcome_screening_sha256=screening_sha,
        source_feature_summary_sha256=feature_summary_sha,
        source_diag_v2_sha256=diag_v2_sha,
        no_update_invariant=no_update_invariant,
        closeout_source=source,
    )
    audit["resources"] = {
        "wall_seconds_before_write": time.perf_counter() - started,
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "python": platform.python_version(),
    }
    if _contains_forbidden_inference_key(audit):
        raise RuntimeError("DIAG v3 Gate 2B closeout leaked inferential-statistic fields")

    run = RunLog(
        root_path,
        "diag_family_feature_audit",
        {
            "schema_version": FINAL_AUDIT_SCHEMA_VERSION,
            "source_outcome_screening_run_id": outcome_screening_run_id,
            "source_outcome_screening_sha256": screening_sha,
            "source_feature_run_id": screening.get("source_feature_run_id"),
            "source_feature_summary_sha256": feature_summary_sha,
            "source_diag_v2_run_id": EXPECTED_DIAG_V2_RUN_ID,
            "source_diag_v2_sha256": diag_v2_sha,
            "classification": audit["classification"],
            "causal_claim": False,
            "confirmatory_verdict": None,
            "outcome_joined": True,
            "diag_v2_overlay_joined": True,
            "new_training_arm_introduced": False,
        },
        producer="diagnostic_audit",
    )
    try:
        _write_json(run.path / "diag-family-feature-audit.json", audit)
        report_paths: list[str] = []
        if write_reports:
            reports = root_path / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            json_path = reports / "diag-family-feature-audit-v3.json"
            md_path = reports / "diag-family-feature-audit-v3.md"
            _write_json(json_path, audit)
            md_path.write_text(render_markdown_report(audit), encoding="utf-8")
            report_paths = [
                str(json_path.relative_to(root_path)),
                str(md_path.relative_to(root_path)),
            ]

        run.finish(
            "completed",
            result="diag-family-feature-audit.json",
            classification=audit["classification"],
            diag_v2_overlay_joined=True,
            causal_claim=False,
            confirmatory_verdict=None,
            reports=report_paths,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
