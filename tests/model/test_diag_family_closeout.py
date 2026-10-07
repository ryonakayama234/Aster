"""DIAG v3 Gate 2B closeout tests."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from aster.training.diag_family_audit import DIAG_V2_FAMILY_IDS
from aster.training.diag_family_closeout import (
    build_feature_audit,
    render_markdown_report,
)


def _lead(
    *,
    kind: str | None,
    confounded: bool,
    qualifies: bool,
    add: bool = False,
    subtract: bool = False,
) -> dict[str, object]:
    return {
        "lead_kind": kind,
        "global_descriptive_lead": kind == "global_descriptive_lead",
        "stratum_specific_lead": kind == "stratum_specific_lead",
        "add_stratum_lead": add,
        "subtract_stratum_lead": subtract,
        "bidirectional_stratum_heterogeneity": add and subtract,
        "design_order_confounded": confounded,
        "qualifies_unconfounded_group": qualifies,
    }


def _screening() -> dict[str, object]:
    selected = {
        "learn-confirm-keyshift-03": ("add", "loss", -0.5),
        "learn-confirm-keyshift-15": ("add", "win", 0.5),
        "learn-confirm-keyshift-02": ("subtract", "loss", -0.5),
        "learn-confirm-keyshift-12": ("subtract", "win", 0.5),
    }
    family_ids = list(selected)
    family_ids.extend(f"family-{index:02d}" for index in range(1, 27))

    rows: list[dict[str, object]] = []
    for index, family_id in enumerate(family_ids, 1):
        if family_id in selected:
            operation, outcome, delta = selected[family_id]
        else:
            operation = "add" if index % 2 else "subtract"
            outcome = "tie"
            delta = 0.0
        rows.append(
            {
                "family_id": family_id,
                "operation": operation,
                "outcome": outcome,
                "primary_accuracy_delta": delta,
                "repair_cr": delta / 2.0,
                "absolute_operand_gap": float(index),
                "sibling_target_margin_min": float(31 - index),
            }
        )

    return {
        "schema_version": "aster-diag-family-outcome-screening-0",
        "status": "outcome_screening_complete",
        "families": 30,
        "feature_schema_sha256": "schema",
        "manifest_sha256": "manifest",
        "parent_artifact_id": "decision_model:" + ("a" * 64),
        "source_feature_run_id": "feature-run",
        "source_preoutcome_run_id": "preoutcome-run",
        "learn_evidence_campaign_run_id": "learn-run",
        "learn_evidence_source": {"mode": "reproduction"},
        "outcome_joined": True,
        "diag_v2_overlay_joined": False,
        "classification": None,
        "causal_claim": False,
        "inferential_statistics_emitted": False,
        "joined_families": rows,
        "continuous_screening": {
            "absolute_operand_gap": {
                "feature_group": "task_state",
                "primitive_feature": "absolute_operand_gap",
                "all30": {"rho": 0.2},
                "add": {"rho": 0.1},
                "subtract": {"rho": 0.8},
                "design_order_confounded": True,
                "lead": _lead(
                    kind="stratum_specific_lead",
                    confounded=True,
                    qualifies=False,
                    subtract=True,
                ),
            },
            "sibling_target_margin_min": {
                "feature_group": "parent_geometry",
                "primitive_feature": "target_margin",
                "all30": {"rho": 0.6},
                "add": {"rho": 0.5},
                "subtract": {"rho": 0.5},
                "design_order_confounded": False,
                "lead": _lead(
                    kind="global_descriptive_lead",
                    confounded=False,
                    qualifies=True,
                ),
            },
        },
        "categorical_screening": {
            "operation": {
                "feature_group": "task_state",
                "primitive_feature": "operation",
                "categories": [],
            }
        },
        "collinearity_cluster_summaries": {},
    }


def _diag_v2_result(scale: float = 1.0) -> dict[str, object]:
    operation = {
        "learn-confirm-keyshift-03": "add",
        "learn-confirm-keyshift-15": "add",
        "learn-confirm-keyshift-02": "subtract",
        "learn-confirm-keyshift-12": "subtract",
    }
    stratum = {
        "learn-confirm-keyshift-03": "loss",
        "learn-confirm-keyshift-15": "control",
        "learn-confirm-keyshift-02": "loss",
        "learn-confirm-keyshift-12": "control",
    }
    summaries: list[dict[str, object]] = []
    for family_index, family_id in enumerate(DIAG_V2_FAMILY_IDS, 1):
        conditions = {}
        for condition_index, condition in enumerate(
            ("head-only", "last-block", "full"),
            1,
        ):
            value = scale * family_index * condition_index
            conditions[condition] = {
                "D_sibling_margin": value,
                "Repair_CR": -value / 10.0,
            }
        summaries.append(
            {
                "family_id": family_id,
                "family_stratum": stratum[family_id],
                "operation": operation[family_id],
                "conditions": conditions,
                "contrasts": {
                    "stability_gain_last_vs_full": -1.0,
                    "stability_cost_last_vs_head": 1.0,
                    "repair_gain_last_vs_head": -0.1,
                    "repair_gap_last_to_full": 0.1,
                },
            }
        )
    return {
        "schema_version": "aster-diag-update-depth-result-0",
        "status": "complete",
        "confirmatory_verdict": None,
        "family_summaries": summaries,
    }


def _build(
    screening: dict[str, object] | None = None,
    diag_v2_result: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_feature_audit(
        _screening() if screening is None else screening,
        _diag_v2_result() if diag_v2_result is None else diag_v2_result,
        source_outcome_screening_run_id="gate2a-run",
        source_outcome_screening_sha256="screening-sha",
        source_feature_summary_sha256="feature-sha",
        source_diag_v2_sha256="diag-v2-sha",
        no_update_invariant={
            "optimizer_constructed": False,
            "model_training": False,
            "all_parameters_requires_grad_false": True,
            "model_state_unchanged": True,
            "tokenizer_artifact_unchanged": True,
            "model_artifact_promoted": False,
        },
        closeout_source={
            "git_sha": "b" * 40,
            "dirty": False,
            "dirty_entry_count": 0,
        },
    )


def test_gate2b_classifies_from_gate2a_only_and_overlays_four_families():
    result = _build()

    assert result["status"] == "feature_audit_complete"
    assert result["classification"] == "parent_geometry_linked"
    assert result["diag_v2_overlay_joined"] is True
    assert result["diag_v2_overlay_can_override_classification"] is False
    assert result["causal_claim"] is False
    assert result["confirmatory_verdict"] is None
    assert result["inferential_statistics_emitted"] is False
    assert result["stop_rule_satisfied"] is True
    assert result["new_training_arm_introduced"] is False

    inputs = cast(dict[str, object], result["classification_inputs"])
    assert inputs["qualifying_unconfounded_groups"] == ["parent_geometry"]
    assert inputs["task_state_lead_exists"] is True
    assert inputs["all_task_state_leads_design_order_confounded"] is True

    pairs = cast(list[dict[str, object]], result["pair_inspection"])
    assert [
        (pair["loss_family_id"], pair["control_family_id"])
        for pair in pairs
    ] == [
        ("learn-confirm-keyshift-03", "learn-confirm-keyshift-15"),
        ("learn-confirm-keyshift-02", "learn-confirm-keyshift-12"),
    ]
    assert all(len(cast(list[object], pair["feature_comparison"])) == 3 for pair in pairs)


def test_diag_v2_overlay_values_cannot_change_classification():
    ordinary = _build(diag_v2_result=_diag_v2_result(scale=1.0))
    extreme = _build(diag_v2_result=_diag_v2_result(scale=1_000_000.0))

    assert ordinary["classification"] == "parent_geometry_linked"
    assert extreme["classification"] == ordinary["classification"]
    assert (
        extreme["classification_inputs"]
        == ordinary["classification_inputs"]
    )


def test_confounded_task_only_screening_closes_as_design_order_confounded():
    screening = _screening()
    continuous = cast(
        dict[str, dict[str, object]],
        screening["continuous_screening"],
    )
    continuous["sibling_target_margin_min"]["lead"] = _lead(
        kind=None,
        confounded=False,
        qualifies=False,
    )

    result = _build(screening=screening)

    assert result["classification"] == "design_order_confounded"
    inputs = cast(dict[str, object], result["classification_inputs"])
    assert inputs["qualifying_unconfounded_groups"] == []
    assert inputs["task_state_lead_exists"] is True
    assert inputs["all_task_state_leads_design_order_confounded"] is True


def test_mixed_unconfounded_groups_close_as_mixed():
    screening = deepcopy(_screening())
    continuous = cast(
        dict[str, dict[str, object]],
        screening["continuous_screening"],
    )
    continuous["absolute_operand_gap"]["design_order_confounded"] = False
    continuous["absolute_operand_gap"]["lead"] = _lead(
        kind="stratum_specific_lead",
        confounded=False,
        qualifies=True,
        subtract=True,
    )

    result = _build(screening=screening)

    assert result["classification"] == "mixed"
    inputs = cast(dict[str, object], result["classification_inputs"])
    assert inputs["qualifying_unconfounded_groups"] == [
        "parent_geometry",
        "task_state",
    ]


def test_markdown_report_keeps_descriptive_boundary_and_overlay_order():
    report = render_markdown_report(_build())

    assert "parent_geometry_linked" in report
    assert "30-family operational leads" in report
    assert "Four-family pair inspection" in report
    assert "DIAG v2 overlay" in report
    assert "causal_claim=false" in report
    assert "confirmatory_verdict=null" in report
    assert "Gate 2Aの30-family descriptive screeningだけから決定" in report
    assert "DIAG v2 overlayは別のcase-comparison evidence" in report
    pair_table = report.index("lead feature")
    pair_overlay = report.index("DIAG v2 overlay:", pair_table)
    assert pair_table < pair_overlay
