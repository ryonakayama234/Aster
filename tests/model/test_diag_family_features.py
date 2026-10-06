"""Pure DIAG v3 feature-audit contract tests."""

import math

import pytest

from aster.training.diag_family_features import (
    FEATURE_GROUPS,
    average_ranks,
    build_frozen_feature_schema,
    classify_diagnostic,
    collinearity_clusters,
    design_order_confounded,
    feature_schema_sha256,
    levenshtein_distance,
    longest_common_prefix_length,
    longest_common_prefix_ratio,
    numeric_summary,
    operational_lead,
    spearman_rho,
)


def test_frozen_schema_has_unique_output_columns_and_single_owners():
    schema = build_frozen_feature_schema()
    features = schema["features"]
    assert isinstance(features, list)

    names = [feature["name"] for feature in features]
    assert len(names) == len(set(names))

    output_names = schema["output_names"]
    assert isinstance(output_names, list)
    assert len(output_names) == len(set(output_names))

    by_name = {feature["name"]: feature for feature in features}
    assert by_name["candidate_set_size"]["group"] == "candidate_action"
    assert by_name["target_candidate_token_length"]["group"] == "candidate_action"
    assert by_name["encoded_input_token_length"]["group"] == "serializer_tokenization"
    assert by_name["family_index"]["screening_eligible"] is False
    assert by_name["stratum_index"]["screening_eligible"] is False
    assert {feature["group"] for feature in features} == set(FEATURE_GROUPS)


def test_feature_schema_hash_is_deterministic_and_content_addressed():
    first = build_frozen_feature_schema()
    second = build_frozen_feature_schema()
    assert feature_schema_sha256(first) == feature_schema_sha256(second)

    changed = build_frozen_feature_schema()
    changed["aggregation"]["mode_tie"] = "different"
    assert feature_schema_sha256(changed) != feature_schema_sha256(first)

    changed_handling = build_frozen_feature_schema()
    changed_handling["missing_degenerate_handling"]["missing_primitive_measurement"] = "null"
    assert feature_schema_sha256(changed_handling) != feature_schema_sha256(first)


def test_frozen_schema_records_missing_and_degenerate_handling():
    handling = build_frozen_feature_schema()["missing_degenerate_handling"]
    assert handling["undefined_spearman"] == {"rho": None, "degenerate": True}
    assert handling["missing_primitive_measurement"] == "reject"
    assert handling["nonfinite_numeric_measurement"] == "reject"
    assert handling["empty_numeric_aggregation"] == "reject"
    assert handling["empty_categorical_aggregation"] == "reject"


def test_average_ranks_use_average_for_ties():
    assert average_ranks([10, 10, 30, 20]) == [1.5, 1.5, 4.0, 3.0]


def test_spearman_matches_wolfram_tied_rank_reference():
    rho = spearman_rho([1, 1, 2, 3], [10, 20, 30, 40])
    assert rho == pytest.approx(3 / math.sqrt(10))


def test_spearman_returns_none_for_degenerate_feature():
    assert spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman_rho([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_fixed_numeric_and_token_geometry_helpers():
    assert numeric_summary([3, 1, 5]) == {"min": 1.0, "mean": 3.0, "max": 5.0}
    with pytest.raises(ValueError, match="At least one numeric feature value is required"):
        numeric_summary([])
    assert levenshtein_distance([1, 2, 3], [1, 4, 3, 5]) == 2
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert longest_common_prefix_ratio([1, 2, 3, 4], [1, 2, 9]) == pytest.approx(0.5)
    assert longest_common_prefix_ratio([], []) == 1.0


def test_design_order_confounding_uses_preregistered_threshold():
    order = [1, 2, 3, 4, 5]
    assert design_order_confounded([10, 20, 30, 40, 50], order)
    assert not design_order_confounded([10, 30, 20, 50, 40], order)


def test_collinearity_clusters_are_connected_and_deterministic():
    clusters = collinearity_clusters(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [5, 1, 4, 2, 3],
            "constant": [7, 7, 7, 7, 7],
        }
    )
    assert clusters == [("a", "b"), ("c",), ("constant",)]


def test_operational_lead_global_rule():
    result = operational_lead(
        rho_all30=0.55,
        rho_add=0.40,
        rho_subtract=0.30,
        design_order_confounded_flag=False,
    )
    assert result["lead_kind"] == "global_descriptive_lead"
    assert result["qualifies_unconfounded_group"] is True


def test_operational_lead_stratum_rule_and_bidirectional_case():
    weak_other = operational_lead(
        rho_all30=0.10,
        rho_add=0.60,
        rho_subtract=0.10,
        design_order_confounded_flag=False,
    )
    assert weak_other["lead_kind"] == "stratum_specific_lead"
    assert weak_other["add_stratum_lead"] is True

    opposite = operational_lead(
        rho_all30=0.0,
        rho_add=0.70,
        rho_subtract=-0.60,
        design_order_confounded_flag=False,
    )
    assert opposite["lead_kind"] == "stratum_specific_lead"
    assert opposite["bidirectional_stratum_heterogeneity"] is True


def test_operational_lead_same_sign_midrange_other_is_not_stratum_specific():
    result = operational_lead(
        rho_all30=0.20,
        rho_add=0.60,
        rho_subtract=0.30,
        design_order_confounded_flag=False,
    )
    assert result["lead_kind"] is None


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("parent_geometry", "parent_geometry_linked"),
        ("serializer_tokenization", "serializer_tokenization_linked"),
        ("candidate_action", "candidate_action_linked"),
        ("task_state", "task_state_linked"),
    ],
)
def test_classification_single_unconfounded_group(group, expected):
    result = classify_diagnostic(
        {group},
        task_state_lead_exists=(group == "task_state"),
        all_task_state_leads_design_order_confounded=False,
    )
    assert result == expected


def test_classification_stratum_specific_groups_are_qualifying_groups():
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
            add_stratum_groups={"parent_geometry"},
        )
        == "parent_geometry_linked"
    )
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
            add_stratum_groups={"parent_geometry"},
            subtract_stratum_groups={"parent_geometry"},
        )
        == "parent_geometry_linked"
    )


def test_classification_mixed_for_multiple_groups_or_different_strata():
    assert (
        classify_diagnostic(
            {"parent_geometry", "candidate_action"},
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
        )
        == "mixed"
    )
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
            add_stratum_groups={"parent_geometry", "candidate_action"},
        )
        == "mixed"
    )
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
            add_stratum_groups={"parent_geometry"},
            subtract_stratum_groups={"candidate_action"},
        )
        == "mixed"
    )


def test_classification_design_order_confounded_and_no_clear():
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=True,
            all_task_state_leads_design_order_confounded=True,
        )
        == "design_order_confounded"
    )
    assert (
        classify_diagnostic(
            set(),
            task_state_lead_exists=False,
            all_task_state_leads_design_order_confounded=False,
        )
        == "no_clear_static_explanation"
    )


def test_classification_rejects_inconsistent_task_lead_inputs():
    with pytest.raises(ValueError, match="unconfounded task-state lead"):
        classify_diagnostic(
            set(),
            task_state_lead_exists=True,
            all_task_state_leads_design_order_confounded=False,
        )
