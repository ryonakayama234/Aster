"""Pure schema/statistics utilities for DIAG v3 family/state feature audit."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import math
from typing import cast

from aster.corpus.pipeline import digest, json_bytes


SCHEMA_VERSION = "aster-diag-family-feature-schema-0"
FEATURE_GROUPS = (
    "task_state",
    "serializer_tokenization",
    "parent_geometry",
    "candidate_action",
)

GLOBAL_RHO_THRESHOLD = 0.40
STRATUM_MIN_RHO_THRESHOLD = 0.25
STRATUM_SPECIFIC_RHO_THRESHOLD = 0.50
COLLINEARITY_RHO_THRESHOLD = 0.95
DESIGN_ORDER_RHO_THRESHOLD = 0.95

_AGGREGATIONS = ("min", "mean", "max")


def _output_names(name: str, scope: str) -> list[str]:
    if scope == "family_numeric" or scope == "family_categorical":
        return [name]
    if scope == "per_member_numeric":
        return [
            f"{member}_{name}_{aggregation}"
            for member in ("correction", "sibling")
            for aggregation in _AGGREGATIONS
        ]
    if scope == "pair_numeric":
        return [f"pair_{name}_{aggregation}" for aggregation in _AGGREGATIONS]
    if scope == "per_member_categorical":
        return [
            f"{member}_{name}_{suffix}"
            for member in ("correction", "sibling")
            for suffix in ("counts", "mode")
        ]
    if scope == "pair_categorical":
        return [f"pair_{name}_{suffix}" for suffix in ("counts", "mode")]
    raise ValueError(f"Unsupported feature scope: {scope}")


def _feature(
    name: str,
    *,
    group: str,
    kind: str,
    scope: str,
    unit: str,
    definition: str,
    screening_eligible: bool = True,
) -> dict[str, object]:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown DIAG v3 feature group: {group}")
    if kind not in {"continuous", "categorical"}:
        raise ValueError(f"Unsupported DIAG v3 feature kind: {kind}")
    return {
        "name": name,
        "group": group,
        "kind": kind,
        "scope": scope,
        "unit": unit,
        "definition": definition,
        "screening_eligible": screening_eligible,
        "output_names": _output_names(name, scope),
    }


def build_frozen_feature_schema() -> dict[str, object]:
    """Return the outcome-independent DIAG v3 feature schema.

    Primitive measurements have exactly one owning feature group. Output columns are
    generated deterministically from the primitive scope and fixed aggregation rule.
    """
    features = [
        _feature(
            "family_index",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="ordinal_index",
            definition="1..30 manifest order; design control only",
            screening_eligible=False,
        ),
        _feature(
            "stratum_index",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="ordinal_index",
            definition="1..15 order separately within add/subtract; design control only",
            screening_eligible=False,
        ),
        _feature(
            "operation",
            group="task_state",
            kind="categorical",
            scope="family_categorical",
            unit="category",
            definition="Frozen task operation: add or subtract",
        ),
        _feature(
            "left_operand",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="integer",
            definition="Frozen left arithmetic operand",
        ),
        _feature(
            "right_operand",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="integer",
            definition="Frozen right arithmetic operand",
        ),
        _feature(
            "result",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="integer",
            definition="Exact arithmetic result of the frozen task",
        ),
        _feature(
            "left_digits",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="digits",
            definition="Base-10 digit count of left operand",
        ),
        _feature(
            "right_digits",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="digits",
            definition="Base-10 digit count of right operand",
        ),
        _feature(
            "result_digits",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="digits",
            definition="Base-10 digit count of arithmetic result",
        ),
        _feature(
            "absolute_operand_gap",
            group="task_state",
            kind="continuous",
            scope="family_numeric",
            unit="integer",
            definition="Absolute difference between left and right operands",
        ),
        _feature(
            "serialized_input_char_length",
            group="serializer_tokenization",
            kind="continuous",
            scope="per_member_numeric",
            unit="characters",
            definition="Character length of one serialized Decision input",
        ),
        _feature(
            "encoded_input_token_length",
            group="serializer_tokenization",
            kind="continuous",
            scope="per_member_numeric",
            unit="tokens",
            definition="Tokenizer length of serialized Decision input with BOS/EOS",
        ),
        _feature(
            "correction_sibling_input_token_length_difference",
            group="serializer_tokenization",
            kind="continuous",
            scope="pair_numeric",
            unit="tokens",
            definition="Absolute token-length difference for aligned correction/sibling inputs",
        ),
        _feature(
            "token_levenshtein_distance",
            group="serializer_tokenization",
            kind="continuous",
            scope="pair_numeric",
            unit="token_edits",
            definition="Token-level Levenshtein distance for aligned correction/sibling inputs",
        ),
        _feature(
            "longest_common_token_prefix_length",
            group="serializer_tokenization",
            kind="continuous",
            scope="pair_numeric",
            unit="tokens",
            definition="Longest common token-prefix length for aligned correction/sibling inputs",
        ),
        _feature(
            "longest_common_token_prefix_ratio",
            group="serializer_tokenization",
            kind="continuous",
            scope="pair_numeric",
            unit="ratio",
            definition="LCP length divided by max correction/sibling token length",
        ),
        _feature(
            "target_raw_score",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="raw_score",
            definition="Uncalibrated saved-parent score of the teacher target candidate",
        ),
        _feature(
            "best_non_target_raw_score",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="raw_score",
            definition="Highest uncalibrated saved-parent score among non-target candidates",
        ),
        _feature(
            "target_margin",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="raw_score",
            definition="Target raw score minus best non-target raw score",
        ),
        _feature(
            "target_rank",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="rank",
            definition="1 + count of candidates with raw score greater than target score",
        ),
        _feature(
            "parent_correct",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="binary",
            definition="1 when saved parent selects teacher target, otherwise 0",
        ),
        _feature(
            "raw_target_nll",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="nats",
            definition="Negative log softmax probability of target at temperature 1",
        ),
        _feature(
            "raw_score_spread",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="raw_score",
            definition="Maximum candidate raw score minus minimum candidate raw score",
        ),
        _feature(
            "raw_score_entropy",
            group="parent_geometry",
            kind="continuous",
            scope="per_member_numeric",
            unit="nats",
            definition="Entropy of softmax(raw scores) at temperature 1",
        ),
        _feature(
            "target_action_type",
            group="candidate_action",
            kind="categorical",
            scope="per_member_categorical",
            unit="category",
            definition="Teacher target Action type",
        ),
        _feature(
            "top_competitor_action_type",
            group="candidate_action",
            kind="categorical",
            scope="per_member_categorical",
            unit="category",
            definition="Highest-scoring non-target Action type",
        ),
        _feature(
            "candidate_set_size",
            group="candidate_action",
            kind="continuous",
            scope="per_member_numeric",
            unit="candidates",
            definition="Number of candidates scored at one Decision state",
        ),
        _feature(
            "target_candidate_token_length",
            group="candidate_action",
            kind="continuous",
            scope="per_member_numeric",
            unit="tokens",
            definition="Encoded token length of the serialized target candidate input",
        ),
        _feature(
            "top_competitor_token_length",
            group="candidate_action",
            kind="continuous",
            scope="per_member_numeric",
            unit="tokens",
            definition="Encoded token length of the serialized top-competitor input",
        ),
        _feature(
            "target_minus_competitor_token_length_difference",
            group="candidate_action",
            kind="continuous",
            scope="per_member_numeric",
            unit="tokens",
            definition="Target candidate token length minus top-competitor token length",
        ),
        _feature(
            "top_competitor_type_matches_pair",
            group="candidate_action",
            kind="categorical",
            scope="pair_categorical",
            unit="boolean",
            definition="Whether aligned correction/sibling top competitors share Action type",
        ),
    ]
    output_names = [
        output_name
        for feature in features
        for output_name in cast(list[str], feature["output_names"])
    ]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("DIAG v3 feature schema contains duplicate output columns")
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation": {
            "continuous_decision_level": list(_AGGREGATIONS),
            "categorical": ["counts", "deterministic_mode"],
            "mode_tie": "sorted_list",
        },
        "missing_degenerate_handling": {
            "undefined_spearman": {
                "rho": None,
                "degenerate": True,
            },
            "missing_primitive_measurement": "reject",
            "nonfinite_numeric_measurement": "reject",
            "empty_numeric_aggregation": "reject",
            "empty_categorical_aggregation": "reject",
        },
        "features": features,
        "output_names": output_names,
    }


def feature_schema_sha256(schema: dict[str, object] | None = None) -> str:
    """Hash the canonical serialized outcome-independent feature schema."""
    return digest(json_bytes(build_frozen_feature_schema() if schema is None else schema))


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return 1-based average ranks, deterministically handling ties."""
    if not values:
        raise ValueError("At least one value is required")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Rank inputs must be finite")
    indexed = sorted(enumerate(numeric), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(numeric)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average
        start = end
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Spearman rho with average ranks; return None for degenerate inputs."""
    if len(left) != len(right):
        raise ValueError("Spearman vectors must have equal length")
    if len(left) < 2:
        raise ValueError("Spearman rho requires at least two paired values")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    return _pearson(left_ranks, right_ranks)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    left_ss = sum(value * value for value in centered_left)
    right_ss = sum(value * value for value in centered_right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    numerator = sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
    rho = numerator / math.sqrt(left_ss * right_ss)
    return max(-1.0, min(1.0, rho))


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    """Return the fixed min/mean/max family aggregation."""
    if not values:
        raise ValueError("At least one numeric feature value is required")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Feature values must be finite")
    return {
        "min": min(numeric),
        "mean": sum(numeric) / len(numeric),
        "max": max(numeric),
    }


def levenshtein_distance(left: Sequence[object], right: Sequence[object]) -> int:
    """Return insertion/deletion/substitution distance for two token sequences."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def longest_common_prefix_length(left: Sequence[object], right: Sequence[object]) -> int:
    """Count equal leading tokens in two aligned serialized inputs."""
    length = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        length += 1
    return length


def longest_common_prefix_ratio(left: Sequence[object], right: Sequence[object]) -> float:
    """Return LCP / max sequence length, with two empty inputs defined as 1."""
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return longest_common_prefix_length(left, right) / denominator


def design_order_confounded(
    feature_values: Sequence[float],
    stratum_index: Sequence[float],
    *,
    threshold: float = DESIGN_ORDER_RHO_THRESHOLD,
) -> bool:
    """Whether one within-stratum feature is nearly monotonic with design order."""
    rho = spearman_rho(feature_values, stratum_index)
    return rho is not None and abs(rho) >= threshold


def collinearity_clusters(
    feature_values: Mapping[str, Sequence[float]],
    *,
    threshold: float = COLLINEARITY_RHO_THRESHOLD,
) -> list[tuple[str, ...]]:
    """Build deterministic connected components from absolute Spearman edges."""
    names = sorted(feature_values)
    adjacency = {name: set() for name in names}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            rho = spearman_rho(feature_values[left_name], feature_values[right_name])
            if rho is not None and abs(rho) >= threshold:
                adjacency[left_name].add(right_name)
                adjacency[right_name].add(left_name)

    components: list[tuple[str, ...]] = []
    unseen = set(names)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        components.append(tuple(sorted(component)))
    return sorted(components)


def operational_lead(
    *,
    rho_all30: float | None,
    rho_add: float | None,
    rho_subtract: float | None,
    design_order_confounded_flag: bool,
) -> dict[str, object]:
    """Apply the preregistered descriptive lead heuristics."""
    global_lead = (
        rho_all30 is not None
        and rho_add is not None
        and rho_subtract is not None
        and abs(rho_all30) >= GLOBAL_RHO_THRESHOLD
        and rho_add * rho_subtract > 0
        and min(abs(rho_add), abs(rho_subtract)) >= STRATUM_MIN_RHO_THRESHOLD
    )

    add_stratum_lead = (
        rho_add is not None
        and abs(rho_add) >= STRATUM_SPECIFIC_RHO_THRESHOLD
        and _opposite_stratum_is_specific(rho_add, rho_subtract)
    )
    subtract_stratum_lead = (
        rho_subtract is not None
        and abs(rho_subtract) >= STRATUM_SPECIFIC_RHO_THRESHOLD
        and _opposite_stratum_is_specific(rho_subtract, rho_add)
    )
    stratum_specific = add_stratum_lead or subtract_stratum_lead
    bidirectional = (
        rho_add is not None
        and rho_subtract is not None
        and abs(rho_add) >= STRATUM_SPECIFIC_RHO_THRESHOLD
        and abs(rho_subtract) >= STRATUM_SPECIFIC_RHO_THRESHOLD
        and rho_add * rho_subtract < 0
    )

    if global_lead:
        lead_kind: str | None = "global_descriptive_lead"
    elif stratum_specific:
        lead_kind = "stratum_specific_lead"
    else:
        lead_kind = None

    return {
        "lead_kind": lead_kind,
        "global_descriptive_lead": global_lead,
        "stratum_specific_lead": stratum_specific,
        "add_stratum_lead": add_stratum_lead,
        "subtract_stratum_lead": subtract_stratum_lead,
        "bidirectional_stratum_heterogeneity": bidirectional,
        "design_order_confounded": design_order_confounded_flag,
        "qualifies_unconfounded_group": (
            lead_kind is not None and not design_order_confounded_flag
        ),
    }


def _opposite_stratum_is_specific(
    strong_rho: float,
    other_rho: float | None,
) -> bool:
    if other_rho is None:
        return True
    if abs(other_rho) < STRATUM_MIN_RHO_THRESHOLD:
        return True
    return strong_rho * other_rho < 0


_CLASSIFICATION_BY_GROUP = {
    "parent_geometry": "parent_geometry_linked",
    "serializer_tokenization": "serializer_tokenization_linked",
    "candidate_action": "candidate_action_linked",
    "task_state": "task_state_linked",
}


def classify_diagnostic(
    qualifying_unconfounded_groups: Collection[str],
    *,
    task_state_lead_exists: bool,
    all_task_state_leads_design_order_confounded: bool,
    add_stratum_groups: Collection[str] = (),
    subtract_stratum_groups: Collection[str] = (),
) -> str:
    """Return exactly one preregistered DIAG v3 diagnostic classification."""
    add_groups = set(add_stratum_groups)
    subtract_groups = set(subtract_stratum_groups)
    groups = set(qualifying_unconfounded_groups) | add_groups | subtract_groups
    unknown = groups - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown DIAG v3 feature groups: {sorted(unknown)}")
    if all_task_state_leads_design_order_confounded and not task_state_lead_exists:
        raise ValueError("Confounded task-state leads cannot exist when there is no task lead")
    if (
        task_state_lead_exists
        and not all_task_state_leads_design_order_confounded
        and "task_state" not in groups
    ):
        raise ValueError("An unconfounded task-state lead must qualify task_state")

    if len(groups) >= 2:
        return "mixed"

    if len(groups) == 1:
        group = next(iter(groups))
        return _CLASSIFICATION_BY_GROUP[group]

    if task_state_lead_exists and all_task_state_leads_design_order_confounded:
        return "design_order_confounded"
    return "no_clear_static_explanation"
