"""Pure aggregation and exact inference for the frozen LEARN-v0 confirmatory protocol."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Sequence, cast

from aster.training.learn_confirmatory import (
    PROTOCOL_ID,
    confirmatory_manifest_sha256,
    validate_confirmatory_manifest,
)


def exact_two_sided_sign_test_p_value(wins: int, losses: int) -> float | None:
    """Exact two-sided sign-test p-value under P(win)=P(loss)=0.5.

    Ties are excluded before calling this function. With no non-tied families the
    confirmatory p-value is undefined and None is returned.
    """
    if type(wins) is not int or type(losses) is not int or wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative integers")
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    tail_count = sum(comb(n, k) for k in range(tail + 1))
    p_exact = min(Fraction(1, 1), Fraction(2 * tail_count, 2**n))
    return float(p_exact)


def aggregate_confirmatory_units(
    manifest: dict[str, object],
    units: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate fixed-seed units without treating seeds as independent families.

    Partial campaigns intentionally expose progress only. Family effects and the
    confirmatory verdict are emitted only after every frozen family/seed unit exists.
    """
    validate_confirmatory_manifest(manifest)
    protocol_id = _require_str(manifest, "protocol_id")
    if protocol_id != PROTOCOL_ID:
        raise ValueError("Unexpected confirmatory protocol")
    manifest_sha256 = confirmatory_manifest_sha256(manifest)

    design = _require_dict(manifest, "family_design")
    training = _require_dict(manifest, "training")
    analysis = _require_dict(manifest, "analysis")
    families = _require_list(design, "families")
    seeds_raw = training.get("seeds")
    if not isinstance(seeds_raw, list) or any(type(seed) is not int for seed in seeds_raw):
        raise ValueError("Confirmatory seeds are invalid")
    seeds = tuple(cast(list[int], seeds_raw))

    family_ids: list[str] = []
    for raw in families:
        if not isinstance(raw, dict):
            raise ValueError("Confirmatory family is invalid")
        family_ids.append(_require_str(cast(dict[str, object], raw), "family_id"))

    expected = {(family_id, seed) for family_id in family_ids for seed in seeds}
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for unit in units:
        _validate_unit_identity(unit, protocol_id, manifest_sha256)
        family_id = _require_str(unit, "family_id")
        seed = unit.get("seed")
        if type(seed) is not int:
            raise ValueError("Confirmatory unit seed must be an integer")
        key = (family_id, cast(int, seed))
        if key not in expected:
            raise ValueError(f"Unexpected confirmatory unit: {family_id} seed {seed}")
        if key in indexed:
            raise ValueError(f"Duplicate confirmatory unit: {family_id} seed {seed}")
        indexed[key] = unit

    missing = sorted(expected - set(indexed))
    if missing:
        return {
            "schema_version": "aster-learn-confirmatory-result-0",
            "protocol_id": protocol_id,
            "manifest_sha256": manifest_sha256,
            "status": "measurement_incomplete",
            "expected_units": len(expected),
            "completed_units": len(indexed),
            "missing_units": [
                {"family_id": family_id, "seed": seed}
                for family_id, seed in missing
            ],
            "family_results": None,
            "primary": None,
            "verdict": None,
        }

    family_results: list[dict[str, object]] = []
    family_deltas: list[Fraction] = []
    wins = 0
    losses = 0
    ties = 0

    for family_id in family_ids:
        family_units = [indexed[(family_id, seed)] for seed in seeds]
        deltas: list[Fraction] = []
        nll_effects: list[float] = []
        run_ids: list[str] = []
        sequential_task_success: dict[str, list[bool]] = {"replay": [], "correction": []}
        sequential_goal_verified: dict[str, list[bool]] = {"replay": [], "correction": []}
        repair_accuracy: dict[str, list[float]] = {
            "parent": [],
            "replay": [],
            "correction": [],
        }

        for unit in family_units:
            primary = _require_dict(unit, "primary")
            replay_correct = _require_int(primary, "replay_correct")
            correction_correct = _require_int(primary, "correction_correct")
            examples = _require_int(primary, "examples")
            if examples <= 0:
                raise ValueError("Primary examples must be positive")
            if not 0 <= replay_correct <= examples or not 0 <= correction_correct <= examples:
                raise ValueError("Primary correct counts are outside [0, examples]")
            deltas.append(
                Fraction(correction_correct, examples)
                - Fraction(replay_correct, examples)
            )
            nll_effects.append(
                _require_float(primary, "replay_nll")
                - _require_float(primary, "correction_nll")
            )
            run_ids.append(_require_str(unit, "run_id"))

            sequential = _require_dict(unit, "sequential")
            for arm in ("replay", "correction"):
                arm_result = _require_dict(sequential, arm)
                sequential_task_success[arm].append(
                    _require_bool(arm_result, "task_success")
                )
                sequential_goal_verified[arm].append(
                    _require_bool(arm_result, "goal_verified")
                )

            repair = _require_dict(unit, "repair")
            for arm in ("parent", "replay", "correction"):
                arm_result = _require_dict(repair, arm)
                repair_accuracy[arm].append(_require_float(arm_result, "accuracy"))

        family_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
        family_deltas.append(family_delta)
        if family_delta > 0:
            outcome = "win"
            wins += 1
        elif family_delta < 0:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1

        family_results.append(
            {
                "family_id": family_id,
                "seed_runs": [
                    {"seed": seed, "run_id": run_id}
                    for seed, run_id in zip(seeds, run_ids, strict=True)
                ],
                "primary_accuracy_delta": float(family_delta),
                "primary_accuracy_delta_exact": (
                    f"{family_delta.numerator}/{family_delta.denominator}"
                ),
                "outcome": outcome,
                "secondary_nll_effect_replay_minus_correction": (
                    sum(nll_effects) / len(nll_effects)
                ),
                "sequential_task_success_rate": {
                    arm: sum(values) / len(values)
                    for arm, values in sequential_task_success.items()
                },
                "sequential_goal_verified_rate": {
                    arm: sum(values) / len(values)
                    for arm, values in sequential_goal_verified.items()
                },
                "repair_accuracy": {
                    arm: sum(values) / len(values)
                    for arm, values in repair_accuracy.items()
                },
            }
        )

    non_tied = wins + losses
    p_value = exact_two_sided_sign_test_p_value(wins, losses)
    median_delta = _median_fraction(family_deltas)
    alpha = _require_float(analysis, "alpha")
    minimum_non_tied = _require_int(
        analysis, "minimum_non_tied_families_for_confirmatory_verdict"
    )

    if non_tied < minimum_non_tied:
        verdict = "Inconclusive"
        reason = "fewer_than_minimum_non_tied_families"
    elif p_value is not None and p_value < alpha and wins > losses and median_delta > 0:
        verdict = "Supported"
        reason = "primary_rule_supported"
    elif p_value is not None and p_value < alpha and losses > wins:
        verdict = "Not supported"
        reason = "primary_rule_not_supported"
    else:
        verdict = "Inconclusive"
        reason = "primary_rule_not_decisive"

    return {
        "schema_version": "aster-learn-confirmatory-result-0",
        "protocol_id": protocol_id,
        "manifest_sha256": manifest_sha256,
        "status": "complete",
        "expected_units": len(expected),
        "completed_units": len(indexed),
        "family_results": family_results,
        "primary": {
            "endpoint": "uncorrected_sibling_teacher_prefix_accuracy",
            "unit_of_evidence": "task_generator_family",
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "non_tied_families": non_tied,
            "sign_test_p_value": p_value,
            "median_family_accuracy_delta": float(median_delta),
            "median_family_accuracy_delta_exact": (
                f"{median_delta.numerator}/{median_delta.denominator}"
            ),
            "alpha": alpha,
        },
        "verdict": {
            "label": verdict,
            "reason": reason,
        },
    }


def _validate_unit_identity(
    unit: dict[str, object],
    protocol_id: str,
    manifest_sha256: str,
) -> None:
    if unit.get("schema_version") != "aster-learn-confirmatory-unit-0":
        raise ValueError("Unsupported confirmatory unit schema")
    if unit.get("protocol_id") != protocol_id:
        raise ValueError("Confirmatory unit protocol mismatch")
    if unit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Confirmatory unit manifest digest mismatch")


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("At least one family delta is required")
    ordered = sorted(values)
    size = len(ordered)
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _require_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Confirmatory field {key!r} must be an object")
    return cast(dict[str, object], value)


def _require_list(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Confirmatory field {key!r} must be a list")
    return cast(list[object], value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Confirmatory field {key!r} must be a non-empty string")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"Confirmatory field {key!r} must be an integer")
    return cast(int, value)


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Confirmatory field {key!r} must be numeric")
    return float(value)


def _require_bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise ValueError(f"Confirmatory field {key!r} must be boolean")
    return cast(bool, value)
