"""Frozen LEARN-v0 confirmatory protocol validation.

This module validates protocol structure only. It does not run confirmatory measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import cast

from aster.agent.policy import RuleBasedPolicy
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.service_agent import build_executor
from aster.training.decision import examples_from_teacher_trajectory

SCHEMA_VERSION = "aster-learn-confirmatory-manifest-0"
PROTOCOL_ID = "learn-correction-transfer-confirmatory-v0"
EXPECTED_MANIFEST_SHA256 = "72fa77acf48403d5a45f927f1122d6fb5753b1118d9f25322e1d6700f2cb1563"
DEFAULT_MANIFEST_PATH = Path(
    "docs/experiments/learn-correction-transfer-confirmatory-v0.json"
)
_RUN_ID = re.compile(r"[0-9a-f]{32}")


def load_confirmatory_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, object]:
    """Load and validate the frozen protocol without opening any evaluation data."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Confirmatory manifest must be a JSON object")
    manifest = cast(dict[str, object], raw)
    validate_confirmatory_manifest(manifest)
    return manifest


def validate_confirmatory_candidate_coverage(
    manifest: dict[str, object],
) -> dict[str, int]:
    """Verify every frozen correction/sibling teacher Action is representable by candidates."""
    validate_confirmatory_manifest(manifest)
    design = _require_dict(manifest, "family_design")
    families = _require_list(design, "families")
    tasks_checked = 0
    decisions_checked = 0
    for raw in families:
        if not isinstance(raw, dict):
            raise ValueError("Each confirmatory family must be an object")
        family = cast(dict[str, object], raw)
        for role in ("correction_task", "sibling_task"):
            task = _require_dict(family, role)
            trajectory = run_loop(
                policy=RuleBasedPolicy(),
                executor=build_executor(),
                evaluator=TaskEvaluator(),
                context=RuntimeContext(task=cast(dict[str, JsonValue], task)),
            )
            examples = examples_from_teacher_trajectory(
                trajectory,
                teacher=f"rule-v0:confirmatory-coverage:{role}",
            )
            if not examples:
                raise ValueError("Confirmatory task produced no teacher decisions")
            tasks_checked += 1
            decisions_checked += len(examples)
    return {
        "tasks_checked": tasks_checked,
        "decisions_checked": decisions_checked,
    }


def confirmatory_manifest_sha256(manifest: dict[str, object]) -> str:
    """Return the canonical manifest digest used to freeze protocol v0."""
    return digest(json_bytes(manifest))


def validate_confirmatory_manifest(manifest: dict[str, object]) -> None:
    """Reject protocol drift that would change the confirmatory claim."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported confirmatory manifest schema")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected confirmatory protocol ID")
    if manifest.get("status") != "frozen_before_measurement":
        raise ValueError("Confirmatory protocol must be frozen before measurement")

    lineage = _require_dict(manifest, "lineage")
    run_id = _require_str(lineage, "parent_act_run_id")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Confirmatory parent ACT Run ID is invalid")
    if lineage.get("parent_recipe_id") != "agent-decision-model-v0":
        raise ValueError("Unexpected confirmatory parent recipe")
    if lineage.get("parent_suite_id") != "calculate-and-store-v0":
        raise ValueError("Unexpected confirmatory parent suite")
    if lineage.get("require_exact_parent_suite_digest") is not True:
        raise ValueError("Exact parent suite digest verification must remain enabled")

    training = _require_dict(manifest, "training")
    if training.get("seeds") != [42, 43, 44]:
        raise ValueError("Confirmatory seeds must remain [42, 43, 44]")
    if training.get("optimizer_steps") != 100:
        raise ValueError("Confirmatory optimizer-step budget changed")
    if training.get("learning_rate") != 0.003:
        raise ValueError("Confirmatory learning rate changed")
    if training.get("train_backbone") is not True:
        raise ValueError("Confirmatory train_backbone changed")
    if training.get("max_episode_steps") != 8:
        raise ValueError("Confirmatory episode horizon changed")
    if training.get("policy_mode") != "model_only" or training.get("fallback") is not False:
        raise ValueError("Confirmatory policy must remain model-only without fallback")
    if training.get("serializer_id") != "aster-decision-input-0":
        raise ValueError("Confirmatory serializer changed")
    if training.get("candidate_builder_id") != "calculate-and-store-v0":
        raise ValueError("Confirmatory candidate builder changed")
    if training.get("equal_flops_claimed") is not False:
        raise ValueError("Confirmatory protocol must not claim equal FLOPs")

    design = _require_dict(manifest, "family_design")
    if design.get("planned_families") != 30:
        raise ValueError("Confirmatory family count must remain 30")
    if design.get("shift_axis") != "store_as_key_only":
        raise ValueError("Confirmatory shift axis changed")
    if design.get("sibling_used_for_training") is not False:
        raise ValueError("Uncorrected siblings must never enter training")
    if design.get("development_family_excluded") != "learn-dev-key-shift-family-v0":
        raise ValueError("Development family exclusion changed")
    families = _require_list(design, "families")
    if len(families) != 30:
        raise ValueError("Confirmatory manifest must explicitly list 30 families")
    _validate_families(families)

    endpoints = _require_dict(manifest, "endpoints")
    primary = _require_dict(endpoints, "primary")
    if primary.get("level") != "L1_local_transfer":
        raise ValueError("L1 Local Transfer must remain the primary endpoint")
    if primary.get("metric") != "uncorrected_sibling_teacher_prefix_accuracy":
        raise ValueError("Primary endpoint metric changed")
    if primary.get("contrast") != "correction_minus_replay":
        raise ValueError("Primary arm contrast changed")
    if primary.get("seed_aggregation") != "mean_within_family_over_fixed_seeds":
        raise ValueError("Primary seed aggregation changed")
    if primary.get("family_outcome") != "win_if_positive_loss_if_negative_tie_if_zero":
        raise ValueError("Primary family outcome rule changed")

    analysis = _require_dict(manifest, "analysis")
    if analysis.get("primary_test") != "exact_two_sided_sign_test_on_non_tied_family_outcomes":
        raise ValueError("Primary confirmatory test changed")
    if analysis.get("alpha") != 0.05:
        raise ValueError("Confirmatory alpha changed")
    if analysis.get("planned_family_count") != 30:
        raise ValueError("Analysis family count changed")
    if analysis.get("minimum_non_tied_families_for_confirmatory_verdict") != 20:
        raise ValueError("Minimum non-tied family rule changed")

    sealed = _require_dict(manifest, "sealed_test")
    if sealed.get("project_sealed_test_opened") is not False:
        raise ValueError("Project sealed test must remain unopened")
    if sealed.get("sibling_members_must_never_enter_training") is not True:
        raise ValueError("Sibling training exclusion must remain explicit")


def _validate_families(families: list[object]) -> None:
    family_ids: set[str] = set()
    leakage_groups: set[str] = set()
    task_ids: set[str] = set()
    task_payloads: set[str] = set()
    operation_counts = {"add": 0, "subtract": 0}

    for raw in families:
        if not isinstance(raw, dict):
            raise ValueError("Each confirmatory family must be an object")
        family = cast(dict[str, object], raw)
        family_id = _require_str(family, "family_id")
        leakage_group = _require_str(family, "leakage_group")
        if family_id in family_ids or leakage_group in leakage_groups:
            raise ValueError("Confirmatory family IDs and leakage groups must be unique")
        family_ids.add(family_id)
        leakage_groups.add(leakage_group)

        correction = _require_dict(family, "correction_task")
        sibling = _require_dict(family, "sibling_task")
        _validate_task_pair(correction, sibling)

        for task in (correction, sibling):
            task_id = _require_str(task, "task_id")
            if task_id in task_ids:
                raise ValueError("Confirmatory task IDs must be unique")
            task_ids.add(task_id)
            serialized = json.dumps(task, ensure_ascii=False, sort_keys=True)
            if serialized in task_payloads:
                raise ValueError("Confirmatory tasks must be unique")
            task_payloads.add(serialized)

        operation = _require_str(correction, "operation")
        if operation not in operation_counts:
            raise ValueError("Confirmatory operations are limited to add/subtract")
        operation_counts[operation] += 1

    if operation_counts != {"add": 15, "subtract": 15}:
        raise ValueError("Confirmatory families must remain balanced across add/subtract")


def _validate_task_pair(
    correction: dict[str, object],
    sibling: dict[str, object],
) -> None:
    required = {"task_id", "kind", "operation", "left", "right", "store_as"}
    if set(correction) != required or set(sibling) != required:
        raise ValueError("Confirmatory tasks have an unexpected schema")
    if correction.get("kind") != "calculate_and_store" or sibling.get("kind") != "calculate_and_store":
        raise ValueError("Confirmatory tasks must be calculate_and_store")

    for field in ("operation", "left", "right"):
        if correction.get(field) != sibling.get(field):
            raise ValueError("Correction and sibling may differ only in task_id/store_as")
    if correction.get("task_id") == sibling.get("task_id"):
        raise ValueError("Correction and sibling task IDs must differ")
    if correction.get("store_as") == sibling.get("store_as"):
        raise ValueError("Correction and sibling store_as keys must differ")
    if not isinstance(correction.get("left"), int) or isinstance(correction.get("left"), bool):
        raise ValueError("Confirmatory left operand must be an integer")
    if not isinstance(correction.get("right"), int) or isinstance(correction.get("right"), bool):
        raise ValueError("Confirmatory right operand must be an integer")
    if correction.get("operation") == "subtract":
        left = cast(int, correction["left"])
        right = cast(int, correction["right"])
        if left <= right:
            raise ValueError("Confirmatory subtraction tasks must remain positive")


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
