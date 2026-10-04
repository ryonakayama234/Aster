"""DIAG v1: paired full-backbone vs frozen-backbone causal Trial."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from typing import Sequence, cast

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_artifact import load_decision_artifact
from aster.records.decision import DecisionExample
from aster.records.intervention import read_intervention_traces_jsonl
from aster.records.runlog import RunLog
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.runtime.service_agent import build_executor
from aster.training.decision import DecisionTrainConfig, train_decision_with_checkpoints
from aster.training.diag_interference import (
    TRIAL_CHECKPOINTS,
    _checkpoint_metrics,
    _checkpoint_resource_budget,
    _clean_git_sha,
    _family_by_id,
    _read_json,
    _require_dict,
    _require_float,
    _require_int,
    _require_list,
    _require_str,
    _task_from_family,
    _teacher_examples,
    _validate_parent_suite_lineage,
    _write_json,
    summarize_diag_trial_unit,
)
from aster.training.intervention import (
    aggregate_decision_examples,
    examples_from_interventions,
    replay_decision_examples,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    load_confirmatory_manifest,
)
from aster.training.learn_probe import DEFAULT_ACT_RUN_ID, resolve_act_parent_artifact

TRIAL_CYCLE_ID = "diag-backbone-freeze-trial-0"
TRIAL_PARENT_CYCLE_ID = "diag-correction-interference-trial-1"
TRIAL_FAMILY_IDS = (
    "learn-confirm-keyshift-03",
    "learn-confirm-keyshift-15",
    "learn-confirm-keyshift-02",
    "learn-confirm-keyshift-12",
)
TRIAL_FAMILY_STRATA = {
    "learn-confirm-keyshift-03": "loss",
    "learn-confirm-keyshift-15": "control",
    "learn-confirm-keyshift-02": "loss",
    "learn-confirm-keyshift-12": "control",
}
TRIAL_BACKBONE_CONDITIONS = (True, False)
TRIAL_SEED = 42
TRIAL_STEPS = 100
TRIAL_LEARNING_RATE = 3e-3
TRIAL_MAX_EPISODE_STEPS = 8
ONSET_EPSILON = 1e-6

EXPECTED_FAMILY_BLOCKS = len(TRIAL_FAMILY_IDS)
EXPECTED_EXPERIMENT_UNITS = len(TRIAL_FAMILY_IDS) * len(TRIAL_BACKBONE_CONDITIONS)
EXPECTED_ARM_TRAJECTORIES = EXPECTED_EXPERIMENT_UNITS * 2
EXPECTED_UNIT_CHECKPOINT_RECORDS = EXPECTED_EXPERIMENT_UNITS * len(TRIAL_CHECKPOINTS)
EXPECTED_ARM_CHECKPOINT_RECORDS = EXPECTED_UNIT_CHECKPOINT_RECORDS * 2


def backbone_condition_id(train_backbone: bool) -> str:
    return "full" if train_backbone else "freeze"


def run_backbone_freeze_unit(
    root: str | Path,
    *,
    family_id: str,
    train_backbone: bool,
    expected_source_git_sha: str | None = None,
) -> Path:
    """Run one family x backbone-condition unit with matched Replay/Correction arms."""
    if family_id not in TRIAL_FAMILY_IDS:
        raise ValueError(f"Family is not selected for DIAG v1: {family_id}")
    if type(train_backbone) is not bool:
        raise ValueError("train_backbone must be a bool")

    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    if (
        expected_source_git_sha is not None
        and source_git_sha != expected_source_git_sha
    ):
        raise RuntimeError("Git revision changed during DIAG v1")

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    family = _family_by_id(manifest, family_id)
    correction_task = _task_from_family(family, "correction_task")
    sibling_task = _task_from_family(family, "sibling_task")
    operation_value = correction_task.get("operation")
    if not isinstance(operation_value, str) or not operation_value:
        raise ValueError("DIAG v1 correction operation must be a non-empty string")
    operation = operation_value
    condition_id = backbone_condition_id(train_backbone)

    artifact_id, artifact_path = resolve_act_parent_artifact(
        root_path, DEFAULT_ACT_RUN_ID
    )
    parent_model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved DIAG v1 parent artifact identity mismatch")
    baseline = build_calculate_and_store_suite()
    _validate_parent_suite_lineage(parent_manifest, baseline)
    parent_model_id = _require_str(parent_manifest, "model_id")

    config = DecisionTrainConfig(
        steps=TRIAL_STEPS,
        learning_rate=TRIAL_LEARNING_RATE,
        train_backbone=train_backbone,
        seed=TRIAL_SEED,
    )
    policy = SelectivePolicy(
        ModelPolicy(parent_model, tokenizer, model_id=parent_model_id),
        fallback=None,
        config=SelectivePolicyConfig(
            temperature=1.0,
            autonomous_threshold=0.0,
            fallback_threshold=0.0,
        ),
    )
    provenance = {
        "experiment_mode": "trial",
        "cycle_id": TRIAL_CYCLE_ID,
        "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "planned_experiment_units": EXPECTED_EXPERIMENT_UNITS,
        "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
        "source_git_sha": source_git_sha,
        "parent_artifact_id": artifact_id,
        "act_run_id": DEFAULT_ACT_RUN_ID,
        "family_id": family_id,
        "family_stratum": TRIAL_FAMILY_STRATA[family_id],
        "operation": operation,
        "seed": TRIAL_SEED,
        "train_backbone": train_backbone,
        "backbone_condition": condition_id,
    }
    run = RunLog(
        root_path,
        "diag_backbone_freeze_trial_unit",
        {
            "provenance": provenance,
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "train_config": asdict(config),
            "correction_task": correction_task,
            "uncorrected_sibling_task": sibling_task,
        },
        producer="trainer",
    )

    try:
        rollout_path = run_logged_intervention_agent(
            root_path,
            policy=policy,
            teacher=RuleBasedPolicy(),
            teacher_id="rule-v0:diag-backbone-freeze-trial-0",
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=RuntimeContext(task=correction_task),
            max_steps=TRIAL_MAX_EPISODE_STEPS,
        )
        traces = read_intervention_traces_jsonl(
            rollout_path / "interventions.jsonl"
        )
        if not traces:
            raise ValueError("DIAG v1 correction rollout produced no intervention traces")
        if any(trace.route != "model" for trace in traces):
            raise ValueError("DIAG v1 correction rollout must remain model-only")
        if any(trace.executed_action != trace.student_action for trace in traces):
            raise ValueError("DIAG v1 model-only rollout must execute student Action")

        correction_examples = examples_from_interventions(traces)
        if not correction_examples:
            raise ValueError("DIAG v1 correction rollout produced no trainable labels")
        base_examples = tuple(baseline.training_examples())
        replay_examples = replay_decision_examples(
            base_examples, len(correction_examples)
        )
        replay_training = aggregate_decision_examples(
            base_examples, replay_examples
        )
        correction_training = aggregate_decision_examples(
            base_examples, correction_examples
        )
        if len(replay_training) != len(correction_training):
            raise RuntimeError("DIAG v1 Replay/Correction training lengths differ")

        training_digests = {
            "replay": _examples_digest(replay_training),
            "correction": _examples_digest(correction_training),
        }
        sibling_examples = tuple(_teacher_examples(sibling_task))
        replay_candidate = deepcopy(parent_model)
        correction_candidate = deepcopy(parent_model)

        replay_losses, replay_snapshots, replay_elapsed = train_decision_with_checkpoints(
            replay_candidate,
            tokenizer,
            replay_training,
            checkpoints=TRIAL_CHECKPOINTS,
            config=config,
        )
        correction_losses, correction_snapshots, correction_elapsed = (
            train_decision_with_checkpoints(
                correction_candidate,
                tokenizer,
                correction_training,
                checkpoints=TRIAL_CHECKPOINTS,
                config=config,
            )
        )
        if len(replay_losses) != len(correction_losses):
            raise RuntimeError("DIAG v1 Replay/Correction optimizer-step counts differ")

        resources = {
            "replay": _checkpoint_resource_budget(
                tokenizer,
                replay_training,
                step=config.steps,
                elapsed_wall_seconds=replay_elapsed[config.steps],
            ),
            "correction": _checkpoint_resource_budget(
                tokenizer,
                correction_training,
                step=config.steps,
                elapsed_wall_seconds=correction_elapsed[config.steps],
            ),
        }
        _write_json(
            run.path / "training-replay.json",
            {
                "schema_version": "aster-diag-backbone-freeze-training-0",
                "arm": "replay",
                "backbone_condition": condition_id,
                "config": asdict(config),
                "training_examples_sha256": training_digests["replay"],
                "losses": replay_losses,
                "resources": resources["replay"],
            },
        )
        _write_json(
            run.path / "training-correction.json",
            {
                "schema_version": "aster-diag-backbone-freeze-training-0",
                "arm": "correction",
                "backbone_condition": condition_id,
                "config": asdict(config),
                "training_examples_sha256": training_digests["correction"],
                "losses": correction_losses,
                "resources": resources["correction"],
            },
        )

        checkpoints: list[dict[str, object]] = []
        for step in TRIAL_CHECKPOINTS:
            checkpoints.append(
                {
                    "step": step,
                    "replay": _checkpoint_metrics(
                        parent_model,
                        replay_snapshots[step],
                        tokenizer,
                        repair_examples=correction_examples,
                        sibling_examples=sibling_examples,
                        base_examples=base_examples,
                        resources=_checkpoint_resource_budget(
                            tokenizer,
                            replay_training,
                            step=step,
                            elapsed_wall_seconds=replay_elapsed[step],
                        ),
                    ),
                    "correction": _checkpoint_metrics(
                        parent_model,
                        correction_snapshots[step],
                        tokenizer,
                        repair_examples=correction_examples,
                        sibling_examples=sibling_examples,
                        base_examples=base_examples,
                        resources=_checkpoint_resource_budget(
                            tokenizer,
                            correction_training,
                            step=step,
                            elapsed_wall_seconds=correction_elapsed[step],
                        ),
                    ),
                }
            )

        unit: dict[str, object] = {
            "schema_version": "aster-diag-backbone-freeze-unit-0",
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "source_git_sha": source_git_sha,
            "run_id": run.id,
            "rollout_run_id": rollout_path.name,
            "parent_artifact_id": artifact_id,
            "family_id": family_id,
            "family_stratum": TRIAL_FAMILY_STRATA[family_id],
            "operation": operation,
            "seed": TRIAL_SEED,
            "train_backbone": train_backbone,
            "backbone_condition": condition_id,
            "independent_family_block": family_id,
            "backbone_condition_is_independent_sample": False,
            "checkpoint_observations_are_independent": False,
            "training_examples_sha256": training_digests,
            "checkpoints": checkpoints,
            "resources": resources,
        }
        unit["trajectory_summary"] = summarize_diag_trial_unit(unit)
        unit["condition_summary"] = summarize_backbone_freeze_unit(unit)
        _validate_frozen_backbone_drift(unit)
        _write_json(run.path / "diag-backbone-freeze-unit.json", unit)
        run.finish(
            "completed",
            result="diag-backbone-freeze-unit.json",
            rollout_run_id=rollout_path.name,
            family_id=family_id,
            backbone_condition=condition_id,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def summarize_backbone_freeze_unit(unit: dict[str, object]) -> dict[str, object]:
    """Compute within-unit causal contrasts and baseline-relative onset metrics."""
    trajectory_summary = _require_dict(unit, "trajectory_summary")
    replay_summary = _require_dict(trajectory_summary, "replay")
    correction_summary = _require_dict(trajectory_summary, "correction")

    replay_sibling_margin_delta = _require_float(
        replay_summary, "sibling_margin_delta"
    )
    correction_sibling_margin_delta = _require_float(
        correction_summary, "sibling_margin_delta"
    )
    replay_repair_accuracy_delta = _require_float(
        replay_summary, "repair_accuracy_delta"
    )
    correction_repair_accuracy_delta = _require_float(
        correction_summary, "repair_accuracy_delta"
    )

    checkpoints = _checkpoint_rows(unit)
    baseline_replay_margin = _sibling_margin(checkpoints[0], "replay")
    baseline_correction_margin = _sibling_margin(checkpoints[0], "correction")

    correction_change_rows: list[tuple[int, float]] = []
    cr_divergence_rows: list[tuple[int, float]] = []
    for step in TRIAL_CHECKPOINTS:
        replay_change = _sibling_margin(checkpoints[step], "replay") - baseline_replay_margin
        correction_change = (
            _sibling_margin(checkpoints[step], "correction")
            - baseline_correction_margin
        )
        correction_change_rows.append((step, correction_change))
        cr_divergence_rows.append((step, correction_change - replay_change))

    first_negative_correction_change = next(
        (
            step
            for step, value in correction_change_rows
            if step > 0 and value < -ONSET_EPSILON
        ),
        None,
    )
    first_negative_cr_divergence = next(
        (
            step
            for step, value in cr_divergence_rows
            if step > 0 and value < -ONSET_EPSILON
        ),
        None,
    )

    return {
        "sibling_margin_correction_minus_replay_delta": (
            correction_sibling_margin_delta - replay_sibling_margin_delta
        ),
        "repair_accuracy_correction_minus_replay_delta": (
            correction_repair_accuracy_delta - replay_repair_accuracy_delta
        ),
        "correction_sibling_margin_change_by_checkpoint": {
            str(step): value for step, value in correction_change_rows
        },
        "correction_minus_replay_sibling_margin_divergence_by_checkpoint": {
            str(step): value for step, value in cr_divergence_rows
        },
        "first_negative_correction_sibling_margin_change_step": (
            first_negative_correction_change
        ),
        "first_negative_correction_minus_replay_divergence_step": (
            first_negative_cr_divergence
        ),
        "onset_epsilon": ONSET_EPSILON,
    }


def aggregate_backbone_freeze_units(
    units: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate paired family conditions without treating conditions as samples."""
    if len(units) != EXPECTED_EXPERIMENT_UNITS:
        return {
            "schema_version": "aster-diag-backbone-freeze-result-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "status": "trial_incomplete",
            "expected_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "expected_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "completed_experiment_units": len(units),
            "confirmatory_verdict": None,
        }

    source_git_sha: str | None = None
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for unit in units:
        if unit.get("cycle_id") != TRIAL_CYCLE_ID:
            raise ValueError("Mixed DIAG v1 cycle IDs")
        family_id = _require_str(unit, "family_id")
        condition_id = _require_str(unit, "backbone_condition")
        if family_id not in TRIAL_FAMILY_IDS:
            raise ValueError(f"Unexpected DIAG v1 family: {family_id}")
        if condition_id not in ("full", "freeze"):
            raise ValueError(f"Unexpected backbone condition: {condition_id}")
        key = (family_id, condition_id)
        if key in indexed:
            raise ValueError(f"Duplicate DIAG v1 unit: {key}")
        unit_sha = _require_str(unit, "source_git_sha")
        if source_git_sha is None:
            source_git_sha = unit_sha
        elif unit_sha != source_git_sha:
            raise ValueError("DIAG v1 units were produced by different Git SHAs")
        indexed[key] = unit

    expected_keys = {
        (family_id, backbone_condition_id(train_backbone))
        for family_id in TRIAL_FAMILY_IDS
        for train_backbone in TRIAL_BACKBONE_CONDITIONS
    }
    if set(indexed) != expected_keys:
        raise ValueError("DIAG v1 completed with unexpected family/condition set")

    family_summaries: list[dict[str, object]] = []
    for family_id in TRIAL_FAMILY_IDS:
        full = indexed[(family_id, "full")]
        freeze = indexed[(family_id, "freeze")]
        _validate_matched_training_data(full, freeze)

        full_summary = _require_dict(full, "condition_summary")
        freeze_summary = _require_dict(freeze, "condition_summary")
        full_d = _require_float(
            full_summary, "sibling_margin_correction_minus_replay_delta"
        )
        freeze_d = _require_float(
            freeze_summary, "sibling_margin_correction_minus_replay_delta"
        )
        full_repair = _require_float(
            full_summary, "repair_accuracy_correction_minus_replay_delta"
        )
        freeze_repair = _require_float(
            freeze_summary, "repair_accuracy_correction_minus_replay_delta"
        )
        family_summaries.append(
            {
                "family_id": family_id,
                "family_stratum": _require_str(full, "family_stratum"),
                "operation": _require_str(full, "operation"),
                "full": {
                    "D_sibling_margin": full_d,
                    "Repair_CR": full_repair,
                },
                "freeze": {
                    "D_sibling_margin": freeze_d,
                    "Repair_CR": freeze_repair,
                },
                "freeze_effect": freeze_d - full_d,
                "repair_cr_change_under_freeze": freeze_repair - full_repair,
                "training_examples_sha256": _require_dict(
                    full, "training_examples_sha256"
                ),
            }
        )

    loss_rows = [
        row for row in family_summaries if row["family_stratum"] == "loss"
    ]
    control_rows = [
        row for row in family_summaries if row["family_stratum"] == "control"
    ]

    return {
        "schema_version": "aster-diag-backbone-freeze-result-0",
        "cycle_id": TRIAL_CYCLE_ID,
        "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
        "status": "complete",
        "experiment_mode": "trial",
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "source_git_sha": source_git_sha,
        "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
        "paired_backbone_conditions_per_family": 2,
        "experiment_units": EXPECTED_EXPERIMENT_UNITS,
        "arm_training_trajectories": EXPECTED_ARM_TRAJECTORIES,
        "unit_checkpoint_records": EXPECTED_UNIT_CHECKPOINT_RECORDS,
        "arm_checkpoint_records": EXPECTED_ARM_CHECKPOINT_RECORDS,
        "backbone_conditions_are_independent_samples": False,
        "checkpoint_observations_are_independent": False,
        "confirmatory_verdict": None,
        "primary_estimand": (
            "freeze_effect = D_freeze - D_full; "
            "D = Correction sibling-margin delta - Replay sibling-margin delta"
        ),
        "family_summaries": family_summaries,
        "descriptive_summary": {
            "loss_freeze_effect_median": median(
                _row_float(row, "freeze_effect") for row in loss_rows
            ),
            "control_freeze_effect_median": median(
                _row_float(row, "freeze_effect") for row in control_rows
            ),
            "loss_repair_cr_change_under_freeze_median": median(
                _row_float(row, "repair_cr_change_under_freeze")
                for row in loss_rows
            ),
            "control_repair_cr_change_under_freeze_median": median(
                _row_float(row, "repair_cr_change_under_freeze")
                for row in control_rows
            ),
        },
    }


def run_backbone_freeze_campaign(root: str | Path) -> Path:
    """Run or resume the fixed 4-family x 2-condition DIAG v1 campaign."""
    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    completed = discover_completed_backbone_freeze_units(
        root_path, source_git_sha=source_git_sha
    )
    campaign = RunLog(
        root_path,
        "diag_backbone_freeze_trial_campaign",
        {
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "planned_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "source_git_sha": source_git_sha,
            "families": list(TRIAL_FAMILY_IDS),
            "backbone_conditions": ["full", "freeze"],
            "seed": TRIAL_SEED,
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "completed_units_at_start": len(completed),
        },
        producer="trainer",
    )
    try:
        for family_id in TRIAL_FAMILY_IDS:
            for train_backbone in TRIAL_BACKBONE_CONDITIONS:
                condition_id = backbone_condition_id(train_backbone)
                key = (family_id, condition_id)
                if key not in completed:
                    path = run_backbone_freeze_unit(
                        root_path,
                        family_id=family_id,
                        train_backbone=train_backbone,
                        expected_source_git_sha=source_git_sha,
                    )
                    unit = _read_json(path / "diag-backbone-freeze-unit.json")
                    completed[key] = (path, unit)
                campaign.event(
                    "unit_available",
                    {
                        "family_id": family_id,
                        "backbone_condition": condition_id,
                        "run_id": completed[key][0].name,
                    },
                )
                _write_progress(campaign.path, completed)

        result = aggregate_backbone_freeze_units(
            [unit for _, unit in completed.values()]
        )
        if result.get("status") != "complete":
            raise RuntimeError("DIAG v1 ended without all fixed units")
        _write_json(campaign.path / "diag-backbone-freeze-results.json", result)
        campaign.finish(
            "completed",
            results="diag-backbone-freeze-results.json",
            completed_experiment_units=len(completed),
        )
        return campaign.path
    except BaseException as error:
        campaign.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def discover_completed_backbone_freeze_units(
    root: str | Path,
    *,
    source_git_sha: str,
) -> dict[tuple[str, str], tuple[Path, dict[str, object]]]:
    """Reuse only completed DIAG v1 units from the exact source revision."""
    runs_root = Path(root).resolve() / "runs"
    if not runs_root.exists():
        return {}

    found: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        result_file = run_dir / "diag-backbone-freeze-unit.json"
        if not run_file.is_file():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "diag_backbone_freeze_trial_unit":
            continue
        inputs = run.get("inputs")
        if not isinstance(inputs, dict):
            continue
        provenance = inputs.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if provenance.get("cycle_id") != TRIAL_CYCLE_ID:
            continue
        if run.get("status") == "running":
            raise RuntimeError("A DIAG v1 unit is still marked running")
        if run.get("status") != "completed":
            continue
        if not result_file.is_file():
            raise RuntimeError("Completed DIAG v1 unit is missing its result")
        unit = _read_json(result_file)
        if unit.get("source_git_sha") != source_git_sha:
            raise RuntimeError(
                "Existing DIAG v1 evidence uses a different Git SHA; "
                "start a new Trial cycle after code changes"
            )
        family_id = _require_str(unit, "family_id")
        condition_id = _require_str(unit, "backbone_condition")
        key = (family_id, condition_id)
        if key in found:
            raise RuntimeError(f"Duplicate completed DIAG v1 unit: {key}")
        found[key] = (run_dir, unit)
    return found


def _checkpoint_rows(unit: dict[str, object]) -> dict[int, dict[str, object]]:
    rows = _require_list(unit, "checkpoints")
    if len(rows) != len(TRIAL_CHECKPOINTS):
        raise ValueError("DIAG v1 checkpoint count changed")
    by_step: dict[int, dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("DIAG v1 checkpoint must be an object")
        row = cast(dict[str, object], raw)
        step = _require_int(row, "step")
        by_step[step] = row
    if tuple(sorted(by_step)) != TRIAL_CHECKPOINTS:
        raise ValueError("DIAG v1 checkpoint steps changed")
    return by_step


def _sibling_margin(row: dict[str, object], arm: str) -> float:
    arm_row = _require_dict(row, arm)
    sibling = _require_dict(arm_row, "sibling")
    return _require_float(sibling, "margin_mean")


def _examples_digest(examples: Sequence[DecisionExample]) -> str:
    return digest(json_bytes([example.to_dict() for example in examples]))


def _validate_matched_training_data(
    full: dict[str, object],
    freeze: dict[str, object],
) -> None:
    full_digests = _require_dict(full, "training_examples_sha256")
    freeze_digests = _require_dict(freeze, "training_examples_sha256")
    for arm in ("replay", "correction"):
        if _require_str(full_digests, arm) != _require_str(freeze_digests, arm):
            raise ValueError(
                f"DIAG v1 causal factor changed {arm} training examples"
            )


def _validate_frozen_backbone_drift(unit: dict[str, object]) -> None:
    if unit.get("train_backbone") is not False:
        return
    checkpoints = _checkpoint_rows(unit)
    for step in TRIAL_CHECKPOINTS:
        for arm in ("replay", "correction"):
            arm_row = _require_dict(checkpoints[step], arm)
            drift = _require_dict(arm_row, "parameter_drift")
            if _require_float(drift, "backbone_l2") != 0.0:
                raise RuntimeError(
                    "Frozen-backbone DIAG v1 unit changed backbone parameters"
                )


def _row_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DIAG v1 row field {key!r} must be numeric")
    return float(value)


def _write_progress(
    campaign_path: Path,
    completed: dict[tuple[str, str], tuple[Path, dict[str, object]]],
) -> None:
    rows = [
        {
            "family_id": family_id,
            "backbone_condition": condition_id,
            "run_id": path.name,
        }
        for (family_id, condition_id), (path, _) in sorted(completed.items())
    ]
    _write_json(
        campaign_path / "progress.json",
        {
            "schema_version": "aster-diag-backbone-freeze-progress-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "expected_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "expected_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "completed_experiment_units": len(rows),
            "units": rows,
            "trial_endpoint_metrics_may_be_inspected": True,
        },
    )
