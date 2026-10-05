"""DIAG v2: last-block unfreeze stability/plasticity causal Trial."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from statistics import median
from typing import Sequence, cast

import torch

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_artifact import load_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.intervention import read_intervention_traces_jsonl
from aster.records.runlog import RunLog
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.runtime.service_agent import build_executor
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.decision import (
    DecisionTrainConfig,
    evaluate_decision_diagnostics,
    train_decision_with_checkpoints,
)
from aster.training.diag_backbone_freeze import summarize_backbone_freeze_unit
from aster.training.diag_interference import (
    TRIAL_CHECKPOINTS,
    _checkpoint_resource_budget,
    _clean_git_sha,
    _family_by_id,
    _read_json,
    _require_dict,
    _require_float,
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


TRIAL_CYCLE_ID = "diag-update-depth-trial-0"
TRIAL_PARENT_CYCLE_ID = "diag-backbone-freeze-trial-0"
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
UPDATE_DEPTHS = ("head-only", "last-block", "full")
TRIAL_SEED = 42
TRIAL_STEPS = 100
TRIAL_LEARNING_RATE = 3e-3
TRIAL_MAX_EPISODE_STEPS = 8


EXPECTED_FAMILY_BLOCKS = len(TRIAL_FAMILY_IDS)
EXPECTED_UPDATE_DEPTH_CONDITIONS = len(UPDATE_DEPTHS)
EXPECTED_EXPERIMENT_UNITS = EXPECTED_FAMILY_BLOCKS * EXPECTED_UPDATE_DEPTH_CONDITIONS
EXPECTED_ARM_TRAJECTORIES = EXPECTED_EXPERIMENT_UNITS * 2
EXPECTED_UNIT_CHECKPOINT_RECORDS = EXPECTED_EXPERIMENT_UNITS * len(TRIAL_CHECKPOINTS)
EXPECTED_ARM_CHECKPOINT_RECORDS = EXPECTED_UNIT_CHECKPOINT_RECORDS * 2


def _parameter_group(name: str) -> str:
    if name.startswith(("backbone.token_embedding.", "backbone.position_embedding.")):
        return "embeddings"
    block_prefix = "backbone.blocks."
    if name.startswith(block_prefix):
        suffix = name[len(block_prefix):]
        index_text, separator, _ = suffix.partition(".")
        if separator and index_text.isdigit():
            return f"block_{int(index_text)}"
    if name.startswith("backbone.final_norm."):
        return "final_norm"
    if name.startswith("backbone.lm_head."):
        return "lm_head"
    if name.startswith("head."):
        return "decision_head"
    raise ValueError(f"Unexpected DecisionModel parameter: {name}")


def parameter_groups(model: DecisionModel) -> tuple[str, ...]:
    """Return the parameter groups that actually exist in one saved model."""
    block_count = len(model.backbone.blocks)
    if block_count < 1:
        raise ValueError("DIAG v2 requires at least one Transformer block")
    return (
        "embeddings",
        *(f"block_{index}" for index in range(block_count)),
        "final_norm",
        "lm_head",
        "decision_head",
    )


def _trainable_groups(model: DecisionModel, update_depth: str) -> tuple[str, ...]:
    groups = parameter_groups(model)
    block_groups = tuple(group for group in groups if group.startswith("block_"))
    last_block_group = block_groups[-1]

    if update_depth == "head-only":
        return ("decision_head",)
    if update_depth == "last-block":
        return (last_block_group, "final_norm", "decision_head")
    if update_depth == "full":
        return (
            "embeddings",
            *block_groups,
            "final_norm",
            "decision_head",
        )
    raise ValueError(f"Unknown DIAG v2 update depth: {update_depth}")


def trainability_spec(model: DecisionModel, update_depth: str) -> dict[str, object]:
    """Return the exact parameter mask used by one DIAG v2 condition."""
    if update_depth not in UPDATE_DEPTHS:
        raise ValueError(f"Unknown DIAG v2 update depth: {update_depth}")

    groups = parameter_groups(model)
    block_count = len(model.backbone.blocks)
    last_block_index = block_count - 1
    last_block_group = f"block_{last_block_index}"
    trainable_groups = _trainable_groups(model, update_depth)
    trainable_names = tuple(
        name
        for name, _ in model.named_parameters()
        if _parameter_group(name) in trainable_groups
    )
    if not trainable_names:
        raise RuntimeError("DIAG v2 selected no trainable parameters")

    all_names = tuple(name for name, _ in model.named_parameters())
    frozen_names = tuple(name for name in all_names if name not in trainable_names)
    frozen_groups = tuple(group for group in groups if group not in trainable_groups)

    return {
        "update_depth": update_depth,
        "backbone_layers": block_count,
        "last_block_index": last_block_index,
        "last_block_group": last_block_group,
        "parameter_groups": list(groups),
        "trainable_groups": list(trainable_groups),
        "frozen_groups": list(frozen_groups),
        "trainable_parameter_names": list(trainable_names),
        "frozen_parameter_names": list(frozen_names),
        "trainable_parameter_names_sha256": digest(json_bytes(list(trainable_names))),
        "lm_head_in_decision_forward_path": False,
    }


def component_parameter_l2_drift(
    parent: DecisionModel,
    candidate: DecisionModel,
) -> dict[str, float]:
    """Measure parameter movement by model component plus legacy aggregates."""
    parent_parameters = dict(parent.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if parent_parameters.keys() != candidate_parameters.keys():
        raise ValueError("Parent/candidate parameter structures differ")

    groups = parameter_groups(parent)
    sums = {group: 0.0 for group in groups}
    for name, parent_parameter in parent_parameters.items():
        group = _parameter_group(name)
        if group not in sums:
            raise ValueError(f"Candidate parameter group is absent from parent: {group}")
        delta = (
            candidate_parameters[name].detach().float()
            - parent_parameter.detach().float()
        )
        sums[group] += float(torch.sum(delta * delta).item())

    result = {f"{group}_l2": math.sqrt(sums[group]) for group in groups}
    result["head_l2"] = result["decision_head_l2"]
    result["backbone_l2"] = math.sqrt(
        sum(sums[group] for group in groups if group != "decision_head")
    )
    return result


def _checkpoint_metrics_v2(
    parent: DecisionModel,
    candidate: DecisionModel,
    tokenizer: AsterTokenizer,
    *,
    repair_examples: Sequence[DecisionExample],
    sibling_examples: Sequence[DecisionExample],
    base_examples: Sequence[DecisionExample],
    resources: dict[str, float | int],
) -> dict[str, object]:
    return {
        "repair": evaluate_decision_diagnostics(candidate, tokenizer, repair_examples),
        "sibling": evaluate_decision_diagnostics(candidate, tokenizer, sibling_examples),
        "base_retention": evaluate_decision_diagnostics(candidate, tokenizer, base_examples),
        "parameter_drift": component_parameter_l2_drift(parent, candidate),
        "resources": resources,
    }


def _examples_digest(examples: Sequence[DecisionExample]) -> str:
    return digest(json_bytes([example.to_dict() for example in examples]))


def run_update_depth_unit(
    root: str | Path,
    *,
    family_id: str,
    update_depth: str,
    expected_source_git_sha: str | None = None,
) -> Path:
    """Run one family x update-depth unit with matched Replay/Correction arms."""
    if family_id not in TRIAL_FAMILY_IDS:
        raise ValueError(f"Family is not selected for DIAG v2: {family_id}")
    if update_depth not in UPDATE_DEPTHS:
        raise ValueError(f"Unknown DIAG v2 update depth: {update_depth}")

    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    if expected_source_git_sha is not None and source_git_sha != expected_source_git_sha:
        raise RuntimeError("Git revision changed during DIAG v2")

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    family = _family_by_id(manifest, family_id)
    correction_task = _task_from_family(family, "correction_task")
    sibling_task = _task_from_family(family, "sibling_task")
    operation_value = correction_task.get("operation")
    if not isinstance(operation_value, str) or not operation_value:
        raise ValueError("DIAG v2 correction operation must be a non-empty string")
    operation = operation_value

    artifact_id, artifact_path = resolve_act_parent_artifact(root_path, DEFAULT_ACT_RUN_ID)
    parent_model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved DIAG v2 parent artifact identity mismatch")
    baseline = build_calculate_and_store_suite()
    _validate_parent_suite_lineage(parent_manifest, baseline)
    parent_model_id = _require_str(parent_manifest, "model_id")

    trainability = trainability_spec(parent_model, update_depth)
    trainable_names = tuple(cast(list[str], trainability["trainable_parameter_names"]))
    config = DecisionTrainConfig(
        steps=TRIAL_STEPS,
        learning_rate=TRIAL_LEARNING_RATE,
        train_backbone=True,
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
        "update_depth": update_depth,
    }
    run = RunLog(
        root_path,
        "diag_update_depth_trial_unit",
        {
            "provenance": provenance,
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "train_config": asdict(config),
            "trainability": trainability,
            "explicit_parameter_mask_overrides_train_backbone": True,
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
            teacher_id="rule-v0:diag-update-depth-trial-0",
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=RuntimeContext(task=correction_task),
            max_steps=TRIAL_MAX_EPISODE_STEPS,
        )
        traces = read_intervention_traces_jsonl(rollout_path / "interventions.jsonl")
        if not traces:
            raise ValueError("DIAG v2 correction rollout produced no intervention traces")
        if any(trace.route != "model" for trace in traces):
            raise ValueError("DIAG v2 correction rollout must remain model-only")
        if any(trace.executed_action != trace.student_action for trace in traces):
            raise ValueError("DIAG v2 model-only rollout must execute student Action")

        correction_examples = examples_from_interventions(traces)
        if not correction_examples:
            raise ValueError("DIAG v2 correction rollout produced no trainable labels")
        base_examples = tuple(baseline.training_examples())
        replay_examples = replay_decision_examples(base_examples, len(correction_examples))
        replay_training = aggregate_decision_examples(base_examples, replay_examples)
        correction_training = aggregate_decision_examples(base_examples, correction_examples)
        if len(replay_training) != len(correction_training):
            raise RuntimeError("DIAG v2 Replay/Correction training lengths differ")

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
            trainable_parameter_names=trainable_names,
        )
        correction_losses, correction_snapshots, correction_elapsed = (
            train_decision_with_checkpoints(
                correction_candidate,
                tokenizer,
                correction_training,
                checkpoints=TRIAL_CHECKPOINTS,
                config=config,
                trainable_parameter_names=trainable_names,
            )
        )
        if len(replay_losses) != len(correction_losses):
            raise RuntimeError("DIAG v2 Replay/Correction optimizer-step counts differ")

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
        for arm, losses in (("replay", replay_losses), ("correction", correction_losses)):
            _write_json(
                run.path / f"training-{arm}.json",
                {
                    "schema_version": "aster-diag-update-depth-training-0",
                    "arm": arm,
                    "update_depth": update_depth,
                    "config": asdict(config),
                    "explicit_parameter_mask_overrides_train_backbone": True,
                    "trainability": trainability,
                    "training_examples_sha256": training_digests[arm],
                    "losses": losses,
                    "resources": resources[arm],
                },
            )

        checkpoints: list[dict[str, object]] = []
        for step in TRIAL_CHECKPOINTS:
            checkpoints.append(
                {
                    "step": step,
                    "replay": _checkpoint_metrics_v2(
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
                    "correction": _checkpoint_metrics_v2(
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
            "schema_version": "aster-diag-update-depth-unit-0",
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
            "update_depth": update_depth,
            "independent_family_block": family_id,
            "update_depth_condition_is_independent_sample": False,
            "checkpoint_observations_are_independent": False,
            "trainability": trainability,
            "training_examples_sha256": training_digests,
            "checkpoints": checkpoints,
            "resources": resources,
        }
        unit["trajectory_summary"] = summarize_diag_trial_unit(unit)
        unit["condition_summary"] = summarize_backbone_freeze_unit(unit)
        _validate_frozen_group_drift(unit)
        _write_json(run.path / "diag-update-depth-unit.json", unit)
        run.finish(
            "completed",
            result="diag-update-depth-unit.json",
            rollout_run_id=rollout_path.name,
            family_id=family_id,
            update_depth=update_depth,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _validate_frozen_group_drift(unit: dict[str, object]) -> None:
    trainability = _require_dict(unit, "trainability")
    frozen_groups_raw = trainability.get("frozen_groups")
    if not isinstance(frozen_groups_raw, list) or not all(
        isinstance(group, str) for group in frozen_groups_raw
    ):
        raise ValueError("DIAG v2 frozen_groups must be a list of strings")
    frozen_groups = tuple(frozen_groups_raw)

    checkpoints_raw = unit.get("checkpoints")
    if not isinstance(checkpoints_raw, list):
        raise ValueError("DIAG v2 checkpoints must be a list")
    for row in checkpoints_raw:
        if not isinstance(row, dict):
            raise ValueError("DIAG v2 checkpoint must be an object")
        for arm in ("replay", "correction"):
            arm_row = _require_dict(row, arm)
            drift = _require_dict(arm_row, "parameter_drift")
            for group in frozen_groups:
                if _require_float(drift, f"{group}_l2") != 0.0:
                    raise RuntimeError(
                        f"Frozen DIAG v2 parameter group changed: {group}"
                    )


def _validate_matched_training_data(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> None:
    reference_digests = _require_dict(reference, "training_examples_sha256")
    candidate_digests = _require_dict(candidate, "training_examples_sha256")
    for arm in ("replay", "correction"):
        if _require_str(reference_digests, arm) != _require_str(candidate_digests, arm):
            raise ValueError(
                f"DIAG v2 causal factor changed {arm} training examples"
            )


def aggregate_update_depth_units(
    units: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate paired update-depth conditions without pseudo-replication."""
    if len(units) != EXPECTED_EXPERIMENT_UNITS:
        return {
            "schema_version": "aster-diag-update-depth-result-0",
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
            raise ValueError("Mixed DIAG v2 cycle IDs")
        family_id = _require_str(unit, "family_id")
        update_depth = _require_str(unit, "update_depth")
        if family_id not in TRIAL_FAMILY_IDS:
            raise ValueError(f"Unexpected DIAG v2 family: {family_id}")
        if update_depth not in UPDATE_DEPTHS:
            raise ValueError(f"Unexpected DIAG v2 update depth: {update_depth}")
        key = (family_id, update_depth)
        if key in indexed:
            raise ValueError(f"Duplicate DIAG v2 unit: {key}")
        unit_sha = _require_str(unit, "source_git_sha")
        if source_git_sha is None:
            source_git_sha = unit_sha
        elif unit_sha != source_git_sha:
            raise ValueError("DIAG v2 units were produced by different Git SHAs")
        indexed[key] = unit

    expected_keys = {
        (family_id, update_depth)
        for family_id in TRIAL_FAMILY_IDS
        for update_depth in UPDATE_DEPTHS
    }
    if set(indexed) != expected_keys:
        raise ValueError("DIAG v2 completed with unexpected family/condition set")

    family_summaries: list[dict[str, object]] = []
    for family_id in TRIAL_FAMILY_IDS:
        head = indexed[(family_id, "head-only")]
        last = indexed[(family_id, "last-block")]
        full = indexed[(family_id, "full")]
        _validate_matched_training_data(head, last)
        _validate_matched_training_data(head, full)

        condition_rows: dict[str, dict[str, float]] = {}
        for update_depth, unit in (
            ("head-only", head),
            ("last-block", last),
            ("full", full),
        ):
            summary = _require_dict(unit, "condition_summary")
            condition_rows[update_depth] = {
                "D_sibling_margin": _require_float(
                    summary, "sibling_margin_correction_minus_replay_delta"
                ),
                "Repair_CR": _require_float(
                    summary, "repair_accuracy_correction_minus_replay_delta"
                ),
            }

        d_head = condition_rows["head-only"]["D_sibling_margin"]
        d_last = condition_rows["last-block"]["D_sibling_margin"]
        d_full = condition_rows["full"]["D_sibling_margin"]
        repair_head = condition_rows["head-only"]["Repair_CR"]
        repair_last = condition_rows["last-block"]["Repair_CR"]
        repair_full = condition_rows["full"]["Repair_CR"]

        family_summaries.append(
            {
                "family_id": family_id,
                "family_stratum": _require_str(head, "family_stratum"),
                "operation": _require_str(head, "operation"),
                "conditions": condition_rows,
                "contrasts": {
                    "stability_gain_last_vs_full": d_last - d_full,
                    "stability_cost_last_vs_head": d_last - d_head,
                    "repair_gain_last_vs_head": repair_last - repair_head,
                    "repair_gap_last_to_full": repair_last - repair_full,
                },
                "training_examples_sha256": _require_dict(
                    head, "training_examples_sha256"
                ),
            }
        )

    loss_rows = [
        row for row in family_summaries if row["family_stratum"] == "loss"
    ]
    control_rows = [
        row for row in family_summaries if row["family_stratum"] == "control"
    ]

    def contrast_median(rows: list[dict[str, object]], key: str) -> float:
        values = []
        for row in rows:
            contrasts = _require_dict(row, "contrasts")
            values.append(_require_float(contrasts, key))
        return median(values)

    contrast_keys = (
        "stability_gain_last_vs_full",
        "stability_cost_last_vs_head",
        "repair_gain_last_vs_head",
        "repair_gap_last_to_full",
    )
    descriptive: dict[str, float] = {}
    for key in contrast_keys:
        descriptive[f"loss_{key}_median"] = contrast_median(loss_rows, key)
        descriptive[f"control_{key}_median"] = contrast_median(control_rows, key)

    return {
        "schema_version": "aster-diag-update-depth-result-0",
        "cycle_id": TRIAL_CYCLE_ID,
        "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
        "status": "complete",
        "experiment_mode": "trial",
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "source_git_sha": source_git_sha,
        "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
        "paired_update_depth_conditions_per_family": EXPECTED_UPDATE_DEPTH_CONDITIONS,
        "experiment_units": EXPECTED_EXPERIMENT_UNITS,
        "arm_training_trajectories": EXPECTED_ARM_TRAJECTORIES,
        "unit_checkpoint_records": EXPECTED_UNIT_CHECKPOINT_RECORDS,
        "arm_checkpoint_records": EXPECTED_ARM_CHECKPOINT_RECORDS,
        "update_depth_conditions_are_independent_samples": False,
        "checkpoint_observations_are_independent": False,
        "confirmatory_verdict": None,
        "primary_quantities": {
            "D": "Correction sibling-margin delta - Replay sibling-margin delta",
            "Repair_CR": "Correction repair-accuracy delta - Replay repair-accuracy delta",
        },
        "family_summaries": family_summaries,
        "descriptive_summary": descriptive,
    }


def run_update_depth_campaign(root: str | Path) -> Path:
    """Run or resume the fixed 4-family x 3-condition DIAG v2 campaign."""
    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    completed = discover_completed_update_depth_units(
        root_path, source_git_sha=source_git_sha
    )
    campaign = RunLog(
        root_path,
        "diag_update_depth_trial_campaign",
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
            "update_depths": list(UPDATE_DEPTHS),
            "seed": TRIAL_SEED,
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "completed_units_at_start": len(completed),
        },
        producer="trainer",
    )
    try:
        for family_id in TRIAL_FAMILY_IDS:
            for update_depth in UPDATE_DEPTHS:
                key = (family_id, update_depth)
                if key not in completed:
                    path = run_update_depth_unit(
                        root_path,
                        family_id=family_id,
                        update_depth=update_depth,
                        expected_source_git_sha=source_git_sha,
                    )
                    unit = _read_json(path / "diag-update-depth-unit.json")
                    completed[key] = (path, unit)
                campaign.event(
                    "unit_available",
                    {
                        "family_id": family_id,
                        "update_depth": update_depth,
                        "run_id": completed[key][0].name,
                    },
                )
                _write_progress(campaign.path, completed)

        result = aggregate_update_depth_units(
            [unit for _, unit in completed.values()]
        )
        if result.get("status") != "complete":
            raise RuntimeError("DIAG v2 ended without all fixed units")
        _write_json(campaign.path / "diag-update-depth-results.json", result)
        campaign.finish(
            "completed",
            results="diag-update-depth-results.json",
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


def discover_completed_update_depth_units(
    root: str | Path,
    *,
    source_git_sha: str,
) -> dict[tuple[str, str], tuple[Path, dict[str, object]]]:
    """Reuse only completed DIAG v2 units from the exact source revision."""
    runs_root = Path(root).resolve() / "runs"
    if not runs_root.exists():
        return {}

    found: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        result_file = run_dir / "diag-update-depth-unit.json"
        if not run_file.is_file():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "diag_update_depth_trial_unit":
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
            raise RuntimeError("A DIAG v2 unit is still marked running")
        if run.get("status") != "completed":
            continue
        if not result_file.is_file():
            raise RuntimeError("Completed DIAG v2 unit is missing its result")
        unit = _read_json(result_file)
        if unit.get("source_git_sha") != source_git_sha:
            raise RuntimeError(
                "Existing DIAG v2 evidence uses a different Git SHA; "
                "start a new Trial cycle after code changes"
            )
        family_id = _require_str(unit, "family_id")
        update_depth = _require_str(unit, "update_depth")
        key = (family_id, update_depth)
        if key in found:
            raise RuntimeError(f"Duplicate completed DIAG v2 unit: {key}")
        found[key] = (run_dir, unit)
    return found


def _write_progress(
    campaign_path: Path,
    completed: dict[tuple[str, str], tuple[Path, dict[str, object]]],
) -> None:
    rows = [
        {
            "family_id": family_id,
            "update_depth": update_depth,
            "run_id": path.name,
        }
        for (family_id, update_depth), (path, _) in sorted(completed.items())
    ]
    _write_json(
        campaign_path / "progress.json",
        {
            "schema_version": "aster-diag-update-depth-progress-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "expected_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "expected_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "completed_experiment_units": len(rows),
            "units": rows,
            "trial_endpoint_metrics_may_be_inspected": True,
        },
    )
