"""DIAG v0 Trial 0: checkpointed Correction/Replay interference curves."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
from statistics import median
from time import perf_counter
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
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.records.intervention import read_intervention_traces_jsonl
from aster.records.runlog import RunLog
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.runtime.loop import run_loop
from aster.runtime.service_agent import build_executor
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.decision import (
    DecisionTrainConfig,
    evaluate_decision_diagnostics,
    examples_from_teacher_trajectory,
    train_decision_with_checkpoints,
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

TRIAL_CYCLE_ID = "diag-correction-interference-trial-0"
TRIAL_FAMILY_IDS = (
    "learn-confirm-keyshift-03",
    "learn-confirm-keyshift-15",
    "learn-confirm-keyshift-21",
    "learn-confirm-keyshift-02",
    "learn-confirm-keyshift-12",
    "learn-confirm-keyshift-22",
)
TRIAL_SEEDS = (42, 43, 44)
TRIAL_CHECKPOINTS = (0, 10, 25, 50, 100)
TRIAL_CONFIG = DecisionTrainConfig(
    steps=100,
    learning_rate=3e-3,
    train_backbone=True,
    seed=42,
)
TRIAL_MAX_EPISODE_STEPS = 8
EXPECTED_TRIAL_UNITS = len(TRIAL_FAMILY_IDS) * len(TRIAL_SEEDS)


def run_diag_trial_unit(
    root: str | Path,
    *,
    family_id: str,
    seed: int,
    expected_source_git_sha: str | None = None,
) -> Path:
    """Run one post-hoc diagnostic family/seed trajectory with fixed checkpoints."""
    if family_id not in TRIAL_FAMILY_IDS:
        raise ValueError(f"Family is not selected for DIAG Trial 0: {family_id}")
    if seed not in TRIAL_SEEDS:
        raise ValueError(f"Seed is not selected for DIAG Trial 0: {seed}")

    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    if (
        expected_source_git_sha is not None
        and source_git_sha != expected_source_git_sha
    ):
        raise RuntimeError("Git revision changed during DIAG Trial 0")

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    family = _family_by_id(manifest, family_id)
    correction_task = _task_from_family(family, "correction_task")
    sibling_task = _task_from_family(family, "sibling_task")
    operation = _require_str(correction_task, "operation")

    artifact_id, artifact_path = resolve_act_parent_artifact(
        root_path, DEFAULT_ACT_RUN_ID
    )
    parent_model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved DIAG parent artifact identity mismatch")
    baseline = build_calculate_and_store_suite()
    _validate_parent_suite_lineage(parent_manifest, baseline)
    parent_model_id = _require_str(parent_manifest, "model_id")

    config = DecisionTrainConfig(
        steps=TRIAL_CONFIG.steps,
        learning_rate=TRIAL_CONFIG.learning_rate,
        train_backbone=TRIAL_CONFIG.train_backbone,
        seed=seed,
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
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "planned_units": EXPECTED_TRIAL_UNITS,
        "source_git_sha": source_git_sha,
        "parent_artifact_id": artifact_id,
        "act_run_id": DEFAULT_ACT_RUN_ID,
        "family_id": family_id,
        "operation": operation,
        "seed": seed,
    }
    run = RunLog(
        root_path,
        "diag_correction_interference_trial_unit",
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
            teacher_id="rule-v0:diag-correction-interference-trial-0",
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=RuntimeContext(task=correction_task),
            max_steps=TRIAL_MAX_EPISODE_STEPS,
        )
        traces = read_intervention_traces_jsonl(
            rollout_path / "interventions.jsonl"
        )
        if not traces:
            raise ValueError("DIAG correction rollout produced no intervention traces")
        if any(trace.route != "model" for trace in traces):
            raise ValueError("DIAG correction rollout must remain model-only")
        if any(trace.executed_action != trace.student_action for trace in traces):
            raise ValueError("DIAG model-only rollout must execute the student Action")

        correction_examples = examples_from_interventions(traces)
        if not correction_examples:
            raise ValueError("DIAG correction rollout produced no trainable labels")
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
            raise RuntimeError("DIAG Replay/Correction training lengths differ")

        sibling_examples = tuple(_teacher_examples(sibling_task))
        replay_candidate = deepcopy(parent_model)
        correction_candidate = deepcopy(parent_model)

        replay_started = perf_counter()
        replay_losses, replay_snapshots = train_decision_with_checkpoints(
            replay_candidate,
            tokenizer,
            replay_training,
            checkpoints=TRIAL_CHECKPOINTS,
            config=config,
        )
        replay_wall_seconds = perf_counter() - replay_started

        correction_started = perf_counter()
        correction_losses, correction_snapshots = train_decision_with_checkpoints(
            correction_candidate,
            tokenizer,
            correction_training,
            checkpoints=TRIAL_CHECKPOINTS,
            config=config,
        )
        correction_wall_seconds = perf_counter() - correction_started

        if len(replay_losses) != len(correction_losses):
            raise RuntimeError("DIAG Replay/Correction optimizer-step counts differ")

        resources = {
            "replay": {
                **_training_token_budget(
                    tokenizer, replay_training, config.steps
                ),
                "optimizer_steps": config.steps,
                "training_examples": len(replay_training),
                "wall_seconds_including_checkpoint_snapshots": replay_wall_seconds,
            },
            "correction": {
                **_training_token_budget(
                    tokenizer, correction_training, config.steps
                ),
                "optimizer_steps": config.steps,
                "training_examples": len(correction_training),
                "wall_seconds_including_checkpoint_snapshots": correction_wall_seconds,
            },
        }
        _write_json(
            run.path / "training-replay.json",
            {
                "schema_version": "aster-diag-interference-training-0",
                "arm": "replay",
                "config": asdict(config),
                "losses": replay_losses,
                "resources": resources["replay"],
            },
        )
        _write_json(
            run.path / "training-correction.json",
            {
                "schema_version": "aster-diag-interference-training-0",
                "arm": "correction",
                "config": asdict(config),
                "losses": correction_losses,
                "resources": resources["correction"],
            },
        )

        checkpoints = []
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
                    ),
                    "correction": _checkpoint_metrics(
                        parent_model,
                        correction_snapshots[step],
                        tokenizer,
                        repair_examples=correction_examples,
                        sibling_examples=sibling_examples,
                        base_examples=base_examples,
                    ),
                }
            )

        unit = {
            "schema_version": "aster-diag-interference-unit-0",
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "source_git_sha": source_git_sha,
            "run_id": run.id,
            "rollout_run_id": rollout_path.name,
            "parent_artifact_id": artifact_id,
            "family_id": family_id,
            "operation": operation,
            "seed": seed,
            "independent_training_trajectory": True,
            "checkpoint_observations_are_independent": False,
            "checkpoints": checkpoints,
            "resources": resources,
        }
        unit["trajectory_summary"] = summarize_diag_trial_unit(unit)
        _write_json(run.path / "diag-interference-unit.json", unit)
        run.finish(
            "completed",
            result="diag-interference-unit.json",
            rollout_run_id=rollout_path.name,
            family_id=family_id,
            seed=seed,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def summarize_diag_trial_unit(unit: dict[str, object]) -> dict[str, object]:
    """Reduce one repeated-measures trajectory without treating checkpoints as samples."""
    checkpoints = _require_list(unit, "checkpoints")
    if len(checkpoints) != len(TRIAL_CHECKPOINTS):
        raise ValueError("DIAG unit checkpoint count changed")
    by_step: dict[int, dict[str, object]] = {}
    for raw in checkpoints:
        if not isinstance(raw, dict):
            raise ValueError("DIAG checkpoint must be an object")
        row = cast(dict[str, object], raw)
        step = _require_int(row, "step")
        by_step[step] = row
    if tuple(sorted(by_step)) != TRIAL_CHECKPOINTS:
        raise ValueError("DIAG checkpoint steps changed")

    return {
        arm: _summarize_arm(by_step, arm)
        for arm in ("replay", "correction")
    }


def aggregate_diag_trial_units(
    units: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build a descriptive Trial result; no confirmatory p-value or verdict."""
    if len(units) != EXPECTED_TRIAL_UNITS:
        return {
            "schema_version": "aster-diag-interference-result-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "status": "trial_incomplete",
            "expected_training_trajectories": EXPECTED_TRIAL_UNITS,
            "completed_training_trajectories": len(units),
            "checkpoint_observations_are_independent": False,
            "confirmatory_verdict": None,
        }

    keys: set[tuple[str, int]] = set()
    summaries: list[dict[str, object]] = []
    source_git_sha: str | None = None
    for unit in units:
        if unit.get("cycle_id") != TRIAL_CYCLE_ID:
            raise ValueError("Mixed DIAG cycle IDs")
        family_id = _require_str(unit, "family_id")
        seed = _require_int(unit, "seed")
        key = (family_id, seed)
        if key in keys:
            raise ValueError(f"Duplicate DIAG Trial unit: {family_id} seed {seed}")
        keys.add(key)
        unit_sha = _require_str(unit, "source_git_sha")
        if source_git_sha is None:
            source_git_sha = unit_sha
        elif unit_sha != source_git_sha:
            raise ValueError("DIAG Trial units were produced by different Git SHAs")
        trajectory_summary = _require_dict(unit, "trajectory_summary")
        summaries.append(
            {
                "family_id": family_id,
                "operation": _require_str(unit, "operation"),
                "seed": seed,
                "summary": trajectory_summary,
            }
        )

    expected_keys = {
        (family_id, seed)
        for family_id in TRIAL_FAMILY_IDS
        for seed in TRIAL_SEEDS
    }
    if keys != expected_keys:
        raise ValueError("DIAG Trial completed with an unexpected family/seed set")

    operation_summary: dict[str, object] = {}
    for operation in ("add", "subtract"):
        rows = [
            row for row in summaries if row["operation"] == operation
        ]
        operation_summary[operation] = {
            "training_trajectories": len(rows),
            "correction_repair_accuracy_delta_median": median(
                _summary_float(row, "correction", "repair_accuracy_delta")
                for row in rows
            ),
            "correction_sibling_margin_delta_median": median(
                _summary_float(row, "correction", "sibling_margin_delta")
                for row in rows
            ),
            "correction_base_nll_delta_median": median(
                _summary_float(row, "correction", "base_nll_delta")
                for row in rows
            ),
        }

    return {
        "schema_version": "aster-diag-interference-result-0",
        "cycle_id": TRIAL_CYCLE_ID,
        "status": "complete",
        "experiment_mode": "trial",
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "source_git_sha": source_git_sha,
        "independent_training_trajectories": EXPECTED_TRIAL_UNITS,
        "checkpoint_observations": EXPECTED_TRIAL_UNITS * len(TRIAL_CHECKPOINTS),
        "checkpoint_observations_are_independent": False,
        "confirmatory_verdict": None,
        "trajectory_summaries": summaries,
        "operation_summary": operation_summary,
    }


def run_diag_trial_campaign(root: str | Path) -> Path:
    """Run or resume the fixed 18-unit DIAG Trial 0 campaign."""
    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    completed = discover_completed_diag_trial_units(
        root_path, source_git_sha=source_git_sha
    )
    campaign = RunLog(
        root_path,
        "diag_correction_interference_trial_campaign",
        {
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "planned_units": EXPECTED_TRIAL_UNITS,
            "source_git_sha": source_git_sha,
            "families": list(TRIAL_FAMILY_IDS),
            "seeds": list(TRIAL_SEEDS),
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "completed_units_at_start": len(completed),
        },
        producer="trainer",
    )
    try:
        for family_id in TRIAL_FAMILY_IDS:
            for seed in TRIAL_SEEDS:
                key = (family_id, seed)
                if key not in completed:
                    path = run_diag_trial_unit(
                        root_path,
                        family_id=family_id,
                        seed=seed,
                        expected_source_git_sha=source_git_sha,
                    )
                    unit = _read_json(path / "diag-interference-unit.json")
                    completed[key] = (path, unit)
                campaign.event(
                    "unit_available",
                    {
                        "family_id": family_id,
                        "seed": seed,
                        "run_id": completed[key][0].name,
                    },
                )
                _write_progress(campaign.path, completed)

        result = aggregate_diag_trial_units(
            [unit for _, unit in completed.values()]
        )
        if result.get("status") != "complete":
            raise RuntimeError("DIAG Trial 0 ended without all fixed units")
        _write_json(campaign.path / "diag-interference-results.json", result)
        campaign.finish(
            "completed",
            results="diag-interference-results.json",
            completed_training_trajectories=len(completed),
        )
        return campaign.path
    except BaseException as error:
        campaign.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def discover_completed_diag_trial_units(
    root: str | Path,
    *,
    source_git_sha: str,
) -> dict[tuple[str, int], tuple[Path, dict[str, object]]]:
    """Reuse completed Trial units from the same source revision."""
    runs_root = Path(root).resolve() / "runs"
    if not runs_root.exists():
        return {}

    found: dict[tuple[str, int], tuple[Path, dict[str, object]]] = {}
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        result_file = run_dir / "diag-interference-unit.json"
        if not run_file.is_file():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "diag_correction_interference_trial_unit":
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
            raise RuntimeError("A DIAG Trial unit is still marked running")
        if run.get("status") != "completed":
            continue
        if not result_file.is_file():
            raise RuntimeError("Completed DIAG Trial unit is missing its result")
        unit = _read_json(result_file)
        if unit.get("source_git_sha") != source_git_sha:
            raise RuntimeError(
                "Existing DIAG Trial evidence uses a different Git SHA; "
                "start a new Trial cycle after code changes"
            )
        family_id = _require_str(unit, "family_id")
        seed = _require_int(unit, "seed")
        key = (family_id, seed)
        if key in found:
            raise RuntimeError(f"Duplicate completed DIAG Trial unit: {key}")
        found[key] = (run_dir, unit)
    return found


def model_parameter_l2_drift(
    parent: DecisionModel,
    candidate: DecisionModel,
) -> dict[str, float]:
    """Measure head/backbone parameter movement from one common parent."""
    parent_parameters = dict(parent.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if parent_parameters.keys() != candidate_parameters.keys():
        raise ValueError("Parent/candidate parameter structures differ")

    sums = {"head": 0.0, "backbone": 0.0}
    for name, parent_parameter in parent_parameters.items():
        if name.startswith("head."):
            group = "head"
        elif name.startswith("backbone."):
            group = "backbone"
        else:
            raise ValueError(f"Unexpected DecisionModel parameter group: {name}")
        delta = (
            candidate_parameters[name].detach().float()
            - parent_parameter.detach().float()
        )
        sums[group] += float(torch.sum(delta * delta).item())
    return {
        "head_l2": math.sqrt(sums["head"]),
        "backbone_l2": math.sqrt(sums["backbone"]),
    }


def _checkpoint_metrics(
    parent: DecisionModel,
    candidate: DecisionModel,
    tokenizer: AsterTokenizer,
    *,
    repair_examples: Sequence[DecisionExample],
    sibling_examples: Sequence[DecisionExample],
    base_examples: Sequence[DecisionExample],
) -> dict[str, object]:
    return {
        "repair": evaluate_decision_diagnostics(
            candidate, tokenizer, repair_examples
        ),
        "sibling": evaluate_decision_diagnostics(
            candidate, tokenizer, sibling_examples
        ),
        "base_retention": evaluate_decision_diagnostics(
            candidate, tokenizer, base_examples
        ),
        "parameter_drift": model_parameter_l2_drift(parent, candidate),
    }


def _summarize_arm(
    by_step: dict[int, dict[str, object]],
    arm: str,
) -> dict[str, object]:
    start = _require_dict(by_step[0], arm)
    final = _require_dict(by_step[100], arm)
    start_repair = _require_dict(start, "repair")
    final_repair = _require_dict(final, "repair")
    start_sibling = _require_dict(start, "sibling")
    final_sibling = _require_dict(final, "sibling")
    start_base = _require_dict(start, "base_retention")
    final_base = _require_dict(final, "base_retention")

    sibling_margins = [
        (
            step,
            _require_float(
                _require_dict(_require_dict(by_step[step], arm), "sibling"),
                "margin_mean",
            ),
        )
        for step in TRIAL_CHECKPOINTS
    ]
    first_negative = next(
        (step for step, margin in sibling_margins if margin < 0.0),
        None,
    )
    final_drift = _require_dict(final, "parameter_drift")
    return {
        "repair_accuracy_delta": _require_float(final_repair, "accuracy")
        - _require_float(start_repair, "accuracy"),
        "repair_nll_delta": _require_float(final_repair, "nll")
        - _require_float(start_repair, "nll"),
        "sibling_accuracy_delta": _require_float(final_sibling, "accuracy")
        - _require_float(start_sibling, "accuracy"),
        "sibling_nll_delta": _require_float(final_sibling, "nll")
        - _require_float(start_sibling, "nll"),
        "sibling_margin_delta": _require_float(final_sibling, "margin_mean")
        - _require_float(start_sibling, "margin_mean"),
        "minimum_sibling_margin": min(margin for _, margin in sibling_margins),
        "first_negative_sibling_margin_step": first_negative,
        "base_accuracy_delta": _require_float(final_base, "accuracy")
        - _require_float(start_base, "accuracy"),
        "base_nll_delta": _require_float(final_base, "nll")
        - _require_float(start_base, "nll"),
        "head_l2_step100": _require_float(final_drift, "head_l2"),
        "backbone_l2_step100": _require_float(final_drift, "backbone_l2"),
    }


def _summary_float(
    row: dict[str, object],
    arm: str,
    metric: str,
) -> float:
    summary = row.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("DIAG trajectory summary is invalid")
    arm_summary = summary.get(arm)
    if not isinstance(arm_summary, dict):
        raise ValueError("DIAG arm summary is invalid")
    value = arm_summary.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DIAG summary metric is not numeric: {metric}")
    return float(value)


def _training_token_budget(
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
    steps: int,
) -> dict[str, int]:
    encoded_tokens = 0
    padded_positions = 0
    candidate_sequences = 0
    for step in range(steps):
        example = examples[step % len(examples)]
        lengths = [
            len(
                tokenizer.encode(
                    serialize_decision_input(
                        example.state, example.trajectory, candidate
                    ),
                    add_bos=True,
                    add_eos=True,
                )
            )
            for candidate in example.candidates
        ]
        if not lengths:
            raise ValueError("Decision training example has no candidates")
        encoded_tokens += sum(lengths)
        padded_positions += len(lengths) * max(lengths)
        candidate_sequences += len(lengths)
    return {
        "encoded_tokens_presented": encoded_tokens,
        "padded_token_positions": padded_positions,
        "candidate_sequences_presented": candidate_sequences,
    }


def _teacher_examples(task: dict[str, JsonValue]) -> list[DecisionExample]:
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )
    return examples_from_teacher_trajectory(
        trajectory,
        teacher="rule-v0:diag-interference-benchmark",
    )


def _family_by_id(
    manifest: dict[str, object],
    family_id: str,
) -> dict[str, object]:
    design = _require_dict(manifest, "family_design")
    families = _require_list(design, "families")
    matches = [
        cast(dict[str, object], family)
        for family in families
        if isinstance(family, dict) and family.get("family_id") == family_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous DIAG family: {family_id}")
    return matches[0]


def _task_from_family(
    family: dict[str, object],
    key: str,
) -> dict[str, JsonValue]:
    task = _require_dict(family, key)
    return cast(dict[str, JsonValue], dict(task))


def _validate_parent_suite_lineage(
    parent_manifest: dict[str, object],
    baseline,
) -> None:
    expected = digest(json_bytes(baseline.to_dict()))
    if parent_manifest.get("suite_id") != baseline.suite_id:
        raise ValueError("DIAG parent suite ID mismatch")
    if parent_manifest.get("suite_sha256") != expected:
        raise ValueError("DIAG parent suite digest mismatch")


def _clean_git_sha(root: Path) -> str:
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ValueError("DIAG Trial measurement requires a clean Git worktree")
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("DIAG Trial requires a readable Git revision") from error
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("DIAG source Git SHA is invalid")
    return revision


def _write_progress(
    campaign_path: Path,
    completed: dict[tuple[str, int], tuple[Path, dict[str, object]]],
) -> None:
    rows = [
        {
            "family_id": family_id,
            "seed": seed,
            "run_id": path.name,
        }
        for (family_id, seed), (path, _) in sorted(completed.items())
    ]
    _write_json(
        campaign_path / "progress.json",
        {
            "schema_version": "aster-diag-interference-progress-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "expected_training_trajectories": EXPECTED_TRIAL_UNITS,
            "completed_training_trajectories": len(rows),
            "units": rows,
            "trial_endpoint_metrics_may_be_inspected": True,
        },
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"DIAG field {key!r} must be an object")
    return cast(dict[str, object], value)


def _require_list(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"DIAG field {key!r} must be a list")
    return cast(list[object], value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"DIAG field {key!r} must be a non-empty string")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"DIAG field {key!r} must be an integer")
    return cast(int, value)


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DIAG field {key!r} must be numeric")
    return float(value)
