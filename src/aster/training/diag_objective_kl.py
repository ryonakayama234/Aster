"""DIAG v5: frozen-parent KL preservation objective Trial."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
from statistics import median
from typing import Sequence, cast

import torch
from torch import nn
from torch.nn import functional as F

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy, score_candidates
from aster.model.decision_artifact import load_decision_artifact
from aster.model.decision_head import (
    DecisionHead,
    DecisionModel,
    ResidualDecisionHead,
)
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
from aster.training.diag_update_depth import (
    component_parameter_l2_drift,
    trainability_spec,
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


TRIAL_CYCLE_ID = "diag-objective-kl-trial-0"
TRIAL_PARENT_CYCLE_ID = "diag-objective-kl-trial-0"
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
OBJECTIVES = ("ce", "ce-kl")
TRIAL_SEED = 42
TRIAL_STEPS = 100
TRIAL_LEARNING_RATE = 3e-3
TRIAL_MAX_EPISODE_STEPS = 8
KL_LAMBDA = 1.0
KL_TEMPERATURE = 1.0

EXPECTED_FAMILY_BLOCKS = len(TRIAL_FAMILY_IDS)
EXPECTED_OBJECTIVE_CONDITIONS = len(OBJECTIVES)
EXPECTED_EXPERIMENT_UNITS = EXPECTED_FAMILY_BLOCKS * EXPECTED_OBJECTIVE_CONDITIONS
EXPECTED_ARM_TRAJECTORIES = EXPECTED_EXPERIMENT_UNITS * 2
EXPECTED_UNIT_CHECKPOINT_RECORDS = EXPECTED_EXPERIMENT_UNITS * len(TRIAL_CHECKPOINTS)
EXPECTED_ARM_CHECKPOINT_RECORDS = EXPECTED_UNIT_CHECKPOINT_RECORDS * 2

_TOKENIZER_FILES = (
    "manifest.json",
    "vocab.json",
    "merges.json",
    "special_tokens.json",
)


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _decision_path_parameter_count(model: DecisionModel) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.lm_head.")
    )


def _module_state_digest(module: nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(str(cpu.dtype).encode("ascii"))
        value.update(b"\0")
        value.update(str(tuple(cpu.shape)).encode("ascii"))
        value.update(b"\0")
        value.update(bytes(cpu.view(torch.uint8).reshape(-1).tolist()))
        value.update(b"\0")
    return value.hexdigest()


def _tokenizer_artifact_digest(artifact_path: Path) -> str:
    tokenizer_dir = artifact_path / "tokenizer"
    value = hashlib.sha256()
    for name in _TOKENIZER_FILES:
        item = tokenizer_dir / name
        if not item.is_file() or item.is_symlink():
            raise RuntimeError(f"DIAG v5 tokenizer artifact file is missing: {name}")
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def objective_spec(
    parent: DecisionModel,
    model: DecisionModel,
    objective: str,
) -> dict[str, object]:
    """Verify that only the loss, not the objective, changes."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown DIAG v5 objective: {objective}")
    if not isinstance(parent.head, DecisionHead) or isinstance(
        parent.head, ResidualDecisionHead
    ):
        raise ValueError("DIAG v5 requires the exact linear-head ACT parent")
    if not isinstance(model.head, DecisionHead) or isinstance(
        model.head, ResidualDecisionHead
    ):
        raise ValueError("DIAG v5 requires a linear Decision Head")
    parent_parameters = _decision_path_parameter_count(parent)
    actual_parameters = _decision_path_parameter_count(model)
    if actual_parameters != parent_parameters:
        raise RuntimeError("DIAG v5 changes Decision-path parameter count")
    if _count_parameters(model.head) != _count_parameters(parent.head):
        raise RuntimeError("DIAG v5 changes trainable Head parameter count")
    trainability = trainability_spec(model, "head-only")
    names = tuple(cast(list[str], trainability["trainable_parameter_names"]))
    if not names or any(not name.startswith("head.") for name in names):
        raise RuntimeError("DIAG v5 must train only the Decision Head")
    return {
        "objective": objective,
        "kl_lambda": KL_LAMBDA if objective == "ce-kl" else 0.0,
        "kl_temperature": KL_TEMPERATURE,
        "backbone_width": model.backbone.config.width,
        "backbone_vocab_size": model.backbone.config.vocab_size,
        "parent_decision_path_parameter_count": parent_parameters,
        "decision_path_parameter_count": actual_parameters,
        "head_parameter_count": _count_parameters(model.head),
        "backbone_state_sha256": _module_state_digest(model.backbone),
        "trainability": trainability,
    }


def candidate_objective_components(
    student_scores: torch.Tensor,
    parent_scores: torch.Tensor,
    target_index: int,
    *,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, CE, forward KL) on one ordered candidate set."""
    if coefficient not in (0.0, 1.0):
        raise ValueError("DIAG v5 permits only the fixed coefficients 0/1")
    if student_scores.ndim != 1 or student_scores.numel() == 0:
        raise ValueError("Expected nonempty one-dimensional candidate scores")
    if student_scores.shape != parent_scores.shape:
        raise ValueError("Parent/student candidate ordering or count mismatch")
    if not 0 <= target_index < student_scores.numel():
        raise ValueError("Correct candidate index is out of range")
    target = torch.tensor([target_index], device=student_scores.device)
    ce = F.cross_entropy(student_scores.unsqueeze(0), target)
    q = torch.softmax(parent_scores.detach() / KL_TEMPERATURE, dim=0)
    log_p = F.log_softmax(student_scores / KL_TEMPERATURE, dim=0)
    kl = F.kl_div(log_p, q, reduction="sum")
    return ce + coefficient * kl, ce, kl


class ObjectiveLossRecorder:
    """Track CE and KL separately without changing the common AdamW loop."""

    def __init__(self, parent: DecisionModel, objective: str):
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown DIAG v5 objective: {objective}")
        self.parent = deepcopy(parent).eval()
        self.parent.requires_grad_(False)
        self.coefficient = KL_LAMBDA if objective == "ce-kl" else 0.0
        self.ce_losses: list[float] = []
        self.kl_losses: list[float] = []

    def __call__(
        self,
        model: DecisionModel,
        tokenizer: AsterTokenizer,
        example: DecisionExample,
    ) -> torch.Tensor:
        student_scores = score_candidates(
            model, tokenizer, example.state, example.trajectory, example.candidates
        )
        with torch.no_grad():
            parent_scores = score_candidates(
                self.parent, tokenizer, example.state,
                example.trajectory, example.candidates
            )
        total, ce, kl = candidate_objective_components(
            student_scores, parent_scores, example.target_index,
            coefficient=self.coefficient,
        )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("DIAG v5 objective is non-finite")
        self.ce_losses.append(float(ce.detach().item()))
        self.kl_losses.append(float(kl.detach().item()))
        return total


def assert_initial_objective_equivalence(
    parent: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
) -> dict[str, object]:
    """At the exact parent, KL is zero and CE and CE+KL head gradients agree."""
    if not examples:
        raise ValueError("DIAG v5 needs a nonempty training set")
    baseline = deepcopy(parent).eval()
    baseline.backbone.requires_grad_(False)
    first = examples[0]
    scores = score_candidates(
        baseline, tokenizer, first.state, first.trajectory, first.candidates
    )
    with torch.no_grad():
        teacher_scores = score_candidates(
            parent.eval(), tokenizer, first.state, first.trajectory, first.candidates
        )
    if not torch.equal(scores, teacher_scores):
        raise RuntimeError("DIAG v5 parent and step0 student scores differ")
    loss_ce, _, kl = candidate_objective_components(
        scores, teacher_scores, first.target_index, coefficient=0.0
    )
    loss_kl, _, _ = candidate_objective_components(
        scores, teacher_scores, first.target_index, coefficient=KL_LAMBDA
    )
    trainable = tuple(baseline.head.parameters())
    grad_ce = torch.autograd.grad(loss_ce, trainable, retain_graph=True)
    grad_kl = torch.autograd.grad(loss_kl, trainable)
    max_difference = max(
        float((a-b).abs().max().item()) for a, b in zip(grad_ce, grad_kl)
    )
    if abs(float(kl.item())) > 1e-6 or max_difference > 1e-6:
        raise RuntimeError("DIAG v5 step0 KL or CE gradient identity failed")
    return {
        "first_training_example_only": True,
        "step0_kl": float(kl.item()),
        "step0_head_gradient_max_abs_difference": max_difference,
        "gradient_identity_within_1e_6": True,
    }

def assert_step0_equivalence(
    control: DecisionModel,
    treatment: DecisionModel,
    tokenizer: AsterTokenizer,
    examples: Sequence[DecisionExample],
) -> dict[str, object]:
    """Require identical candidate scores at step 0 for both objectives."""
    control.eval()
    treatment.eval()
    decisions = 0
    candidates = 0
    with torch.no_grad():
        for example in examples:
            a = score_candidates(
                control, tokenizer, example.state, example.trajectory, example.candidates
            )
            b = score_candidates(
                treatment, tokenizer, example.state, example.trajectory, example.candidates
            )
            if not torch.equal(a, b):
                raise RuntimeError("DIAG v5 step-0 candidate scores differ")
            decisions += 1
            candidates += a.numel()
    return {
        "exact_candidate_score_identity": True,
        "exact_argmax_identity": True,
        "decision_examples_checked": decisions,
        "candidate_scores_checked": candidates,
        "max_abs_score_difference": 0.0,
    }

def _examples_digest(examples: Sequence[DecisionExample]) -> str:
    return digest(json_bytes([example.to_dict() for example in examples]))


def _checkpoint_metrics(
    baseline: DecisionModel,
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
        "parameter_drift": component_parameter_l2_drift(baseline, candidate),
        "backbone_state_sha256": _module_state_digest(candidate.backbone),
        "resources": resources,
    }


def _validate_unit_invariants(unit: dict[str, object]) -> None:
    equivalence = _require_dict(unit, "step0_equivalence")
    if equivalence.get("exact_candidate_score_identity") is not True:
        raise RuntimeError("DIAG v5 lacks exact step-0 score identity")
    if equivalence.get("exact_argmax_identity") is not True:
        raise RuntimeError("DIAG v5 lacks exact step-0 argmax identity")
    if _require_float(equivalence, "max_abs_score_difference") != 0.0:
        raise RuntimeError("DIAG v5 step-0 score difference is nonzero")

    tokenizer = _require_dict(unit, "tokenizer_identity")
    before = _require_str(tokenizer, "before_sha256")
    after = _require_str(tokenizer, "after_sha256")
    manifest = _require_str(tokenizer, "manifest_sha256")
    if before != after or before != manifest:
        raise RuntimeError("DIAG v5 tokenizer artifact changed")

    objective = _require_dict(unit, "objective_spec")
    baseline_backbone = _require_str(objective, "backbone_state_sha256")
    checkpoints = unit.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("DIAG v5 checkpoints must be a list")
    for row in checkpoints:
        if not isinstance(row, dict):
            raise ValueError("DIAG v5 checkpoint must be an object")
        for arm in ("replay", "correction"):
            arm_row = _require_dict(row, arm)
            drift = _require_dict(arm_row, "parameter_drift")
            if _require_float(drift, "backbone_l2") != 0.0:
                raise RuntimeError("DIAG v5 frozen backbone parameters changed")
            if _require_str(arm_row, "backbone_state_sha256") != baseline_backbone:
                raise RuntimeError("DIAG v5 frozen backbone digest changed")


def run_objective_kl_unit(
    root: str | Path,
    *,
    family_id: str,
    objective: str,
    expected_source_git_sha: str | None = None,
) -> Path:
    """Run one family x objective unit with matched Replay/Correction arms."""
    if family_id not in TRIAL_FAMILY_IDS:
        raise ValueError(f"Family is not selected for DIAG v5: {family_id}")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown DIAG v5 objective: {objective}")

    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    if expected_source_git_sha is not None and source_git_sha != expected_source_git_sha:
        raise RuntimeError("Git revision changed during DIAG v5")

    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    family = _family_by_id(manifest, family_id)
    correction_task = _task_from_family(family, "correction_task")
    sibling_task = _task_from_family(family, "sibling_task")
    operation_value = correction_task.get("operation")
    if not isinstance(operation_value, str) or not operation_value:
        raise ValueError("DIAG v5 correction operation must be a non-empty string")
    operation = operation_value

    artifact_id, artifact_path = resolve_act_parent_artifact(root_path, DEFAULT_ACT_RUN_ID)
    parent_model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved DIAG v5 parent artifact identity mismatch")
    baseline_suite = build_calculate_and_store_suite()
    _validate_parent_suite_lineage(parent_manifest, baseline_suite)
    parent_model_id = _require_str(parent_manifest, "model_id")
    manifest_tokenizer_sha = _require_str(parent_manifest, "tokenizer_sha256")
    tokenizer_before_sha = _tokenizer_artifact_digest(artifact_path)
    if tokenizer_before_sha != manifest_tokenizer_sha:
        raise RuntimeError("DIAG v5 tokenizer digest disagrees with verified parent manifest")

    config = DecisionTrainConfig(
        steps=TRIAL_STEPS,
        learning_rate=TRIAL_LEARNING_RATE,
        train_backbone=False,
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
        "causal_factor": "training_objective",
        "planned_experiment_units": EXPECTED_EXPERIMENT_UNITS,
        "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
        "source_git_sha": source_git_sha,
        "parent_artifact_id": artifact_id,
        "act_run_id": DEFAULT_ACT_RUN_ID,
        "family_id": family_id,
        "family_stratum": TRIAL_FAMILY_STRATA[family_id],
        "operation": operation,
        "seed": TRIAL_SEED,
        "objective": objective,
    }
    run = RunLog(
        root_path,
        "diag_objective_kl_trial_unit",
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
            teacher_id="rule-v0:diag-objective-kl-trial-0",
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=RuntimeContext(task=correction_task),
            max_steps=TRIAL_MAX_EPISODE_STEPS,
        )
        traces = read_intervention_traces_jsonl(rollout_path / "interventions.jsonl")
        if not traces:
            raise ValueError("DIAG v5 correction rollout produced no intervention traces")
        if any(trace.route != "model" for trace in traces):
            raise ValueError("DIAG v5 correction rollout must remain model-only")
        if any(trace.executed_action != trace.student_action for trace in traces):
            raise ValueError("DIAG v5 model-only rollout must execute student Action")

        correction_examples = examples_from_interventions(traces)
        if not correction_examples:
            raise ValueError("DIAG v5 correction rollout produced no trainable labels")
        base_examples = tuple(baseline_suite.training_examples())
        replay_examples = replay_decision_examples(base_examples, len(correction_examples))
        replay_training = aggregate_decision_examples(base_examples, replay_examples)
        correction_training = aggregate_decision_examples(base_examples, correction_examples)
        if len(replay_training) != len(correction_training):
            raise RuntimeError("DIAG v5 Replay/Correction training lengths differ")
        sibling_examples = tuple(_teacher_examples(sibling_task))
        training_digests = {
            "replay": _examples_digest(replay_training),
            "correction": _examples_digest(correction_training),
        }

        control_baseline = deepcopy(parent_model)
        treatment_baseline = deepcopy(parent_model)
        equivalence_examples = (
            tuple(replay_training)
            + tuple(correction_training)
            + tuple(sibling_examples)
        )
        step0_equivalence = assert_step0_equivalence(
            control_baseline, treatment_baseline, tokenizer, equivalence_examples
        )
        gradient_equivalence = assert_initial_objective_equivalence(
            parent_model, tokenizer, tuple(correction_training)
        )
        selected_baseline = (
            control_baseline if objective == "ce" else treatment_baseline
        )
        spec = objective_spec(parent_model, selected_baseline, objective)
        trainability = _require_dict(spec, "trainability")
        trainable_names = tuple(
            cast(list[str], trainability["trainable_parameter_names"])
        )

        replay_candidate = deepcopy(selected_baseline)
        correction_candidate = deepcopy(selected_baseline)
        replay_recorder = ObjectiveLossRecorder(parent_model, objective)
        correction_recorder = ObjectiveLossRecorder(parent_model, objective)
        replay_losses, replay_snapshots, replay_elapsed = train_decision_with_checkpoints(
            replay_candidate,
            tokenizer,
            replay_training,
            checkpoints=TRIAL_CHECKPOINTS,
            config=config,
            trainable_parameter_names=trainable_names,
            loss_fn=replay_recorder,
        )
        correction_losses, correction_snapshots, correction_elapsed = (
            train_decision_with_checkpoints(
                correction_candidate,
                tokenizer,
                correction_training,
                checkpoints=TRIAL_CHECKPOINTS,
                config=config,
                trainable_parameter_names=trainable_names,
                loss_fn=correction_recorder,
            )
        )
        if len(replay_recorder.ce_losses) != config.steps:
            raise RuntimeError("DIAG v5 Replay component loss history is incomplete")
        if len(correction_recorder.ce_losses) != config.steps:
            raise RuntimeError("DIAG v5 Correction component loss history is incomplete")

        if len(replay_losses) != len(correction_losses):
            raise RuntimeError("DIAG v5 Replay/Correction optimizer-step counts differ")

        tokenizer_after_sha = _tokenizer_artifact_digest(artifact_path)
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
        for arm, losses, recorder in (("replay", replay_losses, replay_recorder), ("correction", correction_losses, correction_recorder)):
            _write_json(
                run.path / f"training-{arm}.json",
                {
                    "schema_version": "aster-diag-objective-kl-training-0",
                    "arm": arm,
                    "objective": objective,
                    "config": asdict(config),
                    "trainability": trainability,
                    "training_examples_sha256": training_digests[arm],
                    "losses": losses,
                    "ce_losses": recorder.ce_losses,
                    "kl_losses": recorder.kl_losses,
                    "resources": resources[arm],
                },
            )

        checkpoints: list[dict[str, object]] = []
        for step in TRIAL_CHECKPOINTS:
            checkpoints.append(
                {
                    "step": step,
                    "replay": _checkpoint_metrics(
                        selected_baseline,
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
                        selected_baseline,
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
            "schema_version": "aster-diag-objective-kl-unit-0",
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "causal_claim": False,
            "confirmatory_verdict": None,
            "source_git_sha": source_git_sha,
            "run_id": run.id,
            "rollout_run_id": rollout_path.name,
            "parent_artifact_id": artifact_id,
            "family_id": family_id,
            "family_stratum": TRIAL_FAMILY_STRATA[family_id],
            "operation": operation,
            "seed": TRIAL_SEED,
            "objective": objective,
            "independent_family_block": family_id,
            "objective_condition_is_independent_sample": False,
            "checkpoint_observations_are_independent": False,
            "objective_spec": spec,
            "step0_equivalence": step0_equivalence,
            "step0_objective_gradient_equivalence": gradient_equivalence,
            "tokenizer_identity": {
                "manifest_sha256": manifest_tokenizer_sha,
                "before_sha256": tokenizer_before_sha,
                "after_sha256": tokenizer_after_sha,
            },
            "training_examples_sha256": training_digests,
            "checkpoints": checkpoints,
            "resources": resources,
        }
        unit["trajectory_summary"] = summarize_diag_trial_unit(unit)
        unit["condition_summary"] = summarize_backbone_freeze_unit(unit)
        _validate_unit_invariants(unit)
        _write_json(run.path / "diag-objective-kl-unit.json", unit)
        run.finish(
            "completed",
            result="diag-objective-kl-unit.json",
            rollout_run_id=rollout_path.name,
            family_id=family_id,
            objective=objective,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _validate_matched_training_data(
    linear: dict[str, object],
    residual: dict[str, object],
) -> None:
    linear_digests = _require_dict(linear, "training_examples_sha256")
    residual_digests = _require_dict(residual, "training_examples_sha256")
    for arm in ("replay", "correction"):
        if _require_str(linear_digests, arm) != _require_str(residual_digests, arm):
            raise ValueError(
                f"DIAG v5 causal factor changed {arm} training examples"
            )


def aggregate_objective_kl_units(
    units: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate paired architectures without treating repeated conditions as samples."""
    if len(units) != EXPECTED_EXPERIMENT_UNITS:
        return {
            "schema_version": "aster-diag-objective-kl-result-0",
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
            raise ValueError("Mixed DIAG v5 cycle IDs")
        family_id = _require_str(unit, "family_id")
        objective = _require_str(unit, "objective")
        if family_id not in TRIAL_FAMILY_IDS:
            raise ValueError(f"Unexpected DIAG v5 family: {family_id}")
        if objective not in OBJECTIVES:
            raise ValueError(f"Unexpected DIAG v5 objective: {objective}")
        key = (family_id, objective)
        if key in indexed:
            raise ValueError(f"Duplicate DIAG v5 unit: {key}")
        unit_sha = _require_str(unit, "source_git_sha")
        if source_git_sha is None:
            source_git_sha = unit_sha
        elif unit_sha != source_git_sha:
            raise ValueError("DIAG v5 units were produced by different Git SHAs")
        indexed[key] = unit

    expected_keys = {
        (family_id, objective)
        for family_id in TRIAL_FAMILY_IDS
        for objective in OBJECTIVES
    }
    if set(indexed) != expected_keys:
        raise ValueError("DIAG v5 completed with unexpected family/condition set")

    family_summaries: list[dict[str, object]] = []
    for family_id in TRIAL_FAMILY_IDS:
        linear = indexed[(family_id, "ce")]
        residual = indexed[(family_id, "ce-kl")]
        _validate_matched_training_data(linear, residual)
        if _require_str(linear, "parent_artifact_id") != _require_str(
            residual, "parent_artifact_id"
        ):
            raise ValueError("DIAG v5 objective conditions use different parents")

        linear_summary = _require_dict(linear, "condition_summary")
        residual_summary = _require_dict(residual, "condition_summary")
        linear_d = _require_float(
            linear_summary, "sibling_margin_correction_minus_replay_delta"
        )
        residual_d = _require_float(
            residual_summary, "sibling_margin_correction_minus_replay_delta"
        )
        linear_repair = _require_float(
            linear_summary, "repair_accuracy_correction_minus_replay_delta"
        )
        residual_repair = _require_float(
            residual_summary, "repair_accuracy_correction_minus_replay_delta"
        )
        family_summaries.append(
            {
                "family_id": family_id,
                "family_stratum": _require_str(linear, "family_stratum"),
                "operation": _require_str(linear, "operation"),
                "ce": {
                    "D_sibling_margin": linear_d,
                    "Repair_CR": linear_repair,
                },
                "objective_kl": {
                    "D_sibling_margin": residual_d,
                    "Repair_CR": residual_repair,
                },
                "contrasts": {
                    "stability_gain_kl": residual_d - linear_d,
                    "repair_gain_kl": residual_repair - linear_repair,
                },
                "training_examples_sha256": _require_dict(
                    linear, "training_examples_sha256"
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
        values: list[float] = []
        for row in rows:
            contrasts = _require_dict(row, "contrasts")
            values.append(_require_float(contrasts, key))
        return median(values)

    descriptive = {
        "loss_stability_gain_kl_median": contrast_median(
            loss_rows, "stability_gain_kl"
        ),
        "loss_repair_gain_kl_median": contrast_median(
            loss_rows, "repair_gain_kl"
        ),
        "control_stability_gain_kl_median": contrast_median(
            control_rows, "stability_gain_kl"
        ),
        "control_repair_gain_kl_median": contrast_median(
            control_rows, "repair_gain_kl"
        ),
    }

    return {
        "schema_version": "aster-diag-objective-kl-result-0",
        "cycle_id": TRIAL_CYCLE_ID,
        "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
        "status": "complete",
        "experiment_mode": "trial",
        "evidence_scope": "diagnostic",
        "post_hoc_selected": True,
        "causal_claim": False,
        "source_git_sha": source_git_sha,
        "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
        "paired_objective_conditions_per_family": EXPECTED_OBJECTIVE_CONDITIONS,
        "experiment_units": EXPECTED_EXPERIMENT_UNITS,
        "arm_training_trajectories": EXPECTED_ARM_TRAJECTORIES,
        "unit_checkpoint_records": EXPECTED_UNIT_CHECKPOINT_RECORDS,
        "arm_checkpoint_records": EXPECTED_ARM_CHECKPOINT_RECORDS,
        "objective_conditions_are_independent_samples": False,
        "checkpoint_observations_are_independent": False,
        "confirmatory_verdict": None,
        "diagnostic_classification": None,
        "primary_quantities": {
            "D": "Correction sibling-margin delta - Replay sibling-margin delta",
            "Repair_CR": "Correction repair-accuracy delta - Replay repair-accuracy delta",
            "stability_gain_kl": "D_residual - D_linear",
            "repair_gain_kl": "Repair_CR_residual - Repair_CR_linear",
        },
        "family_summaries": family_summaries,
        "descriptive_summary": descriptive,
        "stop_rule_satisfied": True,
    }


def run_objective_kl_campaign(root: str | Path) -> Path:
    """Run or resume the fixed 4-family x 2-objective DIAG v5 campaign."""
    root_path = Path(root).resolve()
    source_git_sha = _clean_git_sha(root_path)
    completed = discover_completed_objective_kl_units(
        root_path, source_git_sha=source_git_sha
    )
    campaign = RunLog(
        root_path,
        "diag_objective_kl_trial_campaign",
        {
            "experiment_mode": "trial",
            "cycle_id": TRIAL_CYCLE_ID,
            "parent_cycle_id": TRIAL_PARENT_CYCLE_ID,
            "evidence_scope": "diagnostic",
            "post_hoc_selected": True,
            "causal_factor": "training_objective",
            "planned_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "independent_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "source_git_sha": source_git_sha,
            "families": list(TRIAL_FAMILY_IDS),
            "architectures": list(OBJECTIVES),
            "seed": TRIAL_SEED,
            "checkpoints": list(TRIAL_CHECKPOINTS),
            "completed_units_at_start": len(completed),
        },
        producer="trainer",
    )
    try:
        for family_id in TRIAL_FAMILY_IDS:
            for objective in OBJECTIVES:
                key = (family_id, objective)
                if key not in completed:
                    path = run_objective_kl_unit(
                        root_path,
                        family_id=family_id,
                        objective=objective,
                        expected_source_git_sha=source_git_sha,
                    )
                    unit = _read_json(path / "diag-objective-kl-unit.json")
                    completed[key] = (path, unit)
                campaign.event(
                    "unit_available",
                    {
                        "family_id": family_id,
                        "objective": objective,
                        "run_id": completed[key][0].name,
                    },
                )
                _write_progress(campaign.path, completed)

        result = aggregate_objective_kl_units(
            [unit for _, unit in completed.values()]
        )
        if result.get("status") != "complete":
            raise RuntimeError("DIAG v5 ended without all fixed units")
        _write_json(campaign.path / "diag-objective-kl-results.json", result)
        campaign.finish(
            "completed",
            results="diag-objective-kl-results.json",
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


def discover_completed_objective_kl_units(
    root: str | Path,
    *,
    source_git_sha: str,
) -> dict[tuple[str, str], tuple[Path, dict[str, object]]]:
    """Reuse only completed DIAG v5 units from the exact source revision."""
    runs_root = Path(root).resolve() / "runs"
    if not runs_root.exists():
        return {}

    found: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        result_file = run_dir / "diag-objective-kl-unit.json"
        if not run_file.is_file():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "diag_objective_kl_trial_unit":
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
            raise RuntimeError("A DIAG v5 unit is still marked running")
        if run.get("status") != "completed":
            continue
        if not result_file.is_file():
            raise RuntimeError("Completed DIAG v5 unit is missing its result")
        unit = _read_json(result_file)
        if unit.get("source_git_sha") != source_git_sha:
            raise RuntimeError(
                "Existing DIAG v5 evidence uses a different Git SHA; "
                "start a new Trial cycle after code changes"
            )
        family_id = _require_str(unit, "family_id")
        objective = _require_str(unit, "objective")
        key = (family_id, objective)
        if key in found:
            raise RuntimeError(f"Duplicate completed DIAG v5 unit: {key}")
        found[key] = (run_dir, unit)
    return found


def _write_progress(
    campaign_path: Path,
    completed: dict[tuple[str, str], tuple[Path, dict[str, object]]],
) -> None:
    rows = [
        {
            "family_id": family_id,
            "objective": objective,
            "run_id": path.name,
        }
        for (family_id, objective), (path, _) in sorted(completed.items())
    ]
    _write_json(
        campaign_path / "progress.json",
        {
            "schema_version": "aster-diag-objective-kl-progress-0",
            "cycle_id": TRIAL_CYCLE_ID,
            "expected_family_blocks": EXPECTED_FAMILY_BLOCKS,
            "expected_experiment_units": EXPECTED_EXPERIMENT_UNITS,
            "completed_experiment_units": len(rows),
            "units": rows,
            "trial_endpoint_metrics_may_be_inspected": True,
        },
    )
