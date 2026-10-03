"""Fixed real-artifact development probe for LEARN v0 correction transfer."""

from __future__ import annotations

import json
from pathlib import Path
import re

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_artifact import load_decision_artifact
from aster.records.decision import DecisionExample
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.service_agent import build_executor
from aster.service.artifacts import ArtifactCatalog
from aster.training.decision import DecisionTrainConfig, examples_from_teacher_trajectory
from aster.training.experiment import run_logged_correction_transfer_experiment

DEFAULT_ACT_RUN_ID = "5a8076571dca446fa07190cf4fc62509"
PROBE_ID = "learn-correction-transfer-dev-key-shift-v0"
PROBE_FAMILY_ID = "learn-dev-key-shift-family-v0"
PARENT_SUITE_ID = "calculate-and-store-v0"
ACT_RECIPE_ID = "agent-decision-model-v0"
ACT_TASK_ID = "act-calculate-store-dev-v0"

CORRECTION_TASK: dict[str, JsonValue] = {
    "task_id": "learn-dev-key-shift-correction-v0",
    "kind": "calculate_and_store",
    "operation": "add",
    "left": 2,
    "right": 3,
    "store_as": "diagnostic_total",
}
SIBLING_TASK: dict[str, JsonValue] = {
    "task_id": "learn-dev-key-shift-sibling-v0",
    "kind": "calculate_and_store",
    "operation": "add",
    "left": 2,
    "right": 3,
    "store_as": "sibling_total",
}
TRAIN_CONFIG = DecisionTrainConfig(
    steps=100,
    learning_rate=3e-3,
    train_backbone=True,
    seed=42,
)

_RUN_ID = re.compile(r"[0-9a-f]{32}")


def resolve_act_parent_artifact(
    root: str | Path,
    act_run_id: str = DEFAULT_ACT_RUN_ID,
) -> tuple[str, Path]:
    """Resolve the exact registered DecisionModel referenced by a completed ACT-v0 Run."""
    if not _RUN_ID.fullmatch(act_run_id):
        raise ValueError("ACT run ID must be a 32-character lowercase hex ID")

    root_path = Path(root).resolve()
    run_path = _find_act_run_path(root_path, act_run_id)

    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ACT run.json is invalid") from error
    if not isinstance(run, dict):
        raise ValueError("ACT run.json must be an object")
    if run.get("schema_version") != "aster-run-0":
        raise ValueError("ACT run schema is unsupported")
    if run.get("run_id") != act_run_id:
        raise ValueError("ACT run ID does not match its directory")
    if run.get("kind") != "agent" or run.get("status") != "completed":
        raise ValueError("ACT parent lineage requires one completed agent Run")

    inputs = run.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("ACT run inputs are missing")
    if inputs.get("recipe_id") != ACT_RECIPE_ID:
        raise ValueError("Run is not an ACT-v0 DecisionModel recipe")
    if inputs.get("policy_mode") != "model_only":
        raise ValueError("ACT parent Run must be model-only")
    if inputs.get("task_id") != ACT_TASK_ID:
        raise ValueError("ACT parent Run used an unexpected task")

    artifact_id = inputs.get("decision_model_artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("ACT run does not reference a DecisionModel artifact")
    artifact_path = ArtifactCatalog(root_path).resolve(artifact_id, "decision_model")
    return artifact_id, artifact_path



def _find_act_run_path(root: Path, act_run_id: str) -> Path:
    """Find one RunLog record from either a direct run or a Service workbench job."""
    runs_root = (root / "runs").resolve()
    candidates = [root / "runs" / act_run_id / "run.json"]
    workbench = root / "runs" / "workbench"
    if workbench.is_dir() and not workbench.is_symlink():
        candidates.extend(workbench.glob(f"*/runs/{act_run_id}/run.json"))

    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(runs_root):
            continue
        if any(parent.is_symlink() for parent in candidate.parents if parent != root.parent):
            continue
        matches.append(resolved)

    unique = sorted(set(matches), key=str)
    if not unique:
        raise ValueError(
            "ACT run does not exist under runs/<run_id> or "
            "runs/workbench/<job_id>/runs/<run_id>"
        )
    if len(unique) != 1:
        raise ValueError("ACT run ID is ambiguous across local run stores")
    return unique[0]

def build_learn_dev_probe_suite() -> BenchmarkSuite:
    """Build a development-only sibling suite; it never opens the project sealed test."""
    baseline = build_calculate_and_store_suite()
    cases = list(baseline.cases_for("calibration"))
    for step, example in enumerate(_teacher_examples(SIBLING_TASK)):
        cases.append(
            BenchmarkCase(
                case_id=f"{PROBE_ID}/sibling/step-{step}",
                split="test",
                slice_name="uncorrected_sibling",
                leakage_group=PROBE_FAMILY_ID,
                example=example,
            )
        )
    return BenchmarkSuite(PROBE_ID, tuple(cases))


def run_learn_dev_probe(
    root: str | Path,
    *,
    act_run_id: str = DEFAULT_ACT_RUN_ID,
) -> Path:
    """Run the predeclared 1-family/1-seed development probe from the real ACT parent."""
    root_path = Path(root).resolve()
    artifact_id, artifact_path = resolve_act_parent_artifact(root_path, act_run_id)
    model, tokenizer, manifest = load_decision_artifact(artifact_path)

    if manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved parent artifact identity mismatch")
    if manifest.get("suite_id") != PARENT_SUITE_ID:
        raise ValueError("LEARN dev probe requires the calculate-and-store-v0 parent suite")
    model_id = _require_str(manifest, "model_id")
    temperature = _artifact_temperature(manifest)

    primary = ModelPolicy(model, tokenizer, model_id=model_id)
    policy = SelectivePolicy(
        primary,
        fallback=None,
        config=SelectivePolicyConfig(
            temperature=temperature,
            autonomous_threshold=0.0,
            fallback_threshold=0.0,
        ),
    )
    baseline = build_calculate_and_store_suite()
    suite = build_learn_dev_probe_suite()

    return run_logged_correction_transfer_experiment(
        root_path,
        policy=policy,
        teacher=RuleBasedPolicy(),
        teacher_id="rule-v0:learn-dev-teacher",
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=CORRECTION_TASK),
        parent_model=model,
        tokenizer=tokenizer,
        base_examples=baseline.training_examples(),
        benchmark_suite=suite,
        train_config=TRAIN_CONFIG,
        replay_model_id=f"{model_id}+replay-dev-v0",
        correction_model_id=f"{model_id}+correction-dev-v0",
        max_steps=8,
        provenance={
            "scope": "development_wiring_probe_only",
            "confirmatory_evidence": False,
            "probe_id": PROBE_ID,
            "family_id": PROBE_FAMILY_ID,
            "act_run_id": act_run_id,
            "parent_artifact_id": artifact_id,
            "correction_task": dict(CORRECTION_TASK),
            "uncorrected_sibling_task": dict(SIBLING_TASK),
            "seed": TRAIN_CONFIG.seed,
            "steps": TRAIN_CONFIG.steps,
            "learning_rate": TRAIN_CONFIG.learning_rate,
        },
    )


def _teacher_examples(task: dict[str, JsonValue]) -> list[DecisionExample]:
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )
    return examples_from_teacher_trajectory(
        trajectory,
        teacher="rule-v0:learn-dev-benchmark",
    )


def _artifact_temperature(manifest: dict[str, object]) -> float:
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Parent artifact calibration metadata is missing")
    value = calibration.get("temperature")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError("Parent artifact temperature must be positive")
    return float(value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Decision artifact field {key!r} must be a non-empty string")
    return value
