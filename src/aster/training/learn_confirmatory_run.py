"""Gate 4 execution for the frozen LEARN-v0 confirmatory protocol."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import cast

from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy
from aster.model.decision_artifact import load_decision_artifact
from aster.records.decision import DecisionExample
from aster.records.runlog import RunLog
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.service_agent import build_executor
from aster.training.decision import DecisionTrainConfig, examples_from_teacher_trajectory
from aster.training.experiment import run_logged_correction_transfer_experiment
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
    validate_confirmatory_candidate_coverage,
)
from aster.training.learn_confirmatory_analysis import aggregate_confirmatory_units
from aster.training.learn_probe import resolve_act_parent_artifact


def build_confirmatory_sibling_suite(
    manifest: dict[str, object],
    family_id: str,
) -> BenchmarkSuite:
    """Build one frozen family's sibling-only test suite plus fixed calibration."""
    family = _family_by_id(manifest, family_id)
    sibling = _task_from_family(family, "sibling_task")
    leakage_group = _require_str(family, "leakage_group")
    baseline = build_calculate_and_store_suite()
    cases = list(baseline.cases_for("calibration"))
    for step, example in enumerate(_teacher_examples(sibling)):
        cases.append(
            BenchmarkCase(
                case_id=f"{family_id}/sibling/step-{step}",
                split="test",
                slice_name="uncorrected_sibling",
                leakage_group=leakage_group,
                example=example,
            )
        )
    protocol_id = _require_str(manifest, "protocol_id")
    return BenchmarkSuite(f"{protocol_id}/{family_id}", tuple(cases))


def run_confirmatory_unit(
    root: str | Path,
    manifest: dict[str, object],
    *,
    family_id: str,
    seed: int,
    expected_measurement_git_sha: str | None = None,
) -> Path:
    """Run exactly one frozen family/seed unit from the registered ACT parent."""
    root_path = Path(root).resolve()
    measurement_git_sha = _measurement_git_sha(root_path)
    if (
        expected_measurement_git_sha is not None
        and measurement_git_sha != expected_measurement_git_sha
    ):
        raise RuntimeError(
            "Git revision changed during confirmatory measurement; protocol v0 cannot continue"
        )
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen confirmatory manifest digest mismatch")
    family = _family_by_id(manifest, family_id)
    training = _require_dict(manifest, "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"Seed {seed} is not frozen in the confirmatory manifest")

    lineage = _require_dict(manifest, "lineage")
    act_run_id = _require_str(lineage, "parent_act_run_id")
    artifact_id, artifact_path = resolve_act_parent_artifact(root_path, act_run_id)
    model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise ValueError("Resolved confirmatory parent artifact identity mismatch")

    baseline = build_calculate_and_store_suite()
    _validate_parent_suite_lineage(parent_manifest, baseline)
    model_id = _require_str(parent_manifest, "model_id")

    config = DecisionTrainConfig(
        steps=_require_int(training, "optimizer_steps"),
        learning_rate=_require_float(training, "learning_rate"),
        train_backbone=_require_bool(training, "train_backbone"),
        seed=seed,
    )
    max_steps = _require_int(training, "max_episode_steps")
    correction_task = _task_from_family(family, "correction_task")
    sibling_task = _task_from_family(family, "sibling_task")
    suite = build_confirmatory_sibling_suite(manifest, family_id)

    primary = ModelPolicy(model, tokenizer, model_id=model_id)
    policy = SelectivePolicy(
        primary,
        fallback=None,
        config=SelectivePolicyConfig(
            temperature=1.0,
            autonomous_threshold=0.0,
            fallback_threshold=0.0,
        ),
    )

    run_path = run_logged_correction_transfer_experiment(
        root_path,
        policy=policy,
        teacher=RuleBasedPolicy(),
        teacher_id="rule-v0:learn-confirmatory-teacher",
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=correction_task),
        parent_model=model,
        tokenizer=tokenizer,
        base_examples=baseline.training_examples(),
        benchmark_suite=suite,
        train_config=config,
        replay_model_id=f"{model_id}+replay-{family_id}-s{seed}",
        correction_model_id=f"{model_id}+correction-{family_id}-s{seed}",
        max_steps=max_steps,
        sequential_task=sibling_task,
        parent_artifact_id=artifact_id,
        provenance={
            "scope": "confirmatory_measurement",
            "confirmatory_evidence": True,
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": manifest_sha256,
            "measurement_git_sha": measurement_git_sha,
            "family_id": family_id,
            "leakage_group": family["leakage_group"],
            "seed": seed,
            "act_run_id": act_run_id,
            "parent_artifact_id": artifact_id,
            "correction_task": dict(correction_task),
            "uncorrected_sibling_task": dict(sibling_task),
        },
    )
    unit = extract_confirmatory_unit_result(run_path, manifest)
    _write_json(run_path / "confirmatory-unit.json", unit)
    return run_path


def extract_confirmatory_unit_result(
    run_path: str | Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Extract one compact unit result from canonical experiment evidence."""
    path = Path(run_path)
    experiment = _read_json(path / "correction-transfer.json")
    provenance = _require_dict(experiment, "provenance")
    protocol_id = _require_str(manifest, "protocol_id")
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    if provenance.get("protocol_id") != protocol_id:
        raise ValueError("Confirmatory run protocol mismatch")
    if provenance.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Confirmatory run manifest digest mismatch")
    if provenance.get("confirmatory_evidence") is not True:
        raise ValueError("Run is not marked as confirmatory evidence")

    family_id = _require_str(provenance, "family_id")
    seed = _require_int(provenance, "seed")
    measurement_git_sha = _require_str(provenance, "measurement_git_sha")
    _family_by_id(manifest, family_id)

    replay_primary = _benchmark_primary(path / "benchmark-replay")
    correction_primary = _benchmark_primary(path / "benchmark-correction")
    if replay_primary["examples"] != correction_primary["examples"]:
        raise ValueError("Replay/Correction primary example counts differ")

    repair_raw = _require_dict(experiment, "repair")
    repair: dict[str, object] = {}
    for arm in ("parent", "replay", "correction"):
        metrics = _require_dict(repair_raw, arm)
        repair[arm] = {
            "accuracy": _require_float(metrics, "accuracy"),
            "nll": _require_float(metrics, "nll"),
            "examples": _require_int(metrics, "examples"),
        }

    sequential_raw = _require_dict(experiment, "sequential_transfer")
    sequential_arms = _require_dict(sequential_raw, "arms")
    sequential: dict[str, object] = {}
    for arm in ("parent", "replay", "correction"):
        arm_data = _require_dict(sequential_arms, arm)
        summary = _require_dict(arm_data, "summary")
        evaluation = _require_dict(summary, "evaluation")
        sequential[arm] = {
            "run_id": _require_str(arm_data, "run_id"),
            "task_success": _require_bool(evaluation, "task_success"),
            "goal_verified": _require_bool(evaluation, "goal_verified"),
            "steps": _require_int(evaluation, "steps"),
            "terminal_reached": _require_bool(evaluation, "terminal_reached"),
            "first_goal_verified_step": _optional_int(
                evaluation.get("first_goal_verified_step"),
                "first_goal_verified_step",
            ),
            "stop_reason": _optional_str(evaluation.get("stop_reason"), "stop_reason"),
            "teacher_disagreements": _require_int(summary, "teacher_disagreements"),
            "first_teacher_divergence_step": _optional_int(
                summary.get("first_teacher_divergence_step"),
                "first_teacher_divergence_step",
            ),
            "steps_after_first_teacher_divergence": _require_int(
                summary, "steps_after_first_teacher_divergence"
            ),
            "first_tool_failure_step": _optional_int(
                summary.get("first_tool_failure_step"),
                "first_tool_failure_step",
            ),
        }

    budget = _require_dict(experiment, "budget")
    resources = _require_dict(budget, "measured_resources")
    training_examples = _require_int(budget, "training_examples_per_candidate")
    resource_audit = _require_dict(manifest, "resource_audit")
    required_resources = resource_audit.get("required_per_arm")
    if not isinstance(required_resources, list) or any(
        not isinstance(item, str) for item in required_resources
    ):
        raise ValueError("Confirmatory resource audit fields are invalid")
    unit_resources: dict[str, object] = {}
    for arm in ("replay", "correction"):
        values = dict(_require_dict(resources, arm))
        values["training_examples"] = training_examples
        missing = [
            item for item in required_resources
            if isinstance(item, str) and item not in values
        ]
        if missing:
            raise ValueError(
                f"Confirmatory resource evidence missing for {arm}: {missing}"
            )
        unit_resources[arm] = values

    unit = {
        "schema_version": "aster-learn-confirmatory-unit-0",
        "protocol_id": protocol_id,
        "manifest_sha256": manifest_sha256,
        "family_id": family_id,
        "seed": seed,
        "measurement_git_sha": measurement_git_sha,
        "run_id": path.name,
        "primary": {
            "examples": replay_primary["examples"],
            "replay_correct": replay_primary["correct"],
            "correction_correct": correction_primary["correct"],
            "replay_accuracy": replay_primary["accuracy"],
            "correction_accuracy": correction_primary["accuracy"],
            "accuracy_delta": (
                correction_primary["accuracy"] - replay_primary["accuracy"]
            ),
            "replay_nll": replay_primary["nll"],
            "correction_nll": correction_primary["nll"],
            "nll_effect_replay_minus_correction": (
                replay_primary["nll"] - correction_primary["nll"]
            ),
        },
        "repair": repair,
        "sequential": sequential,
        "resources": unit_resources,
    }
    return unit


def discover_completed_confirmatory_units(
    root: str | Path,
    manifest: dict[str, object],
    *,
    measurement_git_sha: str | None = None,
) -> dict[tuple[str, int], tuple[Path, dict[str, object]]]:
    """Find reusable completed units and reject ambiguous or endpoint-emitting failures."""
    root_path = Path(root).resolve()
    runs_root = root_path / "runs"
    if not runs_root.exists():
        return {}

    protocol_id = _require_str(manifest, "protocol_id")
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    found: dict[tuple[str, int], tuple[Path, dict[str, object]]] = {}
    established_git_sha: str | None = None

    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        if not run_file.is_file() or run_file.is_symlink():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "correction_transfer_experiment":
            continue
        inputs = run.get("inputs")
        if not isinstance(inputs, dict):
            continue
        provenance = inputs.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if provenance.get("protocol_id") != protocol_id:
            continue
        if provenance.get("manifest_sha256") != manifest_sha256:
            continue
        if provenance.get("confirmatory_evidence") is not True:
            continue

        family_id = _require_str(cast(dict[str, object], provenance), "family_id")
        seed = _require_int(cast(dict[str, object], provenance), "seed")
        _family_by_id(manifest, family_id)
        key = (family_id, seed)
        status = run.get("status")

        if status == "completed":
            unit = extract_confirmatory_unit_result(run_dir, manifest)
            unit_git_sha = _require_str(unit, "measurement_git_sha")
            if established_git_sha is None:
                established_git_sha = unit_git_sha
            elif unit_git_sha != established_git_sha:
                raise RuntimeError(
                    "Completed confirmatory units were produced by different Git SHAs"
                )
            if measurement_git_sha is not None and unit_git_sha != measurement_git_sha:
                raise RuntimeError(
                    "Existing confirmatory endpoint evidence was produced by a different "
                    "Git SHA; protocol v0 cannot continue after measurement code changed"
                )
            if key in found:
                raise RuntimeError(
                    f"Duplicate completed confirmatory unit: {family_id} seed {seed}"
                )
            found[key] = (run_dir, unit)
            continue

        if status == "running":
            raise RuntimeError(
                f"Confirmatory unit is still marked running: {family_id} seed {seed}"
            )
        if status in {"failed", "interrupted"} and _primary_endpoint_emitted(run_dir):
            raise RuntimeError(
                "A failed confirmatory attempt already emitted primary endpoint metrics; "
                "protocol v0 forbids retrying that family/seed"
            )

    return found


def run_confirmatory_campaign(root: str | Path) -> Path:
    """Run or resume all 90 frozen units and emit a verdict only when complete."""
    root_path = Path(root).resolve()
    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    measurement_git_sha = _measurement_git_sha(root_path)
    coverage = validate_confirmatory_candidate_coverage(manifest)
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    design = _require_dict(manifest, "family_design")
    training = _require_dict(manifest, "training")
    families = _require_list(design, "families")
    seeds_raw = training.get("seeds")
    if not isinstance(seeds_raw, list) or any(type(seed) is not int for seed in seeds_raw):
        raise ValueError("Confirmatory seeds are invalid")
    seeds = cast(list[int], seeds_raw)

    existing = discover_completed_confirmatory_units(
        root_path,
        manifest,
        measurement_git_sha=measurement_git_sha,
    )
    expected_units = len(families) * len(seeds)
    campaign = RunLog(
        root_path,
        "learn_confirmatory_campaign",
        {
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": manifest_sha256,
            "measurement_git_sha": measurement_git_sha,
            "expected_units": expected_units,
            "completed_units_at_start": len(existing),
            "candidate_coverage": coverage,
        },
        producer="trainer",
    )

    try:
        completed = dict(existing)
        for raw_family in families:
            if not isinstance(raw_family, dict):
                raise ValueError("Confirmatory family is invalid")
            family = cast(dict[str, object], raw_family)
            family_id = _require_str(family, "family_id")
            for seed in seeds:
                key = (family_id, seed)
                if key in completed:
                    campaign.event(
                        "unit_reused",
                        {
                            "family_id": family_id,
                            "seed": seed,
                            "run_id": completed[key][0].name,
                        },
                    )
                    continue
                unit_path = run_confirmatory_unit(
                    root_path,
                    manifest,
                    family_id=family_id,
                    seed=seed,
                    expected_measurement_git_sha=measurement_git_sha,
                )
                unit = extract_confirmatory_unit_result(unit_path, manifest)
                completed[key] = (unit_path, unit)
                campaign.event(
                    "unit_completed",
                    {
                        "family_id": family_id,
                        "seed": seed,
                        "run_id": unit_path.name,
                    },
                )
                _write_progress(campaign.path, expected_units, completed)

        result = aggregate_confirmatory_units(
            manifest,
            [unit for _, unit in completed.values()],
        )
        if result.get("status") != "complete":
            raise RuntimeError("Confirmatory campaign ended without all frozen units")
        _write_json(campaign.path / "confirmatory-results.json", result)
        verdict = _require_dict(result, "verdict")
        campaign.finish(
            "completed",
            results="confirmatory-results.json",
            completed_units=len(completed),
            verdict=verdict.get("label"),
        )
        return campaign.path
    except BaseException as error:
        campaign.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _measurement_git_sha(root: Path) -> str:
    """Require one clean source revision for the entire confirmatory campaign."""
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ValueError(
                "Confirmatory measurement requires a clean Git working tree"
            )
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Confirmatory measurement requires a readable Git revision") from error
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Confirmatory Git SHA is invalid")
    return revision


def _benchmark_primary(directory: Path) -> dict[str, int | float]:
    benchmark = _read_json(directory / "benchmark.json")
    test = _require_dict(benchmark, "test")
    raw = _require_dict(test, "raw")
    predictions_path = directory / "predictions.jsonl"
    correct = 0
    examples = 0
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("split") != "test":
            continue
        value = item.get("correct")
        if type(value) is not bool:
            raise ValueError("Benchmark prediction correct flag is invalid")
        examples += 1
        correct += int(value)
    if examples != _require_int(raw, "examples"):
        raise ValueError("Benchmark summary/prediction example count mismatch")
    accuracy = _require_float(raw, "accuracy")
    if accuracy != correct / examples:
        raise ValueError("Benchmark summary/prediction accuracy mismatch")
    return {
        "examples": examples,
        "correct": correct,
        "accuracy": accuracy,
        "nll": _require_float(raw, "nll"),
    }


def _primary_endpoint_emitted(run_dir: Path) -> bool:
    return any(
        (run_dir / f"benchmark-{arm}" / "benchmark.json").is_file()
        and (run_dir / f"benchmark-{arm}" / "predictions.jsonl").is_file()
        for arm in ("replay", "correction")
    )


def _write_progress(
    campaign_path: Path,
    expected_units: int,
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
            "schema_version": "aster-learn-confirmatory-progress-0",
            "expected_units": expected_units,
            "completed_units": len(rows),
            "units": rows,
            "endpoint_metrics_exposed": False,
        },
    )


def _validate_parent_suite_lineage(
    parent_manifest: dict[str, object],
    baseline: BenchmarkSuite,
) -> None:
    expected = digest(json_bytes(baseline.to_dict()))
    if parent_manifest.get("suite_id") != baseline.suite_id:
        raise ValueError("Confirmatory parent suite ID mismatch")
    if parent_manifest.get("suite_sha256") != expected:
        raise ValueError("Confirmatory parent suite digest mismatch")


def _teacher_examples(task: dict[str, JsonValue]) -> list[DecisionExample]:
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )
    return examples_from_teacher_trajectory(
        trajectory,
        teacher="rule-v0:learn-confirmatory-benchmark",
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
        raise ValueError(f"Unknown or ambiguous confirmatory family: {family_id}")
    return matches[0]


def _task_from_family(
    family: dict[str, object],
    key: str,
) -> dict[str, JsonValue]:
    task = _require_dict(family, key)
    return cast(dict[str, JsonValue], dict(task))


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


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"Confirmatory field {field!r} must be an integer or null")
    return cast(int, value)


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Confirmatory field {field!r} must be a string or null")
    return value


def _require_bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise ValueError(f"Confirmatory field {key!r} must be boolean")
    return cast(bool, value)
