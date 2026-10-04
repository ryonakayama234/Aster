"""End-to-end supervised intervention experiment with held-out v0/v+1 comparison."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import torch

from aster.agent.policy import Policy
from aster.agent.selective import SelectivePolicy
from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.runner import (
    BenchmarkBundle,
    run_decision_benchmark,
    write_benchmark_artifacts,
)
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.model.decision_artifact import load_decision_artifact, save_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.intervention import read_intervention_traces_jsonl
from aster.records.runlog import RunLog
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.service.artifacts import ArtifactCatalog
from aster.tokenizer.artifact import AsterTokenizer
from aster.tools.executor import ToolExecutor
from aster.training.decision import DecisionTrainConfig, evaluate_decisions
from aster.training.intervention import (
    examples_from_interventions,
    train_intervention_candidate,
    train_replay_control_candidate,
)
from aster.training.sequential import run_logged_model_only_sequential_episode

_SCALAR_METRICS = ("accuracy", "nll", "brier", "ece")
_DECISION_CANDIDATE_BUILDER_ID = "calculate-and-store-v0"


def run_logged_intervention_learning_experiment(
    root: str | Path,
    *,
    policy: SelectivePolicy,
    teacher: Policy,
    teacher_id: str,
    executor: ToolExecutor,
    evaluator: TaskEvaluator,
    context: RuntimeContext,
    parent_model: DecisionModel,
    tokenizer: AsterTokenizer,
    base_examples: Sequence[DecisionExample],
    benchmark_suite: BenchmarkSuite,
    train_config: DecisionTrainConfig = DecisionTrainConfig(),
    candidate_model_id: str | None = None,
    max_steps: int = 8,
) -> Path:
    """Close one auditable rollout -> correction -> update -> held-out comparison loop.

    Runtime evidence stays in a child intervention run. This trainer run records lineage to
    that immutable rollout, trains a deep-copied v+1 candidate, independently re-fits
    temperature on the benchmark calibration split, and compares both models on the same
    held-out cases. No candidate is promoted automatically.
    """
    root = Path(root)
    parent_model_id = policy.model_id
    candidate_model_id = candidate_model_id or f"{parent_model_id}+intervention-v1"
    if candidate_model_id == parent_model_id:
        raise ValueError("candidate_model_id must differ from parent model_id")

    primary_model = getattr(policy.primary, "model", None)
    if primary_model is not None and primary_model is not parent_model:
        raise ValueError("parent_model must be the model used by the student policy")

    run = RunLog(
        root,
        "intervention_learning_experiment",
        {
            "parent_model_id": parent_model_id,
            "candidate_model_id": candidate_model_id,
            "teacher_id": teacher_id,
            "benchmark_suite_id": benchmark_suite.suite_id,
            "train_config": asdict(train_config),
            "base_examples": len(base_examples),
            "max_steps": max_steps,
        },
        producer="trainer",
    )

    try:
        rollout_path = run_logged_intervention_agent(
            root,
            policy=policy,
            teacher=teacher,
            teacher_id=teacher_id,
            executor=executor,
            evaluator=evaluator,
            context=context,
            max_steps=max_steps,
        )
        rollout_summary = _read_json(rollout_path / "run.json")
        rollout_run_id = str(rollout_summary["run_id"])
        traces = read_intervention_traces_jsonl(rollout_path / "interventions.jsonl")
        run.event(
            "rollout_collected",
            {
                "rollout_run_id": rollout_run_id,
                "interventions": len(traces),
                "trainable_interventions": sum(trace.trainable for trace in traces),
            },
        )

        parent_benchmark = run_decision_benchmark(
            parent_model,
            tokenizer,
            benchmark_suite,
            model_id=parent_model_id,
        )
        write_benchmark_artifacts(run.path / "benchmark-parent", parent_benchmark)

        update = train_intervention_candidate(
            parent_model,
            tokenizer,
            base_examples,
            traces,
            config=train_config,
        )
        training_payload = {
            "schema_version": "aster-intervention-training-0",
            "parent_model_id": parent_model_id,
            "candidate_model_id": candidate_model_id,
            "rollout_run_id": rollout_run_id,
            "config": asdict(train_config),
            "base_examples": len(base_examples),
            "update": update.summary(),
            "losses": list(update.losses),
        }
        _write_json(run.path / "training.json", training_payload)
        _write_examples_jsonl(run.path / "training-examples.jsonl", update.training_examples)
        run.event(
            "candidate_trained",
            {
                "candidate_model_id": candidate_model_id,
                "training_examples": len(update.training_examples),
                "intervention_examples": len(update.intervention_examples),
                "aggregate_after": update.aggregate_after,
                "intervention_after": update.intervention_after,
            },
        )

        candidate_benchmark = run_decision_benchmark(
            update.model,
            tokenizer,
            benchmark_suite,
            model_id=candidate_model_id,
        )
        write_benchmark_artifacts(run.path / "benchmark-candidate", candidate_benchmark)
        comparison = compare_benchmark_bundles(parent_benchmark, candidate_benchmark)
        experiment = {
            "schema_version": "aster-intervention-learning-experiment-0",
            "parent_model_id": parent_model_id,
            "candidate_model_id": candidate_model_id,
            "teacher_id": teacher_id,
            "rollout_run_id": rollout_run_id,
            "benchmark_suite_id": benchmark_suite.suite_id,
            "artifacts": {
                "training": "training.json",
                "training_examples": "training-examples.jsonl",
                "parent_benchmark": "benchmark-parent/benchmark.json",
                "parent_calibration": "benchmark-parent/calibration.json",
                "candidate_benchmark": "benchmark-candidate/benchmark.json",
                "candidate_calibration": "benchmark-candidate/calibration.json",
            },
            "comparison": comparison,
            "promotion": "candidate_only",
        }
        _write_json(run.path / "experiment.json", experiment)
        run.event(
            "benchmarked",
            {
                "parent_model_id": parent_model_id,
                "candidate_model_id": candidate_model_id,
                "parent_temperature": parent_benchmark.temperature,
                "candidate_temperature": candidate_benchmark.temperature,
                "test": comparison["test"],
            },
        )
        run.finish(
            "completed",
            experiment="experiment.json",
            training="training.json",
            training_examples="training-examples.jsonl",
            rollout_run_id=rollout_run_id,
            parent_benchmark="benchmark-parent/benchmark.json",
            candidate_benchmark="benchmark-candidate/benchmark.json",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise



def run_logged_correction_transfer_experiment(
    root: str | Path,
    *,
    policy: SelectivePolicy,
    teacher: Policy,
    teacher_id: str,
    executor: ToolExecutor,
    evaluator: TaskEvaluator,
    context: RuntimeContext,
    parent_model: DecisionModel,
    tokenizer: AsterTokenizer,
    base_examples: Sequence[DecisionExample],
    benchmark_suite: BenchmarkSuite,
    train_config: DecisionTrainConfig = DecisionTrainConfig(),
    replay_model_id: str | None = None,
    correction_model_id: str | None = None,
    max_steps: int = 8,
    provenance: dict[str, object] | None = None,
    sequential_task: dict[str, JsonValue] | None = None,
    parent_artifact_id: str | None = None,
) -> Path:
    """Run the LEARN-v0 P0/R1/C1 wiring experiment without a promotion verdict."""
    root = Path(root)
    parent_model_id = policy.model_id
    replay_model_id = replay_model_id or f"{parent_model_id}+replay-v1"
    correction_model_id = correction_model_id or f"{parent_model_id}+correction-v1"
    if len({parent_model_id, replay_model_id, correction_model_id}) != 3:
        raise ValueError("Parent, replay, and correction model IDs must be distinct")

    primary_model = getattr(policy.primary, "model", None)
    if primary_model is not None and primary_model is not parent_model:
        raise ValueError("parent_model must be the model used by the student policy")

    run = RunLog(
        root,
        "correction_transfer_experiment",
        {
            "parent_model_id": parent_model_id,
            "replay_model_id": replay_model_id,
            "correction_model_id": correction_model_id,
            "teacher_id": teacher_id,
            "benchmark_suite_id": benchmark_suite.suite_id,
            "train_config": asdict(train_config),
            "base_examples": len(base_examples),
            "max_steps": max_steps,
            "policy_mode": "model_only_required",
            "provenance": provenance,
            "sequential_task": sequential_task,
            "parent_artifact_id": parent_artifact_id,
        },
        producer="trainer",
    )

    try:
        rollout_path = run_logged_intervention_agent(
            root,
            policy=policy,
            teacher=teacher,
            teacher_id=teacher_id,
            executor=executor,
            evaluator=evaluator,
            context=context,
            max_steps=max_steps,
        )
        rollout_summary = _read_json(rollout_path / "run.json")
        rollout_run_id = str(rollout_summary["run_id"])
        traces = read_intervention_traces_jsonl(rollout_path / "interventions.jsonl")
        if not traces:
            raise ValueError("Correction-transfer rollout produced no intervention traces")
        if any(trace.route != "model" for trace in traces):
            raise ValueError("Correction-transfer rollout must be model-only")
        if any(trace.executed_action != trace.student_action for trace in traces):
            raise ValueError("Model-only rollout must execute the student's selected Action")

        intervention_examples = examples_from_interventions(traces)
        if not intervention_examples:
            raise ValueError("Correction-transfer rollout produced no trainable corrections")
        added_examples = len(intervention_examples)

        parent_benchmark = run_decision_benchmark(
            parent_model,
            tokenizer,
            benchmark_suite,
            model_id=parent_model_id,
        )
        write_benchmark_artifacts(run.path / "benchmark-parent", parent_benchmark)

        replay = train_replay_control_candidate(
            parent_model,
            tokenizer,
            base_examples,
            added_examples=added_examples,
            config=train_config,
        )
        correction = train_intervention_candidate(
            parent_model,
            tokenizer,
            base_examples,
            traces,
            config=train_config,
        )
        if len(replay.training_examples) != len(correction.training_examples):
            raise RuntimeError("Replay and correction training-example counts must match")
        if len(replay.losses) != len(correction.losses):
            raise RuntimeError("Replay and correction optimizer-step counts must match")

        replay_training = {
            "schema_version": "aster-replay-control-training-0",
            "parent_model_id": parent_model_id,
            "candidate_model_id": replay_model_id,
            "rollout_run_id": rollout_run_id,
            "config": asdict(train_config),
            "base_examples": len(base_examples),
            "added_examples": added_examples,
            "update": replay.summary(),
            "losses": list(replay.losses),
        }
        correction_training = {
            "schema_version": "aster-correction-training-0",
            "parent_model_id": parent_model_id,
            "candidate_model_id": correction_model_id,
            "rollout_run_id": rollout_run_id,
            "config": asdict(train_config),
            "base_examples": len(base_examples),
            "added_examples": added_examples,
            "update": correction.summary(),
            "losses": list(correction.losses),
        }
        _write_json(run.path / "training-replay.json", replay_training)
        _write_json(run.path / "training-correction.json", correction_training)
        _write_examples_jsonl(
            run.path / "training-examples-replay.jsonl", replay.training_examples
        )
        _write_examples_jsonl(
            run.path / "training-examples-correction.jsonl", correction.training_examples
        )

        replay_benchmark = run_decision_benchmark(
            replay.model,
            tokenizer,
            benchmark_suite,
            model_id=replay_model_id,
        )
        correction_benchmark = run_decision_benchmark(
            correction.model,
            tokenizer,
            benchmark_suite,
            model_id=correction_model_id,
        )
        write_benchmark_artifacts(run.path / "benchmark-replay", replay_benchmark)
        write_benchmark_artifacts(run.path / "benchmark-correction", correction_benchmark)

        suite_sha256 = digest(json_bytes(benchmark_suite.to_dict()))
        replay_model, replay_tokenizer, replay_artifact = _persist_candidate_artifact(
            root,
            run.path,
            arm="replay",
            model=replay.model,
            tokenizer=tokenizer,
            model_id=replay_model_id,
            suite_id=benchmark_suite.suite_id,
            suite_sha256=suite_sha256,
            train_config=train_config,
            temperature=replay_benchmark.temperature,
        )
        correction_model, correction_tokenizer, correction_artifact = (
            _persist_candidate_artifact(
                root,
                run.path,
                arm="correction",
                model=correction.model,
                tokenizer=tokenizer,
                model_id=correction_model_id,
                suite_id=benchmark_suite.suite_id,
                suite_sha256=suite_sha256,
                train_config=train_config,
                temperature=correction_benchmark.temperature,
            )
        )
        parent_eval_model = parent_model
        parent_eval_tokenizer = tokenizer
        parent_artifact: dict[str, object] = {
            "artifact_id": parent_artifact_id,
            "registered": parent_artifact_id is not None,
            "reload_verified": False,
        }
        if parent_artifact_id is not None:
            (
                parent_eval_model,
                parent_eval_tokenizer,
                parent_artifact,
            ) = _reload_parent_artifact(
                root,
                artifact_id=parent_artifact_id,
                model=parent_model,
                model_id=parent_model_id,
            )

        candidate_artifacts = {
            "parent": parent_artifact,
            "replay": replay_artifact,
            "correction": correction_artifact,
        }

        repair = {
            "source": "student_visited_teacher_labels",
            "examples": added_examples,
            "parent": evaluate_decisions(
                parent_eval_model, parent_eval_tokenizer, intervention_examples
            ),
            "replay": evaluate_decisions(
                replay_model, replay_tokenizer, intervention_examples
            ),
            "correction": evaluate_decisions(
                correction_model, correction_tokenizer, intervention_examples
            ),
        }

        sequential_transfer = None
        sequential_run_ids: dict[str, str] = {}
        if sequential_task is not None:
            sequential_arms: dict[str, object] = {}
            for (
                arm,
                arm_model,
                arm_tokenizer,
                arm_model_id,
                arm_artifact_id,
            ) in (
                (
                    "parent",
                    parent_eval_model,
                    parent_eval_tokenizer,
                    parent_model_id,
                    parent_artifact_id,
                ),
                (
                    "replay",
                    replay_model,
                    replay_tokenizer,
                    replay_model_id,
                    replay_artifact["artifact_id"],
                ),
                (
                    "correction",
                    correction_model,
                    correction_tokenizer,
                    correction_model_id,
                    correction_artifact["artifact_id"],
                ),
            ):
                episode_path = run_logged_model_only_sequential_episode(
                    root,
                    model=arm_model,
                    tokenizer=arm_tokenizer,
                    model_id=arm_model_id,
                    artifact_id=(
                        arm_artifact_id
                        if isinstance(arm_artifact_id, str)
                        else None
                    ),
                    arm=arm,
                    task=sequential_task,
                    teacher=teacher,
                    teacher_id=teacher_id,
                    executor=executor,
                    evaluator=evaluator,
                    max_steps=max_steps,
                    lineage={
                        "correction_transfer_run_id": run.id,
                        "rollout_run_id": rollout_run_id,
                    },
                )
                episode_run = _read_json(episode_path / "run.json")
                episode_run_id = str(episode_run["run_id"])
                sequential_run_ids[arm] = episode_run_id
                sequential_arms[arm] = {
                    "run_id": episode_run_id,
                    "summary": _read_json(episode_path / "sequential.json"),
                }
            sequential_transfer = {
                "scope": "model_only_uncorrected_sibling_episode",
                "task": dict(sequential_task),
                "arms": sequential_arms,
            }

        comparisons = {
            "parent_to_replay": compare_benchmark_bundles(
                parent_benchmark, replay_benchmark
            ),
            "parent_to_correction": compare_benchmark_bundles(
                parent_benchmark, correction_benchmark
            ),
            "replay_to_correction": compare_benchmark_bundles(
                replay_benchmark, correction_benchmark
            ),
        }
        experiment = {
            "schema_version": "aster-correction-transfer-experiment-0",
            "parent_model_id": parent_model_id,
            "replay_model_id": replay_model_id,
            "correction_model_id": correction_model_id,
            "teacher_id": teacher_id,
            "rollout_run_id": rollout_run_id,
            "benchmark_suite_id": benchmark_suite.suite_id,
            "policy_mode": "model_only",
            "provenance": provenance,
            "candidate_artifacts": candidate_artifacts,
            "repair": repair,
            "sequential_transfer": sequential_transfer,
            "budget": {
                "base_examples": len(base_examples),
                "added_examples_per_candidate": added_examples,
                "training_examples_per_candidate": len(replay.training_examples),
                "optimizer_steps_per_candidate": len(replay.losses),
                "measured_resources": {
                    "replay": replay.resources,
                    "correction": correction.resources,
                },
                "equal_flops_claimed": False,
            },
            "artifacts": {
                "replay_training": "training-replay.json",
                "correction_training": "training-correction.json",
                "replay_examples": "training-examples-replay.jsonl",
                "correction_examples": "training-examples-correction.jsonl",
                "parent_benchmark": "benchmark-parent/benchmark.json",
                "replay_benchmark": "benchmark-replay/benchmark.json",
                "correction_benchmark": "benchmark-correction/benchmark.json",
                "replay_candidate_manifest": "candidate-artifacts/replay/manifest.json",
                "correction_candidate_manifest": "candidate-artifacts/correction/manifest.json",
            },
            "comparisons": comparisons,
            "promotion": "candidate_only",
        }
        _write_json(run.path / "correction-transfer.json", experiment)
        run.event(
            "correction_transfer_compared",
            {
                "rollout_run_id": rollout_run_id,
                "added_examples": added_examples,
                "training_examples_per_candidate": len(replay.training_examples),
                "optimizer_steps_per_candidate": len(replay.losses),
                "repair": repair,
                "candidate_artifacts": candidate_artifacts,
                "sequential_transfer": sequential_transfer,
            },
        )
        run.finish(
            "completed",
            experiment="correction-transfer.json",
            rollout_run_id=rollout_run_id,
            replay_training="training-replay.json",
            correction_training="training-correction.json",
            parent_benchmark="benchmark-parent/benchmark.json",
            replay_benchmark="benchmark-replay/benchmark.json",
            correction_benchmark="benchmark-correction/benchmark.json",
            replay_artifact_id=replay_artifact["artifact_id"],
            correction_artifact_id=correction_artifact["artifact_id"],
            sequential_runs=sequential_run_ids or None,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise



def _reload_parent_artifact(
    root: Path,
    *,
    artifact_id: str,
    model: DecisionModel,
    model_id: str,
) -> tuple[DecisionModel, AsterTokenizer, dict[str, object]]:
    """Resolve and exactly verify the already-registered P0 parent artifact."""
    catalog = ArtifactCatalog(root)
    registered = catalog.resolve(artifact_id, "decision_model")
    loaded_model, loaded_tokenizer, loaded_manifest = load_decision_artifact(registered)
    if loaded_manifest.get("artifact_id") != artifact_id:
        raise RuntimeError("Reloaded parent artifact identity mismatch")
    if loaded_manifest.get("model_id") != model_id:
        raise RuntimeError("Reloaded parent model ID mismatch")
    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    if original_state.keys() != loaded_state.keys() or any(
        not torch.equal(original_state[name].detach().cpu(), loaded_state[name].detach().cpu())
        for name in original_state
    ):
        raise RuntimeError("Reloaded parent weights differ from experiment parent")
    return loaded_model, loaded_tokenizer, {
        "artifact_id": artifact_id,
        "registered": True,
        "reload_verified": True,
    }

def _persist_candidate_artifact(
    root: Path,
    run_path: Path,
    *,
    arm: str,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    model_id: str,
    suite_id: str,
    suite_sha256: str,
    train_config: DecisionTrainConfig,
    temperature: float,
) -> tuple[DecisionModel, AsterTokenizer, dict[str, object]]:
    """Save, register, reload, and exactly verify one non-promoted candidate."""
    source = run_path / "candidate-artifacts" / arm
    manifest = save_decision_artifact(
        source,
        model,
        tokenizer,
        model_id=model_id,
        candidate_builder_id=_DECISION_CANDIDATE_BUILDER_ID,
        suite_id=suite_id,
        suite_sha256=suite_sha256,
        train_config=asdict(train_config),
        temperature=temperature,
        autonomous_threshold=0.0,
        fallback_threshold=0.0,
        initialization=f"correction_transfer_{arm}_v0",
    )
    catalog = ArtifactCatalog(root)
    ref = catalog.register_decision_model(source)
    registered = catalog.resolve(ref.artifact_id, "decision_model")
    loaded_model, loaded_tokenizer, loaded_manifest = load_decision_artifact(registered)

    if manifest.get("artifact_id") != ref.artifact_id:
        raise RuntimeError("Candidate source and registered artifact IDs differ")
    if loaded_manifest.get("artifact_id") != ref.artifact_id:
        raise RuntimeError("Reloaded candidate artifact identity mismatch")
    if loaded_manifest.get("model_id") != model_id:
        raise RuntimeError("Reloaded candidate model ID mismatch")
    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    if original_state.keys() != loaded_state.keys() or any(
        not torch.equal(original_state[name].detach().cpu(), loaded_state[name].detach().cpu())
        for name in original_state
    ):
        raise RuntimeError("Reloaded candidate weights differ from trained candidate")

    return loaded_model, loaded_tokenizer, {
        "artifact_id": ref.artifact_id,
        "source_manifest": f"candidate-artifacts/{arm}/manifest.json",
        "registered": True,
        "reload_verified": True,
        "temperature": temperature,
    }

def compare_benchmark_bundles(
    parent: BenchmarkBundle,
    candidate: BenchmarkBundle,
) -> dict:
    """Compare the same benchmark without turning metric deltas into a promotion verdict."""
    if parent.suite_id != candidate.suite_id:
        raise ValueError("Benchmark comparison requires the same suite")

    parent_slices = set(parent.by_slice)
    candidate_slices = set(candidate.by_slice)
    if parent_slices != candidate_slices:
        raise ValueError("Benchmark comparison requires matching slices")

    return {
        "delta_definition": "candidate_minus_parent",
        "temperature": {
            "parent": parent.temperature,
            "candidate": candidate.temperature,
            "delta": candidate.temperature - parent.temperature,
        },
        "test": {
            "raw": _compare_metric_summary(parent.test_raw, candidate.test_raw),
            "calibrated": _compare_metric_summary(
                parent.test_calibrated, candidate.test_calibrated
            ),
        },
        "by_slice": {
            slice_name: {
                "raw": _compare_metric_summary(
                    parent.by_slice[slice_name]["raw"],
                    candidate.by_slice[slice_name]["raw"],
                ),
                "calibrated": _compare_metric_summary(
                    parent.by_slice[slice_name]["calibrated"],
                    candidate.by_slice[slice_name]["calibrated"],
                ),
            }
            for slice_name in sorted(parent_slices)
        },
    }


def _compare_metric_summary(parent: dict, candidate: dict) -> dict:
    if parent.get("examples") != candidate.get("examples"):
        raise ValueError("Metric comparison requires the same example count")
    return {
        "parent": parent,
        "candidate": candidate,
        "delta": {
            name: float(candidate[name]) - float(parent[name])
            for name in _SCALAR_METRICS
        },
    }


def _write_examples_jsonl(path: Path, examples: Sequence[DecisionExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
