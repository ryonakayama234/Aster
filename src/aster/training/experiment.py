"""End-to-end supervised intervention experiment with held-out v0/v+1 comparison."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from aster.agent.policy import Policy
from aster.agent.selective import SelectivePolicy
from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.runner import (
    BenchmarkBundle,
    run_decision_benchmark,
    write_benchmark_artifacts,
)
from aster.evaluator.verifier import TaskEvaluator
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample
from aster.records.intervention import read_intervention_traces_jsonl
from aster.records.runlog import RunLog
from aster.runtime.context import RuntimeContext
from aster.runtime.learning import run_logged_intervention_agent
from aster.tokenizer.artifact import AsterTokenizer
from aster.tools.executor import ToolExecutor
from aster.training.decision import DecisionTrainConfig
from aster.training.intervention import train_intervention_candidate

_SCALAR_METRICS = ("accuracy", "nll", "brier", "ece")


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
