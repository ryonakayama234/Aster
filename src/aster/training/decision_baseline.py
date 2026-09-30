"""Leakage-safe CPU baseline for a reloadable AsterDecision model.

The development run trains only on ``train``, fits one temperature only on
``calibration``, evaluates design choices on ``dev``, and leaves ``test`` sealed.
A separate final-test entry point evaluates the saved artifact without updating it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import platform
import resource
from pathlib import Path
import time
from typing import Sequence, cast

import torch

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.agent.selective import SelectivePolicy, SelectivePolicyConfig
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.metrics import (
    DecisionPrediction,
    apply_temperature,
    fit_temperature,
    probabilities_from_logits,
    summarize_predictions,
)
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy, score_candidates
from aster.model.decision_artifact import load_decision_artifact, save_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import AsterTokenizer, train_aster_tokenizer
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.training.decision import DecisionTrainConfig, train_decision


CANDIDATE_BUILDER_ID = "calculate-and-store-v0"
BASELINE_SCHEMA = "aster-decision-baseline-0"
FINAL_TEST_SCHEMA = "aster-decision-baseline-final-test-0"
_SPLITS = ("train", "calibration", "dev", "test")


@dataclass(frozen=True, slots=True)
class DecisionBaselineConfig:
    steps: int = 200
    learning_rate: float = 3e-3
    train_backbone: bool = True
    seed: int = 42
    target_vocab_size: int = 512
    min_pair_frequency: int = 1
    context_length: int = 2048
    width: int = 16
    heads: int = 2
    layers: int = 1
    autonomous_threshold: float = 0.8
    fallback_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.learning_rate <= 0:
            raise ValueError("Decision baseline training settings must be positive")
        if self.target_vocab_size < 256 or self.min_pair_frequency <= 0:
            raise ValueError("Decision baseline tokenizer settings are invalid")
        ModelConfig(
            vocab_size=self.target_vocab_size,
            context_length=self.context_length,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
        )
        if not 0.0 <= self.fallback_threshold <= self.autonomous_threshold <= 1.0:
            raise ValueError("routing thresholds must satisfy 0 <= fallback <= autonomous <= 1")

    def train_config(self) -> DecisionTrainConfig:
        return DecisionTrainConfig(
            steps=self.steps,
            learning_rate=self.learning_rate,
            train_backbone=self.train_backbone,
            seed=self.seed,
        )


def run_logged_decision_baseline(
    root: str | Path,
    suite: BenchmarkSuite,
    *,
    config: DecisionBaselineConfig = DecisionBaselineConfig(),
    model_id: str = "AsterDecision-baseline-v0",
    source_git_sha: str | None = None,
) -> Path:
    """Train/calibrate/dev-evaluate one CPU model while keeping test sealed."""

    _require_development_splits(suite)
    root = Path(root)
    suite_sha256 = _suite_sha256(suite)
    split_manifest = _split_manifest(suite, suite_sha256)
    run = RunLog(
        root,
        "decision_baseline",
        {
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "model_id": model_id,
            "config": asdict(config),
            "test_policy": "sealed",
        },
        producer="trainer",
    )
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    try:
        _write_json(run.path / "split-manifest.json", split_manifest)

        train_cases = suite.cases_for("train")
        train_texts = _serialized_texts(train_cases)
        tokenizer_start = time.perf_counter()
        tokenizer = train_aster_tokenizer(
            train_texts,
            target_vocab_size=config.target_vocab_size,
            min_pair_frequency=config.min_pair_frequency,
            name="AsterDecisionBaselineTokenizer",
            version="0",
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_start

        torch.manual_seed(config.seed)
        model = DecisionModel(
            TinyLM(
                ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_length=config.context_length,
                    width=config.width,
                    heads=config.heads,
                    layers=config.layers,
                )
            )
        )
        _check_context_capacity(model, tokenizer, train_cases, "train")

        initial_start = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            initial_train = _predict_cases(model, tokenizer, train_cases)
        initial_train_metrics = _train_metrics(initial_train)
        _write_train_predictions(run.path / "initial-train-predictions.jsonl", train_cases, initial_train)
        initial_train_seconds = time.perf_counter() - initial_start

        train_start = time.perf_counter()
        losses = train_decision(
            model,
            tokenizer,
            suite.training_examples(),
            config.train_config(),
        )
        train_seconds = time.perf_counter() - train_start

        train_eval_start = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            final_train = _predict_cases(model, tokenizer, train_cases)
        final_train_metrics = _train_metrics(final_train)
        _write_train_predictions(run.path / "train-predictions.jsonl", train_cases, final_train)
        train_eval_seconds = time.perf_counter() - train_eval_start

        calibration_start = time.perf_counter()
        calibration_raw = _predict_cases(
            model, tokenizer, suite.cases_for("calibration")
        )
        temperature = fit_temperature(calibration_raw)
        calibration = apply_temperature(calibration_raw, temperature)
        calibration_metrics = {
            "raw": summarize_predictions(calibration, calibrated=False),
            "calibrated": summarize_predictions(calibration, calibrated=True),
        }
        calibration_seconds = time.perf_counter() - calibration_start

        dev_start = time.perf_counter()
        dev_raw = _predict_cases(model, tokenizer, suite.cases_for("dev"))
        dev_predictions = apply_temperature(dev_raw, temperature)
        dev_metrics = {
            "raw": summarize_predictions(dev_predictions, calibrated=False),
            "calibrated": summarize_predictions(dev_predictions, calibrated=True),
        }
        dev_decision_seconds = time.perf_counter() - dev_start

        artifact_dir = run.path / "model-artifact"
        artifact_manifest = save_decision_artifact(
            artifact_dir,
            model,
            tokenizer,
            model_id=model_id,
            candidate_builder_id=CANDIDATE_BUILDER_ID,
            suite_id=suite.suite_id,
            suite_sha256=suite_sha256,
            train_config=asdict(config),
            temperature=temperature,
            autonomous_threshold=config.autonomous_threshold,
            fallback_threshold=config.fallback_threshold,
            initialization="scratch",
            source_git_sha=source_git_sha,
        )

        reload_model, reload_tokenizer, reload_manifest = load_decision_artifact(
            artifact_dir
        )
        reload_check = _reload_check(
            model,
            tokenizer,
            reload_model,
            reload_tokenizer,
            suite.cases_for("dev")[0],
        )

        episode_start = time.perf_counter()
        dev_episodes = _compare_episode_policies(
            reload_model,
            reload_tokenizer,
            suite,
            split="dev",
            model_id=model_id,
            temperature=temperature,
            autonomous_threshold=config.autonomous_threshold,
            fallback_threshold=config.fallback_threshold,
        )
        dev_episode_seconds = time.perf_counter() - episode_start

        training_payload = {
            "schema_version": "aster-decision-baseline-training-0",
            "model_id": model_id,
            "suite_id": suite.suite_id,
            "training_split": "train",
            "tokenizer_training_split": "train",
            "config": asdict(config),
            "losses": losses,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "loss_semantics": "online pre-update single-example loss; not final train NLL",
            "initial_train": initial_train_metrics,
            "final_train": final_train_metrics,
            "optimizer_updates": config.steps,
            "example_exposures": config.steps,
            "equivalent_epochs": config.steps / len(train_cases),
        }
        _write_json(run.path / "training.json", training_payload)
        _write_predictions(
            run.path / "development-predictions.jsonl",
            calibration + dev_predictions,
        )

        total_wall_seconds = time.perf_counter() - total_wall_start
        total_cpu_seconds = time.process_time() - total_cpu_start
        resources = {
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "tokenizer_wall_seconds": tokenizer_seconds,
            "train_wall_seconds": train_seconds,
            "initial_train_evaluation_wall_seconds": initial_train_seconds,
            "final_train_evaluation_wall_seconds": train_eval_seconds,
            "mean_train_step_wall_seconds": train_seconds / config.steps,
            "calibration_wall_seconds": calibration_seconds,
            "dev_decision_wall_seconds": dev_decision_seconds,
            "dev_episode_wall_seconds": dev_episode_seconds,
            "total_wall_seconds": total_wall_seconds,
            "total_process_cpu_seconds": total_cpu_seconds,
            "process_max_rss_kib": _max_rss_kib(),
        }
        baseline = {
            "schema_version": BASELINE_SCHEMA,
            "model_id": model_id,
            "artifact_id": artifact_manifest["artifact_id"],
            "artifact_dir": "model-artifact",
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "split_manifest": "split-manifest.json",
            "training": "training.json",
            "train": {
                "initial": initial_train_metrics,
                "final": final_train_metrics,
                "initial_predictions": "initial-train-predictions.jsonl",
                "predictions": "train-predictions.jsonl",
                "temperature": 1.0,
                "scope": "memorization diagnostic, not generalization evidence",
            },
            "development_predictions": "development-predictions.jsonl",
            "data_policy": {
                "model_update": "train",
                "tokenizer_training": "train",
                "temperature_fit": "calibration",
                "design_observation": "dev",
                "test": "sealed",
            },
            "calibration": {
                "temperature": temperature,
                "metrics": calibration_metrics,
            },
            "dev": {
                "decision": dev_metrics,
                "episodes": dev_episodes,
            },
            "reload_check": reload_check,
            "artifact_reload_id": reload_manifest["artifact_id"],
            "test": {
                "status": "sealed",
                "instruction": "Use run_logged_decision_final_test on this saved artifact after design is frozen.",
            },
            "resources": resources,
        }
        _write_json(run.path / "baseline.json", baseline)
        run.event(
            "baseline_measured",
            {
                "model_id": model_id,
                "artifact_id": artifact_manifest["artifact_id"],
                "temperature": temperature,
                "dev_accuracy": dev_metrics["calibrated"]["accuracy"],
                "test_status": "sealed",
            },
        )
        run.finish(
            "completed",
            baseline="baseline.json",
            artifact="model-artifact/manifest.json",
            split_manifest="split-manifest.json",
            training="training.json",
            train_predictions="train-predictions.jsonl",
            initial_train_predictions="initial-train-predictions.jsonl",
            development_predictions="development-predictions.jsonl",
            test_status="sealed",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise



def _raw_nll(prediction: DecisionPrediction) -> float:
    """Stable log-sum-exp, without clipping small target probabilities."""
    maximum = max(prediction.logits)
    return (maximum - prediction.logits[prediction.target_index]
            + math.log(sum(math.exp(value - maximum) for value in prediction.logits)))


def _train_metrics(predictions: Sequence[DecisionPrediction]) -> dict[str, float | int]:
    if not predictions:
        raise ValueError("Train diagnostics require at least one case")
    correct = sum(int(row.correct) for row in predictions)
    return {
        "examples": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "nll": sum(_raw_nll(row) for row in predictions) / len(predictions),
    }


def _write_train_predictions(
    path: Path,
    cases: Sequence[BenchmarkCase],
    predictions: Sequence[DecisionPrediction],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for case, prediction in zip(cases, predictions, strict=True):
            if case.case_id != prediction.case_id or case.split != "train":
                raise ValueError("Train prediction does not match its case")
            alternatives = [score for index, score in enumerate(prediction.logits)
                            if index != prediction.target_index]
            row = prediction.to_dict()
            row.update(
                target_action=case.example.target.to_dict(),
                predicted_action=case.example.candidates[prediction.predicted_index].to_dict(),
                candidates=[candidate.to_dict() for candidate in case.example.candidates],
                target_probability=prediction.raw_probabilities[prediction.target_index],
                nll=_raw_nll(prediction),
                target_margin=(prediction.logits[prediction.target_index] - max(alternatives)
                               if alternatives else None),
                step=case.example.state.step,
                leakage_group=case.leakage_group,
            )
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_logged_decision_train_diagnostics(
    root: str | Path,
    artifact_dir: str | Path,
    suite: BenchmarkSuite,
    *,
    source_git_sha: str | None = None,
) -> Path:
    """Read a saved artifact and evaluate train only, without training/calibration."""
    model, tokenizer, manifest = load_decision_artifact(artifact_dir)
    if manifest.get("suite_id") != suite.suite_id or manifest.get("suite_sha256") != _suite_sha256(suite):
        raise ValueError("Train diagnostic suite does not match the model artifact lineage")
    if manifest.get("candidate_builder_id") != CANDIDATE_BUILDER_ID:
        raise ValueError("Unsupported candidate builder for train diagnostics")
    cases = suite.cases_for("train")
    if not cases:
        raise ValueError("Train diagnostics require a non-empty train split")
    run = RunLog(Path(root), "decision_train_diagnostics", {
        "artifact_id": manifest["artifact_id"],
        "suite_id": suite.suite_id,
        "suite_sha256": manifest["suite_sha256"],
        "source_git_sha": source_git_sha,
        "evaluated_split": "train",
    }, producer="evaluator")
    started = time.perf_counter()
    try:
        predictions = _predict_cases(model, tokenizer, cases)
        _write_train_predictions(run.path / "train-predictions.jsonl", cases, predictions)
        payload = {
            "schema_version": "aster-decision-train-diagnostics-0",
            "artifact_id": manifest["artifact_id"],
            "suite_id": suite.suite_id,
            "suite_sha256": manifest["suite_sha256"],
            "source_git_sha": source_git_sha,
            "split": "train",
            "temperature": 1.0,
            "metrics": _train_metrics(predictions),
            "initial_train": None,
            "initial_train_status": "unavailable_from_final_artifact",
            "model_update": False,
            "temperature_fit": False,
            "test": {"status": "sealed"},
            "resources": {
                "device": "cpu",
                "torch_num_threads": torch.get_num_threads(),
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "wall_seconds": time.perf_counter() - started,
            },
        }
        _write_json(run.path / "train-diagnostics.json", payload)
        run.finish("completed", diagnostics="train-diagnostics.json",
                   predictions="train-predictions.jsonl", test_status="sealed")
        return run.path
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error_type=type(error).__name__, error=str(error))
        raise


def run_logged_decision_final_test(
    root: str | Path,
    artifact_dir: str | Path,
    suite: BenchmarkSuite,
) -> Path:
    """Evaluate only the sealed test split of one already-saved artifact."""

    root = Path(root)
    model, tokenizer, manifest = load_decision_artifact(artifact_dir)
    suite_sha256 = _suite_sha256(suite)
    if manifest.get("suite_id") != suite.suite_id or manifest.get("suite_sha256") != suite_sha256:
        raise ValueError("Final test suite does not match the model artifact lineage")
    if manifest.get("candidate_builder_id") != CANDIDATE_BUILDER_ID:
        raise ValueError("Unsupported candidate builder for final test")

    temperature = _nested_float(manifest, "calibration", "temperature")
    autonomous_threshold = _nested_float(manifest, "routing", "autonomous_threshold")
    fallback_threshold = _nested_float(manifest, "routing", "fallback_threshold")
    model_id = _require_manifest_str(manifest, "model_id")
    artifact_id = _require_manifest_str(manifest, "artifact_id")

    run = RunLog(
        root,
        "decision_baseline_final_test",
        {
            "artifact_id": artifact_id,
            "model_id": model_id,
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "evaluated_split": "test",
        },
        producer="evaluator",
    )
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        test_raw = _predict_cases(model, tokenizer, suite.cases_for("test"))
        test_predictions = apply_temperature(test_raw, temperature)
        test_metrics = {
            "raw": summarize_predictions(test_predictions, calibrated=False),
            "calibrated": summarize_predictions(test_predictions, calibrated=True),
        }
        test_episodes = _compare_episode_policies(
            model,
            tokenizer,
            suite,
            split="test",
            model_id=model_id,
            temperature=temperature,
            autonomous_threshold=autonomous_threshold,
            fallback_threshold=fallback_threshold,
        )
        _write_predictions(run.path / "test-predictions.jsonl", test_predictions)
        payload = {
            "schema_version": FINAL_TEST_SCHEMA,
            "artifact_id": artifact_id,
            "model_id": model_id,
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "split": "test",
            "temperature": temperature,
            "decision": test_metrics,
            "episodes": test_episodes,
            "resources": {
                "device": "cpu",
                "torch_num_threads": torch.get_num_threads(),
                "wall_seconds": time.perf_counter() - wall_start,
                "process_cpu_seconds": time.process_time() - cpu_start,
                "process_max_rss_kib": _max_rss_kib(),
            },
        }
        _write_json(run.path / "final-test.json", payload)
        run.event(
            "test_revealed",
            {
                "artifact_id": artifact_id,
                "examples": test_metrics["calibrated"]["examples"],
                "accuracy": test_metrics["calibrated"]["accuracy"],
            },
        )
        run.finish(
            "completed",
            final_test="final-test.json",
            predictions="test-predictions.jsonl",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _predict_cases(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    cases: Sequence[BenchmarkCase],
) -> tuple[DecisionPrediction, ...]:
    if not cases:
        raise ValueError("Decision evaluation requires at least one case")
    _check_context_capacity(model, tokenizer, cases, cases[0].split)
    was_training = model.training
    model.eval()
    rows: list[DecisionPrediction] = []
    try:
        with torch.no_grad():
            for case in cases:
                example = case.example
                scores = score_candidates(
                    model,
                    tokenizer,
                    example.state,
                    example.trajectory,
                    example.candidates,
                )
                logits = tuple(float(value) for value in scores.detach().cpu().tolist())
                probabilities = probabilities_from_logits(logits)
                predicted_index = max(range(len(logits)), key=logits.__getitem__)
                rows.append(
                    DecisionPrediction(
                        case_id=case.case_id,
                        split=case.split,
                        slice_name=case.slice_name,
                        target_index=example.target_index,
                        predicted_index=predicted_index,
                        logits=logits,
                        raw_probabilities=probabilities,
                    )
                )
    finally:
        if was_training:
            model.train()
    return tuple(rows)


def _compare_episode_policies(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    suite: BenchmarkSuite,
    *,
    split: str,
    model_id: str,
    temperature: float,
    autonomous_threshold: float,
    fallback_threshold: float,
) -> dict[str, object]:
    seeds = _episode_seeds(suite, split)
    if not seeds:
        raise ValueError(f"No episode seeds for split {split!r}")

    rows: list[dict[str, object]] = []
    aggregates: dict[str, dict[str, object]] = {}
    candidate_total = 0
    candidate_present = 0
    missing_teacher_actions: list[str] = []
    for policy_name in ("rule", "model", "model+fallback"):
        successes = 0
        steps = 0
        route_counts = {"model": 0, "fallback": 0, "abstain": 0}
        for seed in seeds:
            model_policy = ModelPolicy(
                model,
                tokenizer,
                candidate_builder=CalculateAndStoreCandidates(),
                model_id=model_id,
            )
            selective: SelectivePolicy | None = None
            if policy_name == "rule":
                policy = RuleBasedPolicy()
            elif policy_name == "model":
                policy = model_policy
            else:
                selective = SelectivePolicy(
                    model_policy,
                    fallback=RuleBasedPolicy(),
                    fallback_id="rule-v0:fallback",
                    config=SelectivePolicyConfig(
                        temperature=temperature,
                        autonomous_threshold=autonomous_threshold,
                        fallback_threshold=fallback_threshold,
                    ),
                )
                policy = selective

            trajectory = run_loop(
                policy=policy,
                executor=_build_executor(),
                evaluator=TaskEvaluator(),
                context=RuntimeContext(
                    task=cast(dict[str, JsonValue], seed["task"]),
                    memory=cast(dict[str, JsonValue], seed["memory"]),
                ),
            )
            evaluation = evaluate_episode(trajectory)
            successes += int(evaluation.task_success)
            steps += evaluation.steps
            if policy_name == "rule":
                coverage = _candidate_coverage_for_trajectory(
                    trajectory, str(seed["episode_id"])
                )
                candidate_total += cast(int, coverage["teacher_actions"])
                candidate_present += cast(int, coverage["present"])
                missing_teacher_actions.extend(cast(list[str], coverage["missing"]))
            if selective is not None:
                for trace in selective.routing_traces:
                    route_counts[trace.route] += 1
            rows.append(
                {
                    "episode_id": seed["episode_id"],
                    "slice": seed["slice"],
                    "policy": policy_name,
                    "task_success": evaluation.task_success,
                    "goal_verified": evaluation.goal_verified,
                    "terminal_reached": evaluation.terminal_reached,
                    "steps": evaluation.steps,
                    "stop_reason": evaluation.stop_reason,
                }
            )
        total_routes = sum(route_counts.values())
        aggregates[policy_name] = {
            "episodes": len(seeds),
            "episode_successes": successes,
            "episode_success_rate": successes / len(seeds),
            "mean_steps": steps / len(seeds),
            "routes": route_counts,
            "model_route_rate": None if not total_routes else route_counts["model"] / total_routes,
            "fallback_route_rate": None if not total_routes else route_counts["fallback"] / total_routes,
            "abstain_route_rate": None if not total_routes else route_counts["abstain"] / total_routes,
        }

    return {
        "split": split,
        "episodes": rows,
        "summary": aggregates,
        "candidate_builder": {
            "candidate_builder_id": CANDIDATE_BUILDER_ID,
            "teacher_actions": candidate_total,
            "present": candidate_present,
            "coverage": None if not candidate_total else candidate_present / candidate_total,
            "missing": missing_teacher_actions,
        },
        "note": (
            "Episode tasks are reconstructed from step-0 benchmark states. Candidate-set and "
            "candidate-order perturbations remain decision-level slices; runtime uses the canonical builder."
        ),
    }


def _candidate_coverage_for_trajectory(
    trajectory: Trajectory,
    episode_id: str,
) -> dict[str, object]:
    builder = CalculateAndStoreCandidates()
    present = 0
    missing: list[str] = []
    for index, transition in enumerate(trajectory.transitions):
        state_data = transition.state_before
        task = state_data.get("task")
        memory = state_data.get("memory")
        if not isinstance(task, dict) or not isinstance(memory, dict):
            raise ValueError("Recorded runtime state must contain task and memory objects")
        state = RuntimeState(
            task=cast(dict[str, JsonValue], task),
            memory=cast(dict[str, JsonValue], memory),
            step=transition.step,
        )
        history = Trajectory(trajectory.transitions[:index])
        candidates = builder.build(state, history, transition.available_actions)
        if transition.action in candidates:
            present += 1
        else:
            missing.append(f"{episode_id}/step-{transition.step}")
    return {
        "teacher_actions": len(trajectory.transitions),
        "present": present,
        "missing": missing,
    }


def _episode_seeds(suite: BenchmarkSuite, split: str) -> list[dict[str, object]]:
    seen: set[str] = set()
    seeds: list[dict[str, object]] = []
    for case in suite.cases_for(split):
        if case.leakage_group in seen or case.example.state.step != 0:
            continue
        seen.add(case.leakage_group)
        seeds.append(
            {
                "episode_id": case.leakage_group,
                "slice": case.slice_name,
                "task": dict(case.example.state.task),
                "memory": dict(case.example.state.memory),
            }
        )
    return seeds


def _build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def _serialized_texts(cases: Sequence[BenchmarkCase]) -> list[str]:
    return [
        serialize_decision_input(
            case.example.state,
            case.example.trajectory,
            candidate,
        )
        for case in cases
        for candidate in case.example.candidates
    ]


def _check_context_capacity(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    cases: Sequence[BenchmarkCase],
    split: str,
) -> None:
    longest = max(
        (
            len(tokenizer.encode(text, add_bos=True, add_eos=True))
            for text in _serialized_texts(cases)
        ),
        default=0,
    )
    if longest > model.backbone.config.context_length:
        raise ValueError(
            f"{split} decision input exceeds fixed context_length: "
            f"{longest} > {model.backbone.config.context_length}"
        )


def _reload_check(
    original_model: DecisionModel,
    original_tokenizer: AsterTokenizer,
    loaded_model: DecisionModel,
    loaded_tokenizer: AsterTokenizer,
    case: BenchmarkCase,
) -> dict[str, object]:
    with torch.no_grad():
        original = score_candidates(
            original_model,
            original_tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()
        loaded = score_candidates(
            loaded_model,
            loaded_tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()
    max_abs_diff = float(torch.max(torch.abs(original - loaded)).item())
    if max_abs_diff > 1e-7:
        raise ValueError("Reloaded DecisionModel scores differ from the saved model")
    return {
        "case_id": case.case_id,
        "max_abs_score_diff": max_abs_diff,
        "selected_index_match": int(torch.argmax(original).item())
        == int(torch.argmax(loaded).item()),
    }


def _split_manifest(suite: BenchmarkSuite, suite_sha256: str) -> dict[str, object]:
    splits: dict[str, object] = {}
    for split in _SPLITS:
        cases = suite.cases_for(split)
        splits[split] = {
            "cases": len(cases),
            "case_ids": [case.case_id for case in cases],
            "leakage_groups": sorted({case.leakage_group for case in cases}),
        }
    return {
        "schema_version": "aster-decision-split-manifest-0",
        "suite_id": suite.suite_id,
        "suite_sha256": suite_sha256,
        "splits": splits,
    }


def _suite_sha256(suite: BenchmarkSuite) -> str:
    return digest(json_bytes(suite.to_dict()))


def _write_predictions(path: Path, predictions: Sequence[DecisionPrediction]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(
                json.dumps(prediction.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _max_rss_kib() -> int | None:
    if platform.system() != "Linux":
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _require_development_splits(suite: BenchmarkSuite) -> None:
    for split in ("train", "calibration", "dev", "test"):
        if not suite.cases_for(split):
            raise ValueError(f"Decision baseline requires a non-empty {split} split")


def _require_manifest_str(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Decision artifact field {key!r} must be a non-empty string")
    return value


def _nested_float(manifest: dict[str, object], outer: str, inner: str) -> float:
    section = manifest.get(outer)
    if not isinstance(section, dict):
        raise ValueError(f"Decision artifact field {outer!r} must be an object")
    value = section.get(inner)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Decision artifact field {outer}.{inner} must be numeric")
    return float(value)
