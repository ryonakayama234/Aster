"""Controlled train-fit experiments for AsterDecision-v0.

This module intentionally stays on the train split. It diagnoses whether the current
model/training recipe can fit a tiny supervised decision set before any claim about
generalization, calibration, routing, or Agent capability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import resource
import time
from typing import Literal, Sequence

import torch

from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.metrics import DecisionPrediction, probabilities_from_logits
from aster.corpus.pipeline import digest, json_bytes
from aster.inference.decide import score_candidates
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.records.runlog import RunLog
from aster.tokenizer.artifact import (
    AsterTokenizer,
    load_tokenizer,
    save_tokenizer,
    train_aster_tokenizer,
)
from aster.training.decision import decision_loss


FIT_STUDY_SCHEMA = "aster-decision-fit-study-0"
FIT_CHECKPOINT_SCHEMA = "aster-decision-fit-checkpoint-0"
TrainingMode = Literal["fixed", "shuffle", "full_batch"]


@dataclass(frozen=True, slots=True)
class DecisionFitStudyConfig:
    epochs: int = 25
    learning_rate: float = 3e-3
    train_backbone: bool = True
    seed: int = 42
    target_vocab_size: int = 512
    min_pair_frequency: int = 1
    context_length: int = 2048
    width: int = 16
    heads: int = 2
    layers: int = 1
    stability_window: int = 3

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.target_vocab_size < 256 or self.min_pair_frequency <= 0:
            raise ValueError("tokenizer settings are invalid")
        if self.stability_window <= 0:
            raise ValueError("stability_window must be positive")
        ModelConfig(
            vocab_size=self.target_vocab_size,
            context_length=self.context_length,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
        )


@dataclass(frozen=True, slots=True)
class DecisionFitArm:
    arm_id: str
    case_ids: tuple[str, ...]
    mode: TrainingMode

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("arm_id must not be empty")
        if not self.case_ids or len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids must be non-empty and unique")
        if self.mode not in ("fixed", "shuffle", "full_batch"):
            raise ValueError(f"Unsupported fit mode: {self.mode}")


def default_fit_arms(
    suite: BenchmarkSuite,
    *,
    include_full_batch: bool = False,
) -> tuple[DecisionFitArm, ...]:
    """Return the pre-registered v0 fit study arms for calculate-and-store-v0."""
    train_ids = tuple(case.case_id for case in suite.cases_for("train"))
    required = (
        "train-add-small/step-0",
        "train-add-small/step-2",
    )
    missing = [case_id for case_id in required if case_id not in train_ids]
    if missing:
        raise ValueError("Default fit arm cases are missing: " + ", ".join(missing))

    arms: list[DecisionFitArm] = [
        DecisionFitArm("one-fixed", (required[0],), "fixed"),
        DecisionFitArm("two-fixed", required, "fixed"),
        DecisionFitArm("eight-fixed", train_ids, "fixed"),
        DecisionFitArm("eight-shuffle", train_ids, "shuffle"),
    ]
    if include_full_batch:
        arms.append(DecisionFitArm("eight-full-batch", train_ids, "full_batch"))
    return tuple(arms)


def run_logged_decision_fit_study(
    root: str | Path,
    suite: BenchmarkSuite,
    *,
    config: DecisionFitStudyConfig = DecisionFitStudyConfig(),
    arms: Sequence[DecisionFitArm] | None = None,
    source_git_sha: str | None = None,
) -> Path:
    """Run train-only fit diagnostics from a shared tokenizer and initial weights."""
    root = Path(root)
    train_cases = suite.cases_for("train")
    if not train_cases:
        raise ValueError("Fit study requires a non-empty train split")

    selected_arms = tuple(arms) if arms is not None else default_fit_arms(suite)
    _validate_arms(selected_arms, train_cases)

    suite_sha256 = digest(json_bytes(suite.to_dict()))
    run = RunLog(
        root,
        "decision_fit_study",
        {
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "config": asdict(config),
            "arms": [asdict(arm) for arm in selected_arms],
            "data_policy": {
                "model_update": "train",
                "tokenizer_training": "train",
                "calibration": "not_run",
                "dev": "not_run",
                "test": "sealed",
            },
        },
        producer="trainer",
    )

    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()
    try:
        tokenizer_start = time.perf_counter()
        tokenizer = train_aster_tokenizer(
            _serialized_texts(train_cases),
            target_vocab_size=config.target_vocab_size,
            min_pair_frequency=config.min_pair_frequency,
            name="AsterDecisionFitTokenizer",
            version="0",
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_start
        save_tokenizer(tokenizer, run.path / "tokenizer")

        audit = _audit_train_cases(train_cases, tokenizer, config.context_length)
        _write_jsonl(run.path / "input-audit.jsonl", audit)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            initial_model = _new_model(tokenizer, config)
        initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in initial_model.state_dict().items()
        }
        initial_model_sha256 = _state_digest(initial_state)

        initial_state_path = run.path / "initial-model.pt"
        torch.save(initial_state, initial_state_path)

        arm_summaries: list[dict[str, object]] = []
        for arm in selected_arms:
            cases = _select_cases(train_cases, arm.case_ids)
            arm_summary = _run_arm(
                run.path,
                suite,
                suite_sha256,
                tokenizer,
                initial_state,
                initial_model_sha256,
                cases,
                arm,
                config,
                source_git_sha=source_git_sha,
            )
            arm_summaries.append(arm_summary)

        resources = {
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "tokenizer_wall_seconds": tokenizer_seconds,
            "total_wall_seconds": time.perf_counter() - total_wall_start,
            "total_process_cpu_seconds": time.process_time() - total_cpu_start,
            "process_max_rss_kib": _max_rss_kib(),
        }
        study = {
            "schema_version": FIT_STUDY_SCHEMA,
            "suite_id": suite.suite_id,
            "suite_sha256": suite_sha256,
            "config": asdict(config),
            "initial_model_sha256": initial_model_sha256,
            "tokenizer": "tokenizer",
            "input_audit": "input-audit.jsonl",
            "arms": arm_summaries,
            "resources": resources,
            "interpretation_scope": (
                "train-fit/optimization diagnostic only; not generalization, calibration, "
                "routing, or broad Agent capability evidence"
            ),
            "test": {"status": "sealed"},
        }
        _write_json(run.path / "study.json", study)
        run.event(
            "fit_study_measured",
            {
                "arms": len(arm_summaries),
                "initial_model_sha256": initial_model_sha256,
                "test_status": "sealed",
            },
        )
        run.finish(
            "completed",
            study="study.json",
            input_audit="input-audit.jsonl",
            tokenizer="tokenizer",
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


def _run_arm(
    run_path: Path,
    suite: BenchmarkSuite,
    suite_sha256: str,
    tokenizer: AsterTokenizer,
    initial_state: dict[str, torch.Tensor],
    initial_model_sha256: str,
    cases: tuple[BenchmarkCase, ...],
    arm: DecisionFitArm,
    config: DecisionFitStudyConfig,
    *,
    source_git_sha: str | None,
) -> dict[str, object]:
    arm_path = run_path / "arms" / arm.arm_id
    arm_path.mkdir(parents=True)
    examples = tuple(case.example for case in cases)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = _new_model(tokenizer, config)
        model.load_state_dict(initial_state, strict=True)

        for parameter in model.backbone.parameters():
            parameter.requires_grad_(config.train_backbone)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        history: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        order_rows: list[dict[str, object]] = []
        optimizer_updates = 0
        example_exposures = 0

        initial_eval_start = time.perf_counter()
        initial_predictions = _predict_cases(model, tokenizer, cases)
        initial_metrics = _fit_metrics(initial_predictions)
        initial_evaluation_wall_seconds = time.perf_counter() - initial_eval_start
        history.append(
            _curve_row(
                epoch=0,
                metrics=initial_metrics,
                optimizer_updates=0,
                example_exposures=0,
                update_loss_mean=None,
            )
        )
        prediction_rows.extend(
            _prediction_rows(epoch=0, cases=cases, predictions=initial_predictions)
        )

        update_wall_seconds = 0.0
        evaluation_wall_seconds = 0.0
        for epoch in range(1, config.epochs + 1):
            model.train()
            update_losses: list[float] = []
            update_start = time.perf_counter()
            if arm.mode == "full_batch":
                order = list(range(len(examples)))
                optimizer.zero_grad(set_to_none=True)
                losses = [decision_loss(model, tokenizer, example) for example in examples]
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                update_losses.append(float(loss.detach().item()))
                optimizer_updates += 1
                example_exposures += len(examples)
            else:
                if arm.mode == "shuffle":
                    order = torch.randperm(len(examples), generator=generator).tolist()
                else:
                    order = list(range(len(examples)))
                for index in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss = decision_loss(model, tokenizer, examples[index])
                    loss.backward()
                    optimizer.step()
                    update_losses.append(float(loss.detach().item()))
                    optimizer_updates += 1
                    example_exposures += 1
            update_wall_seconds += time.perf_counter() - update_start

            order_rows.append(
                {
                    "schema_version": "aster-decision-fit-order-0",
                    "epoch": epoch,
                    "case_ids": [cases[index].case_id for index in order],
                }
            )
            evaluation_start = time.perf_counter()
            predictions = _predict_cases(model, tokenizer, cases)
            metrics = _fit_metrics(predictions)
            evaluation_wall_seconds += time.perf_counter() - evaluation_start
            history.append(
                _curve_row(
                    epoch=epoch,
                    metrics=metrics,
                    optimizer_updates=optimizer_updates,
                    example_exposures=example_exposures,
                    update_loss_mean=sum(update_losses) / len(update_losses),
                )
            )
            prediction_rows.extend(
                _prediction_rows(epoch=epoch, cases=cases, predictions=predictions)
            )

    final_metrics = history[-1]["metrics"]
    if not isinstance(final_metrics, dict):
        raise RuntimeError("Fit history metrics must be an object")
    fit_target_met = _stable_fit(history, config.stability_window)

    _write_jsonl(arm_path / "learning-curve.jsonl", history)
    _write_jsonl(arm_path / "epoch-predictions.jsonl", prediction_rows)
    _write_jsonl(arm_path / "epoch-order.jsonl", order_rows)

    checkpoint_path = arm_path / "checkpoint"
    checkpoint_manifest = _save_fit_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        suite=suite,
        suite_sha256=suite_sha256,
        arm=arm,
        config=config,
        initial_model_sha256=initial_model_sha256,
        source_git_sha=source_git_sha,
    )
    reload_start = time.perf_counter()
    reload_model, reload_tokenizer, reload_manifest = load_decision_fit_checkpoint(
        checkpoint_path
    )
    reload_predictions = _predict_cases(reload_model, reload_tokenizer, cases)
    reload_metrics = _fit_metrics(reload_predictions)
    reload_wall_seconds = time.perf_counter() - reload_start
    reload_match = reload_metrics == final_metrics

    summary = {
        "arm_id": arm.arm_id,
        "mode": arm.mode,
        "case_ids": list(arm.case_ids),
        "examples": len(cases),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "initial_model_sha256": initial_model_sha256,
        "optimizer_updates": optimizer_updates,
        "example_exposures": example_exposures,
        "equivalent_epochs": example_exposures / len(cases),
        "initial": initial_metrics,
        "final": final_metrics,
        "stability_window": config.stability_window,
        "fit_target_met": fit_target_met,
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_id": checkpoint_manifest["checkpoint_id"],
        "reload_checkpoint_id": reload_manifest["checkpoint_id"],
        "reload_metrics_match": reload_match,
        "initial_evaluation_wall_seconds": initial_evaluation_wall_seconds,
        "update_wall_seconds": update_wall_seconds,
        "epoch_end_evaluation_wall_seconds": evaluation_wall_seconds,
        "reload_evaluation_wall_seconds": reload_wall_seconds,
        "learning_curve": "learning-curve.jsonl",
        "epoch_predictions": "epoch-predictions.jsonl",
        "epoch_order": "epoch-order.jsonl",
    }
    _write_json(arm_path / "summary.json", summary)
    return summary


def load_decision_fit_checkpoint(
    input_dir: str | Path,
) -> tuple[DecisionModel, AsterTokenizer, dict[str, object]]:
    input_path = Path(input_dir)
    manifest = json.loads((input_path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != FIT_CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported decision fit checkpoint schema")
    expected = manifest.get("checkpoint_id")
    identity = {key: value for key, value in manifest.items() if key != "checkpoint_id"}
    if expected != f"decision_fit:{digest(json_bytes(identity))}":
        raise ValueError("Decision fit checkpoint identity mismatch")

    model_path = input_path / "model.pt"
    if _sha256_file(model_path) != manifest.get("model_state_sha256"):
        raise ValueError("Decision fit model state digest mismatch")
    tokenizer_path = input_path / "tokenizer"
    if _directory_digest(tokenizer_path) != manifest.get("tokenizer_sha256"):
        raise ValueError("Decision fit tokenizer digest mismatch")
    tokenizer = load_tokenizer(tokenizer_path)

    raw_config = manifest.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Decision fit model_config must be an object")
    model_config = ModelConfig(**raw_config)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("Decision fit tokenizer/model vocabulary mismatch")

    with torch.random.fork_rng(devices=[]):
        model = DecisionModel(TinyLM(model_config))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Decision fit model state must be a mapping")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, manifest


def _save_fit_checkpoint(
    output_dir: Path,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    *,
    suite: BenchmarkSuite,
    suite_sha256: str,
    arm: DecisionFitArm,
    config: DecisionFitStudyConfig,
    initial_model_sha256: str,
    source_git_sha: str | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, output_dir / "model.pt")
    save_tokenizer(tokenizer, output_dir / "tokenizer")
    manifest: dict[str, object] = {
        "schema_version": FIT_CHECKPOINT_SCHEMA,
        "suite_id": suite.suite_id,
        "suite_sha256": suite_sha256,
        "arm": asdict(arm),
        "study_config": asdict(config),
        "initial_model_sha256": initial_model_sha256,
        "model_config": asdict(model.backbone.config),
        "model_state_sha256": _sha256_file(output_dir / "model.pt"),
        "tokenizer_sha256": _directory_digest(output_dir / "tokenizer"),
        "source_git_sha": source_git_sha,
        "calibration": {"status": "not_run"},
        "dev": {"status": "not_run"},
        "test": {"status": "sealed"},
        "resume_supported": False,
        "candidate_only": True,
    }
    manifest["checkpoint_id"] = f"decision_fit:{digest(json_bytes(manifest))}"
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _new_model(
    tokenizer: AsterTokenizer,
    config: DecisionFitStudyConfig,
) -> DecisionModel:
    return DecisionModel(
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


def _predict_cases(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    cases: Sequence[BenchmarkCase],
) -> tuple[DecisionPrediction, ...]:
    was_training = model.training
    model.eval()
    rows: list[DecisionPrediction] = []
    with torch.no_grad():
        for case in cases:
            scores = score_candidates(
                model,
                tokenizer,
                case.example.state,
                case.example.trajectory,
                case.example.candidates,
            )
            logits = tuple(float(value) for value in scores.detach().cpu().tolist())
            probabilities = probabilities_from_logits(logits)
            predicted_index = max(range(len(logits)), key=lambda index: logits[index])
            rows.append(
                DecisionPrediction(
                    case_id=case.case_id,
                    split=case.split,
                    slice_name=case.slice_name,
                    target_index=case.example.target_index,
                    predicted_index=predicted_index,
                    logits=logits,
                    raw_probabilities=probabilities,
                )
            )
    if was_training:
        model.train()
    return tuple(rows)


def _fit_metrics(predictions: Sequence[DecisionPrediction]) -> dict[str, object]:
    if not predictions:
        raise ValueError("Fit metrics require at least one prediction")
    margins: list[float | None] = []
    nll = 0.0
    correct = 0
    for prediction in predictions:
        alternatives = [
            score
            for index, score in enumerate(prediction.logits)
            if index != prediction.target_index
        ]
        margins.append(
            prediction.logits[prediction.target_index] - max(alternatives)
            if alternatives
            else None
        )
        maximum = max(prediction.logits)
        nll += (
            maximum
            - prediction.logits[prediction.target_index]
            + math.log(sum(math.exp(value - maximum) for value in prediction.logits))
        )
        correct += int(prediction.correct)
    finite_margins = [margin for margin in margins if margin is not None]
    return {
        "examples": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "nll": nll / len(predictions),
        "min_target_margin": min(finite_margins) if finite_margins else None,
        "all_target_margins_positive": (
            len(finite_margins) == len(margins)
            and all(margin > 0.0 for margin in finite_margins)
        ),
    }


def _curve_row(
    *,
    epoch: int,
    metrics: dict[str, object],
    optimizer_updates: int,
    example_exposures: int,
    update_loss_mean: float | None,
) -> dict[str, object]:
    return {
        "schema_version": "aster-decision-fit-curve-0",
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "example_exposures": example_exposures,
        "update_loss_mean": update_loss_mean,
        "update_loss_semantics": (
            None
            if update_loss_mean is None
            else "mean online pre-update training loss; not epoch-end NLL"
        ),
        "metrics": metrics,
    }


def _prediction_rows(
    *,
    epoch: int,
    cases: Sequence[BenchmarkCase],
    predictions: Sequence[DecisionPrediction],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for case, prediction in zip(cases, predictions, strict=True):
        alternatives = [
            score
            for index, score in enumerate(prediction.logits)
            if index != prediction.target_index
        ]
        row = prediction.to_dict()
        row.update(
            epoch=epoch,
            target_action=case.example.target.to_dict(),
            predicted_action=case.example.candidates[prediction.predicted_index].to_dict(),
            candidates=[candidate.to_dict() for candidate in case.example.candidates],
            target_probability=prediction.raw_probabilities[prediction.target_index],
            target_margin=(
                prediction.logits[prediction.target_index] - max(alternatives)
                if alternatives
                else None
            ),
            step=case.example.state.step,
            leakage_group=case.leakage_group,
        )
        result.append(row)
    return result


def _stable_fit(history: Sequence[dict[str, object]], window: int) -> bool:
    measured = [row for row in history if int(row["epoch"]) > 0]
    if len(measured) < window:
        return False
    for row in measured[-window:]:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        if metrics.get("accuracy") != 1.0:
            return False
        if metrics.get("all_target_margins_positive") is not True:
            return False
    return True


def _audit_train_cases(
    cases: Sequence[BenchmarkCase],
    tokenizer: AsterTokenizer,
    context_length: int,
) -> list[dict[str, object]]:
    signatures: dict[str, tuple[int, str]] = {}
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.split != "train":
            raise ValueError("Fit study audit may only inspect train cases")
        serialized = [
            serialize_decision_input(
                case.example.state,
                case.example.trajectory,
                candidate,
            )
            for candidate in case.example.candidates
        ]
        encoded = [
            tokenizer.encode(text, add_bos=True, add_eos=True)
            for text in serialized
        ]
        if any(len(ids) > context_length for ids in encoded):
            raise ValueError(f"Train case exceeds context length: {case.case_id}")
        signature = hashlib.sha256(
            json.dumps(serialized, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        target_key = json.dumps(
            case.example.target.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = signatures.get(signature)
        if previous is not None and previous != (case.example.target_index, target_key):
            raise ValueError(
                "Conflicting supervision for identical serialized decision inputs: "
                f"{case.case_id}"
            )
        signatures[signature] = (case.example.target_index, target_key)
        rows.append(
            {
                "schema_version": "aster-decision-fit-input-audit-0",
                "case_id": case.case_id,
                "split": case.split,
                "leakage_group": case.leakage_group,
                "step": case.example.state.step,
                "target_index": case.example.target_index,
                "target_action": case.example.target.to_dict(),
                "candidate_count": len(case.example.candidates),
                "candidate_actions": [
                    candidate.to_dict() for candidate in case.example.candidates
                ],
                "serialized_sha256": [
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for text in serialized
                ],
                "token_lengths": [len(ids) for ids in encoded],
                "example_signature_sha256": signature,
            }
        )
    return rows


def _serialized_texts(cases: Sequence[BenchmarkCase]) -> list[str]:
    return [
        serialize_decision_input(case.example.state, case.example.trajectory, candidate)
        for case in cases
        for candidate in case.example.candidates
    ]


def _validate_arms(
    arms: Sequence[DecisionFitArm],
    train_cases: Sequence[BenchmarkCase],
) -> None:
    if not arms:
        raise ValueError("Fit study requires at least one arm")
    ids = {case.case_id for case in train_cases}
    arm_ids: set[str] = set()
    for arm in arms:
        if arm.arm_id in arm_ids:
            raise ValueError(f"Duplicate fit arm ID: {arm.arm_id}")
        arm_ids.add(arm.arm_id)
        missing = [case_id for case_id in arm.case_ids if case_id not in ids]
        if missing:
            raise ValueError(
                f"Fit arm {arm.arm_id} references non-train cases: " + ", ".join(missing)
            )


def _select_cases(
    train_cases: Sequence[BenchmarkCase],
    case_ids: Sequence[str],
) -> tuple[BenchmarkCase, ...]:
    by_id = {case.case_id: case for case in train_cases}
    return tuple(by_id[case_id] for case_id in case_ids)


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(str(tensor.dtype).encode("ascii"))
        value.update(b"\0")
        value.update(json.dumps(list(tensor.shape)).encode("ascii"))
        value.update(b"\0")
        value.update(bytes(tensor.view(torch.uint8).flatten().tolist()))
        value.update(b"\0")
    return value.hexdigest()


def _directory_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(path.iterdir(), key=lambda p: p.name):
        if not item.is_file() or item.is_symlink():
            raise ValueError(f"Unexpected tokenizer artifact entry: {item.name}")
        value.update(item.name.encode("utf-8"))
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _max_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
