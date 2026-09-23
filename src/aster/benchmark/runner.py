"""Run Decision-v0 benchmarks and emit Sites-readable versioned artifacts."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from aster.benchmark.case import BenchmarkSuite
from aster.benchmark.metrics import (
    DecisionPrediction,
    apply_temperature,
    fit_temperature,
    probabilities_from_logits,
    summarize_predictions,
)
from aster.inference.decide import score_candidates
from aster.model.decision_head import DecisionModel
from aster.records.runlog import RunLog
from aster.tokenizer.artifact import AsterTokenizer


@dataclass(frozen=True, slots=True)
class BenchmarkBundle:
    suite_id: str
    model_id: str
    temperature: float
    predictions: tuple[DecisionPrediction, ...]
    test_raw: dict
    test_calibrated: dict
    by_split: dict[str, dict]
    by_slice: dict[str, dict]

    def to_dict(self, *, include_predictions: bool = True) -> dict:
        payload = {
            "schema_version": "aster-decision-benchmark-0",
            "suite_id": self.suite_id,
            "model_id": self.model_id,
            "temperature_scaling": {
                "fit_split": "calibration",
                "temperature": self.temperature,
            },
            "test": {"raw": self.test_raw, "calibrated": self.test_calibrated},
            "by_split": self.by_split,
            "by_slice": self.by_slice,
        }
        if include_predictions:
            payload["predictions"] = [prediction.to_dict() for prediction in self.predictions]
        return payload


def run_decision_benchmark(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    suite: BenchmarkSuite,
    *,
    model_id: str = "AsterDecision-v0",
) -> BenchmarkBundle:
    """Evaluate one model without updating it; fit temperature on calibration only."""
    calibration_cases = suite.cases_for("calibration")
    test_cases = suite.cases_for("test")
    if not calibration_cases:
        raise ValueError("Benchmark suite requires a calibration split")
    if not test_cases:
        raise ValueError("Benchmark suite requires a test split")

    was_training = model.training
    model.eval()
    raw_predictions: list[DecisionPrediction] = []
    try:
        with torch.no_grad():
            for case in suite.cases:
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
                raw_predictions.append(
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

    calibration_predictions = tuple(
        prediction
        for prediction in raw_predictions
        if prediction.split == "calibration"
    )
    temperature = fit_temperature(calibration_predictions)
    predictions = apply_temperature(raw_predictions, temperature)

    by_split: dict[str, dict] = {}
    for split in ("train", "calibration", "dev", "test"):
        group = tuple(prediction for prediction in predictions if prediction.split == split)
        if group:
            by_split[split] = _metric_pair(group)

    by_slice: dict[str, dict] = {}
    for slice_name in sorted({prediction.slice_name for prediction in predictions}):
        group = tuple(
            prediction for prediction in predictions if prediction.slice_name == slice_name
        )
        by_slice[slice_name] = _metric_pair(group)

    test_predictions = tuple(
        prediction for prediction in predictions if prediction.split == "test"
    )
    return BenchmarkBundle(
        suite_id=suite.suite_id,
        model_id=model_id,
        temperature=temperature,
        predictions=predictions,
        test_raw=summarize_predictions(test_predictions, calibrated=False),
        test_calibrated=summarize_predictions(test_predictions, calibrated=True),
        by_split=by_split,
        by_slice=by_slice,
    )


def write_benchmark_artifacts(output_dir: str | Path, bundle: BenchmarkBundle) -> None:
    """Write compact summary plus per-case predictions for later Sites replay."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = bundle.to_dict(include_predictions=False)
    summary.update(
        {
            "predictions_file": "predictions.jsonl",
            "calibration_file": "calibration.json",
        }
    )
    _write_json(output / "benchmark.json", summary)

    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in bundle.predictions:
            stream.write(json.dumps(prediction.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    calibration = {
        "schema_version": "aster-decision-calibration-0",
        "suite_id": bundle.suite_id,
        "model_id": bundle.model_id,
        "fit_split": "calibration",
        "temperature": bundle.temperature,
        "metrics": bundle.by_split["calibration"],
    }
    _write_json(output / "calibration.json", calibration)


def run_logged_decision_benchmark(
    root: str | Path,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    suite: BenchmarkSuite,
    *,
    model_id: str = "AsterDecision-v0",
) -> Path:
    """Run a benchmark as an observable evaluator run for Sites and later comparison."""
    root = Path(root)
    run = RunLog(
        root,
        "decision_benchmark",
        {"suite_id": suite.suite_id, "model_id": model_id},
        producer="evaluator",
    )
    try:
        bundle = run_decision_benchmark(model, tokenizer, suite, model_id=model_id)
        write_benchmark_artifacts(run.path, bundle)
        run.event(
            "evaluated",
            {
                "suite_id": bundle.suite_id,
                "model_id": bundle.model_id,
                "temperature": bundle.temperature,
                "test": {"raw": bundle.test_raw, "calibrated": bundle.test_calibrated},
            },
        )
        run.finish(
            "completed",
            benchmark="benchmark.json",
            predictions="predictions.jsonl",
            calibration="calibration.json",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _metric_pair(predictions: tuple[DecisionPrediction, ...]) -> dict:
    return {
        "raw": summarize_predictions(predictions, calibrated=False),
        "calibrated": summarize_predictions(predictions, calibrated=True),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
