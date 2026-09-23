"""Probability-quality metrics for variable-size decision candidate sets."""

from dataclasses import dataclass, replace
import math
from typing import Sequence

_DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True, slots=True)
class DecisionPrediction:
    case_id: str
    split: str
    slice_name: str
    target_index: int
    predicted_index: int
    logits: tuple[float, ...]
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...] | None = None

    def __post_init__(self):
        size = len(self.logits)
        if not size or len(self.raw_probabilities) != size:
            raise ValueError("Prediction logits and probabilities must have matching lengths")
        if self.calibrated_probabilities is not None and len(self.calibrated_probabilities) != size:
            raise ValueError("Calibrated probabilities must match candidate count")
        if not 0 <= self.target_index < size or not 0 <= self.predicted_index < size:
            raise ValueError("Prediction indices must reference one candidate")

    @property
    def correct(self) -> bool:
        return self.predicted_index == self.target_index

    def probabilities(self, *, calibrated: bool) -> tuple[float, ...]:
        if calibrated:
            if self.calibrated_probabilities is None:
                raise ValueError("Prediction has not been temperature-calibrated")
            return self.calibrated_probabilities
        return self.raw_probabilities

    def to_dict(self) -> dict:
        raw_confidence = max(self.raw_probabilities)
        calibrated_confidence = (
            None
            if self.calibrated_probabilities is None
            else max(self.calibrated_probabilities)
        )
        return {
            "schema_version": "aster-decision-benchmark-prediction-0",
            "case_id": self.case_id,
            "split": self.split,
            "slice": self.slice_name,
            "target_index": self.target_index,
            "predicted_index": self.predicted_index,
            "correct": self.correct,
            "logits": list(self.logits),
            "raw_probabilities": list(self.raw_probabilities),
            "calibrated_probabilities": (
                None
                if self.calibrated_probabilities is None
                else list(self.calibrated_probabilities)
            ),
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
        }


def probabilities_from_logits(
    logits: Sequence[float], temperature: float = 1.0
) -> tuple[float, ...]:
    if not logits:
        raise ValueError("At least one logit is required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def fit_temperature(predictions: Sequence[DecisionPrediction]) -> float:
    """Fit one positive temperature on a held-out calibration split.

    A deterministic log-space search is used instead of another trainable model. The
    search always includes T=1, so calibration cannot worsen its own fitted NLL merely
    because the identity transform was omitted from the candidate temperatures.
    """
    if not predictions:
        raise ValueError("Temperature fitting requires calibration predictions")

    log_lo, log_hi = math.log(0.05), math.log(20.0)
    best_log = 0.0
    best_loss = _temperature_nll(predictions, 1.0)

    for _ in range(4):
        points = [log_lo + (log_hi - log_lo) * i / 80 for i in range(81)]
        if log_lo <= 0.0 <= log_hi:
            points.append(0.0)
        scored = [(_temperature_nll(predictions, math.exp(point)), point) for point in points]
        loss, candidate = min(scored, key=lambda item: (item[0], abs(item[1])))
        if loss <= best_loss:
            best_loss, best_log = loss, candidate
        ordered = sorted(set(points))
        index = min(range(len(ordered)), key=lambda i: abs(ordered[i] - best_log))
        log_lo = ordered[max(0, index - 1)]
        log_hi = ordered[min(len(ordered) - 1, index + 1)]
        if log_lo == log_hi:
            break

    return math.exp(best_log)


def apply_temperature(
    predictions: Sequence[DecisionPrediction], temperature: float
) -> tuple[DecisionPrediction, ...]:
    return tuple(
        replace(
            prediction,
            calibrated_probabilities=probabilities_from_logits(
                prediction.logits, temperature
            ),
        )
        for prediction in predictions
    )


def summarize_predictions(
    predictions: Sequence[DecisionPrediction],
    *,
    calibrated: bool,
    ece_bins: int = 10,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
) -> dict:
    if not predictions:
        raise ValueError("At least one prediction is required")
    if type(ece_bins) is not int or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer")

    count = len(predictions)
    nll = 0.0
    brier = 0.0
    correct = 0
    confidence_rows: list[tuple[float, bool]] = []

    for prediction in predictions:
        probabilities = prediction.probabilities(calibrated=calibrated)
        target_probability = max(probabilities[prediction.target_index], 1e-15)
        nll -= math.log(target_probability)
        brier += sum(
            (probability - (1.0 if index == prediction.target_index else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        correct += int(prediction.correct)
        confidence_rows.append((max(probabilities), prediction.correct))

    reliability = _reliability_bins(confidence_rows, ece_bins)
    ece = sum(
        row["count"] / count * abs(row["accuracy"] - row["mean_confidence"])
        for row in reliability
        if row["count"]
    )

    return {
        "examples": count,
        "accuracy": correct / count,
        "nll": nll / count,
        "brier": brier / count,
        "ece": ece,
        "ece_bins": ece_bins,
        "reliability": reliability,
        "risk_coverage": _risk_coverage(confidence_rows, thresholds),
    }


def _temperature_nll(predictions: Sequence[DecisionPrediction], temperature: float) -> float:
    total = 0.0
    for prediction in predictions:
        probabilities = probabilities_from_logits(prediction.logits, temperature)
        total -= math.log(max(probabilities[prediction.target_index], 1e-15))
    return total / len(predictions)


def _reliability_bins(rows: Sequence[tuple[float, bool]], bins: int) -> list[dict]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in rows:
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, correct))

    result: list[dict] = []
    for index, bucket in enumerate(buckets):
        size = len(bucket)
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": size,
                "mean_confidence": (
                    sum(confidence for confidence, _ in bucket) / size if size else 0.0
                ),
                "accuracy": (
                    sum(int(correct) for _, correct in bucket) / size if size else 0.0
                ),
            }
        )
    return result


def _risk_coverage(
    rows: Sequence[tuple[float, bool]], thresholds: Sequence[float]
) -> list[dict]:
    result = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("coverage thresholds must be finite values in [0, 1]")
        accepted = [correct for confidence, correct in rows if confidence >= threshold]
        coverage = len(accepted) / len(rows)
        accuracy = None if not accepted else sum(int(value) for value in accepted) / len(accepted)
        result.append(
            {
                "threshold": float(threshold),
                "accepted": len(accepted),
                "coverage": coverage,
                "accuracy": accuracy,
                "risk": None if accuracy is None else 1.0 - accuracy,
            }
        )
    return result
