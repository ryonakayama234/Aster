"""Read-only projections of committed research evidence for the local workbench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_SCHEMA = "aster-experiment-bundle-0"
EXPERIMENT_INDEX_SCHEMA = "aster-experiment-index-0"
FAILURE_AUDIT_ID = "decision-failure-audit-v0"
_FAILURE_AUDIT_SCHEMA = "aster-decision-failure-audit-study-0"


class ExperimentCatalog:
    """Expose a small allowlist of committed research reports by logical ID."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._sources = {
            FAILURE_AUDIT_ID: self.root / "reports" / "decision-failure-audit-v0.json",
        }

    def list(self) -> list[dict[str, object]]:
        bundle = self.read(FAILURE_AUDIT_ID)
        arms = bundle["arms"]
        if not isinstance(arms, list):
            raise ValueError("Experiment projection is invalid")
        return [
            {
                "experiment_id": FAILURE_AUDIT_ID,
                "title": "Decision failure audit v0",
                "schema_version": EXPERIMENT_SCHEMA,
                "arm_count": len(arms),
                "case_scope": "phase-zero shared tokenization cases",
                "learning_curve_status": "unavailable_from_committed_evidence",
            }
        ]

    def read(self, experiment_id: str) -> dict[str, object]:
        if experiment_id != FAILURE_AUDIT_ID:
            raise ValueError("Unknown experiment ID")
        path = self._sources[experiment_id]
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Experiment evidence is unavailable")
        raw = path.read_bytes()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Experiment evidence is invalid") from error
        data = _object(value, "experiment evidence")
        if data.get("schema_version") != _FAILURE_AUDIT_SCHEMA:
            raise ValueError("Unsupported experiment evidence schema")

        examples = _object_list(data.get("shared_tokenization_examples"), "shared_tokenization_examples")
        examples_by_id: dict[str, dict[str, object]] = {}
        for example in examples:
            case_id = _string(example.get("case_id"), "shared_tokenization_examples.case_id")
            if case_id in examples_by_id:
                raise ValueError("Duplicate experiment case ID")
            examples_by_id[case_id] = example

        arms: list[dict[str, object]] = []
        for run in _object_list(data.get("runs"), "runs"):
            arms.append(self._project_arm(run, examples_by_id))

        return {
            "schema_version": EXPERIMENT_SCHEMA,
            "experiment_id": experiment_id,
            "title": "Decision failure audit v0",
            "evidence": {
                "source_name": path.name,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_schema_version": _FAILURE_AUDIT_SCHEMA,
            },
            "scope": (
                "Committed failure-audit evidence only. Cases are phase-zero decision probes; "
                "this bundle does not claim independent task-family generalization."
            ),
            "limitations": [
                "learning curves are not present in the committed failure-audit report",
                "candidate logits/scores are not present in the committed failure-audit report",
                "tokenization describes the target candidate serialization, not every candidate",
                "test remains sealed",
            ],
            "comparison": {
                "case_join": "case_id",
                "token_id_cross_tokenizer_alignment": False,
                "ui_recomputes_metrics": False,
            },
            "arms": arms,
        }

    def _project_arm(
        self,
        run: dict[str, object],
        examples_by_id: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        seed = _integer(run.get("seed"), "runs.seed")
        training_arm = _string(run.get("arm"), "runs.arm")
        phase_zero = _object_list(run.get("phase_zero"), "runs.phase_zero")
        cases: list[dict[str, object]] = []
        for prediction in phase_zero:
            case_id = _string(prediction.get("case_id"), "runs.phase_zero.case_id")
            example = examples_by_id.get(case_id)
            if example is None:
                raise ValueError("Experiment prediction has no shared tokenization evidence")
            cases.append(
                {
                    "case_id": case_id,
                    "slice": _string(prediction.get("slice"), "runs.phase_zero.slice"),
                    "target_action": _json_value(prediction.get("target_action"), "target_action"),
                    "selected_action": _json_value(prediction.get("model_action"), "model_action"),
                    "correct": _boolean(prediction.get("model_correct"), "model_correct"),
                    "target_candidate_serialization": _string(
                        example.get("target_input"), "shared_tokenization_examples.target_input"
                    ),
                    "target_tokenization": {
                        "token_ids": _integer_list(
                            example.get("target_token_ids"), "shared_tokenization_examples.target_token_ids"
                        ),
                        "pieces": _string_list(
                            example.get("target_token_pieces"), "shared_tokenization_examples.target_token_pieces"
                        ),
                        "length": _integer(
                            example.get("target_token_length"), "shared_tokenization_examples.target_token_length"
                        ),
                        "unseen_in_anchor_token_occurrences": _integer(
                            example.get("unseen_in_anchor_token_occurrences"),
                            "shared_tokenization_examples.unseen_in_anchor_token_occurrences",
                        ),
                    },
                    "candidate_scores": None,
                    "candidate_scores_status": "unavailable_from_committed_evidence",
                }
            )

        return {
            "arm_id": f"seed-{seed}--{training_arm}",
            "seed": seed,
            "training_arm": training_arm,
            "checkpoint_id": _string(run.get("checkpoint_id"), "runs.checkpoint_id"),
            "source_run_id": _string(run.get("source_run_id"), "runs.source_run_id"),
            "audit_run_id": _string(run.get("audit_run_id"), "runs.audit_run_id"),
            "suite_sha256": _string(run.get("suite_sha256"), "runs.suite_sha256"),
            "tokenizer_sha256": _string(run.get("tokenizer_sha256"), "runs.tokenizer_sha256"),
            "model_update": _boolean(run.get("model_update"), "runs.model_update"),
            "new_inference": _boolean(run.get("new_inference"), "runs.new_inference"),
            "test_status": _string(run.get("test"), "runs.test"),
            "summary": {
                "correct_by_slice": _json_value(run.get("correct_by_slice"), "runs.correct_by_slice"),
                "error_confusion": _json_value(run.get("error_confusion"), "runs.error_confusion"),
                "interpretation": _string(run.get("interpretation"), "runs.interpretation"),
            },
            "learning_curve": None,
            "learning_curve_status": "unavailable_from_committed_evidence",
            "cases": cases,
        }


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _object_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [_object(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise ValueError(f"{label} must be an integer array")
    return list(value)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    return list(value)


def _json_value(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, f"{label}.{key}") for key, item in value.items()}
    raise ValueError(f"{label} is not JSON-compatible")
