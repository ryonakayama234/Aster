"""Versioned JSON contracts shared by Sites and the local Aster service."""

from __future__ import annotations

from dataclasses import dataclass
import math

SCHEMA_JOB_SPEC = "aster-job-spec-0"
ARTIFACT_KINDS = frozenset({"training_view", "tokenizer"})
JOB_KINDS = frozenset({"tokenizer", "pretrain"})


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, "kind": self.kind, "digest": self.digest}


@dataclass(frozen=True)
class JobSpec:
    kind: str
    recipe_id: str
    inputs: dict[str, str]
    parameters: dict[str, object]
    schema_version: str = SCHEMA_JOB_SPEC

    @classmethod
    def from_dict(cls, value: object) -> "JobSpec":
        data = _object(value, "job spec")
        allowed = {"schema_version", "kind", "recipe_id", "inputs", "parameters"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("Unknown job fields: " + ", ".join(sorted(unknown)))
        if data.get("schema_version") != SCHEMA_JOB_SPEC:
            raise ValueError("Unsupported job schema")
        kind = _string(data.get("kind"), "kind")
        if kind not in JOB_KINDS:
            raise ValueError("Unsupported job kind")
        recipe_id = _string(data.get("recipe_id"), "recipe_id")
        inputs_data = _object(data.get("inputs"), "inputs")
        inputs = {key: _string(item, f"inputs.{key}") for key, item in inputs_data.items()}
        parameters_data = data.get("parameters", {})
        parameters = _object(parameters_data, "parameters")
        _require_json(parameters, "parameters")
        return cls(kind=kind, recipe_id=recipe_id, inputs=inputs, parameters=parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "recipe_id": self.recipe_id,
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
        }


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_json(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _require_json(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} must contain JSON values only")
