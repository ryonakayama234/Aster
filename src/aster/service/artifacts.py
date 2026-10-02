"""Resolve logical artifact IDs without exposing repository-local paths to Sites."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile

from aster.service.contracts import ARTIFACT_KINDS, ArtifactRef

_DIGEST = re.compile(r"[0-9a-f]{64}")
_DECISION_MODEL_KIND = "decision_model"
_DECISION_CANDIDATE_BUILDER = "calculate-and-store-v0"
_DECISION_REQUIRED = (
    "manifest.json",
    "model.pt",
    "tokenizer/manifest.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
    "tokenizer/special_tokens.json",
)


class ArtifactCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def make_id(self, kind: str, digest: str) -> str:
        if kind not in ARTIFACT_KINDS or not _DIGEST.fullmatch(digest):
            raise ValueError("Invalid artifact identity")
        return f"{kind}:{digest}"

    def parse(self, artifact_id: str) -> ArtifactRef:
        if not isinstance(artifact_id, str) or ":" not in artifact_id:
            raise ValueError("Invalid artifact ID")
        kind, digest = artifact_id.split(":", 1)
        canonical = self.make_id(kind, digest)
        if artifact_id != canonical:
            raise ValueError("Invalid artifact ID")
        return ArtifactRef(artifact_id=artifact_id, kind=kind, digest=digest)

    def resolve(self, artifact_id: str, expected_kind: str) -> Path:
        ref = self.parse(artifact_id)
        if ref.kind != expected_kind:
            raise ValueError(f"Expected {expected_kind} artifact")
        if ref.kind == "training_view":
            path = self.root / "data" / "training" / ref.digest
            required = ("manifest.json",)
        elif ref.kind == "tokenizer":
            path = self.root / "artifacts" / "tokenizers" / ref.digest
            required = (
                "manifest.json",
                "evaluation.json",
                "vocab.json",
                "merges.json",
                "special_tokens.json",
            )
        elif ref.kind == _DECISION_MODEL_KIND:
            path = self.root / "artifacts" / "decision_models" / ref.digest
            required = _DECISION_REQUIRED
        else:
            raise ValueError("Unsupported artifact kind")
        if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact does not exist")
        if any(not (path / name).is_file() or (path / name).is_symlink() for name in required):
            raise ValueError("Artifact is incomplete")
        if ref.kind == _DECISION_MODEL_KIND:
            self._validate_decision_manifest_identity(path, ref)
        return path

    def register_decision_model(self, source: str | Path) -> ArtifactRef:
        """Validate and immutably register one inference-ready DecisionModel artifact."""
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_dir():
            raise ValueError("Decision artifact source must be a real directory")
        if any(item.is_symlink() for item in source_path.rglob("*")):
            raise ValueError("Decision artifact source must not contain symlinks")

        from aster.model.decision_artifact import load_decision_artifact

        _, _, manifest = load_decision_artifact(source_path)
        artifact_id = manifest.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise ValueError("Decision artifact has no valid artifact_id")
        ref = self.parse(artifact_id)
        if ref.kind != _DECISION_MODEL_KIND:
            raise ValueError("Decision artifact identity has wrong kind")
        if manifest.get("candidate_builder_id") != _DECISION_CANDIDATE_BUILDER:
            raise ValueError("Unsupported Decision artifact candidate builder")

        destination = self.root / "artifacts" / "decision_models" / ref.digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            registered = self.resolve(ref.artifact_id, _DECISION_MODEL_KIND)
            _, _, registered_manifest = load_decision_artifact(registered)
            if registered_manifest.get("artifact_id") != ref.artifact_id:
                raise ValueError("Registered Decision artifact identity mismatch")
            return ref

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{ref.digest}.", dir=destination.parent)
        )
        try:
            shutil.rmtree(temporary)
            shutil.copytree(source_path, temporary)
            if any(item.is_symlink() for item in temporary.rglob("*")):
                raise ValueError("Registered Decision artifact must not contain symlinks")
            _, _, copied_manifest = load_decision_artifact(temporary)
            if copied_manifest.get("artifact_id") != ref.artifact_id:
                raise ValueError("Copied Decision artifact identity mismatch")
            if copied_manifest.get("candidate_builder_id") != _DECISION_CANDIDATE_BUILDER:
                raise ValueError("Unsupported copied Decision artifact candidate builder")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return ref

    def training_views(self) -> list[dict[str, object]]:
        base = self.root / "data" / "training"
        return [
            self._describe_training_view(path)
            for path in sorted(base.glob("*"))
            if self._candidate(path, ("manifest.json",))
        ]

    def tokenizers(self) -> list[dict[str, object]]:
        base = self.root / "artifacts" / "tokenizers"
        required = (
            "manifest.json",
            "evaluation.json",
            "vocab.json",
            "merges.json",
            "special_tokens.json",
        )
        return [
            self._describe_tokenizer(path)
            for path in sorted(base.glob("*"))
            if self._candidate(path, required)
        ]

    def decision_models(self) -> list[dict[str, object]]:
        base = self.root / "artifacts" / "decision_models"
        return [
            self._describe_decision_model(path)
            for path in sorted(base.glob("*"))
            if self._candidate(path, _DECISION_REQUIRED)
        ]

    def all(self) -> dict[str, list[dict[str, object]]]:
        return {
            "training_views": self.training_views(),
            "tokenizers": self.tokenizers(),
            "decision_models": self.decision_models(),
        }

    def tokenizer_view_id(self, artifact_id: str) -> str:
        path = self.resolve(artifact_id, "tokenizer")
        try:
            evaluation = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
            view_id = evaluation["experiment"]["view_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("Tokenizer provenance is invalid") from error
        if not isinstance(view_id, str) or not _DIGEST.fullmatch(view_id):
            raise ValueError("Tokenizer provenance is invalid")
        return view_id

    def _candidate(self, path: Path, required: tuple[str, ...]) -> bool:
        return (
            bool(_DIGEST.fullmatch(path.name))
            and path.is_dir()
            and not path.is_symlink()
            and path.resolve().is_relative_to(self.root)
            and all((path / name).is_file() and not (path / name).is_symlink() for name in required)
        )

    def _describe_training_view(self, path: Path) -> dict[str, object]:
        ref = ArtifactRef(self.make_id("training_view", path.name), "training_view", path.name)
        return ref.to_dict()

    def _describe_tokenizer(self, path: Path) -> dict[str, object]:
        ref = ArtifactRef(self.make_id("tokenizer", path.name), "tokenizer", path.name)
        data: dict[str, object] = ref.to_dict()
        try:
            data["training_view_digest"] = self.tokenizer_view_id(ref.artifact_id)
        except ValueError:
            data["provenance_valid"] = False
        else:
            data["provenance_valid"] = True
        return data

    def _describe_decision_model(self, path: Path) -> dict[str, object]:
        ref = ArtifactRef(self.make_id(_DECISION_MODEL_KIND, path.name), _DECISION_MODEL_KIND, path.name)
        data: dict[str, object] = ref.to_dict()
        try:
            manifest = self._read_decision_manifest(path)
            self._validate_decision_manifest_identity(path, ref)
            data.update(
                model_id=manifest.get("model_id"),
                candidate_builder_id=manifest.get("candidate_builder_id"),
                suite_id=manifest.get("suite_id"),
                source_git_sha=manifest.get("source_git_sha"),
                provenance_valid=True,
            )
        except ValueError:
            data["provenance_valid"] = False
        return data

    def _read_decision_manifest(self, path: Path) -> dict[str, object]:
        try:
            value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Decision artifact manifest is invalid") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("Decision artifact manifest is invalid")
        return dict(value)

    def _validate_decision_manifest_identity(self, path: Path, ref: ArtifactRef) -> None:
        manifest = self._read_decision_manifest(path)
        if manifest.get("artifact_id") != ref.artifact_id:
            raise ValueError("Decision artifact manifest identity mismatch")
        if manifest.get("candidate_builder_id") != _DECISION_CANDIDATE_BUILDER:
            raise ValueError("Unsupported Decision artifact candidate builder")
