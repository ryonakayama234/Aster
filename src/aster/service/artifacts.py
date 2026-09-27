"""Resolve logical artifact IDs without exposing repository-local paths to Sites."""

from __future__ import annotations

import json
from pathlib import Path
import re

from aster.service.contracts import ARTIFACT_KINDS, ArtifactRef

_DIGEST = re.compile(r"[0-9a-f]{64}")


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
            required = ("manifest.json", "evaluation.json", "vocab.json", "merges.json", "special_tokens.json")
        else:
            raise ValueError("Unsupported artifact kind")
        if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact does not exist")
        if any(not (path / name).is_file() or (path / name).is_symlink() for name in required):
            raise ValueError("Artifact is incomplete")
        return path

    def training_views(self) -> list[dict[str, object]]:
        base = self.root / "data" / "training"
        return [
            self._describe_training_view(path)
            for path in sorted(base.glob("*"))
            if self._candidate(path, ("manifest.json",))
        ]

    def tokenizers(self) -> list[dict[str, object]]:
        base = self.root / "artifacts" / "tokenizers"
        required = ("manifest.json", "evaluation.json", "vocab.json", "merges.json", "special_tokens.json")
        return [
            self._describe_tokenizer(path)
            for path in sorted(base.glob("*"))
            if self._candidate(path, required)
        ]

    def all(self) -> dict[str, list[dict[str, object]]]:
        return {"training_views": self.training_views(), "tokenizers": self.tokenizers()}

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
