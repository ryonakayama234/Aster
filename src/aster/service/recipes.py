"""Allowlisted Aster jobs. Sites chooses semantics; Aster constructs commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from aster.service.artifacts import ArtifactCatalog
from aster.service.contracts import JobSpec


@dataclass(frozen=True)
class PreparedJob:
    command: list[str]
    timeout_seconds: int


@dataclass(frozen=True)
class RecipeInfo:
    recipe_id: str
    kind: str
    input_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "kind": self.kind,
            "input_names": list(self.input_names),
            "parameter_names": list(self.parameter_names),
            "description": self.description,
        }


class RecipeRegistry:
    def __init__(self, root: Path, catalog: ArtifactCatalog | None = None, python: str | None = None):
        self.root = root.resolve()
        self.catalog = catalog or ArtifactCatalog(self.root)
        self.python = python or sys.executable
        self._recipes = {
            "tokenizer-bpe-v0": RecipeInfo(
                recipe_id="tokenizer-bpe-v0",
                kind="tokenizer",
                input_names=("training_view",),
                parameter_names=("vocab_size",),
                description="Train and evaluate the fixed byte-level BPE tokenizer recipe.",
            ),
            "tinylm-overfit-v0": RecipeInfo(
                recipe_id="tinylm-overfit-v0",
                kind="pretrain",
                input_names=("training_view", "tokenizer"),
                parameter_names=(),
                description="Run the fixed TinyLM overfit recipe for wiring and memorization observation.",
            ),
        }

    def list(self) -> list[dict[str, object]]:
        return [self._recipes[key].to_dict() for key in sorted(self._recipes)]

    def prepare(self, spec: JobSpec, job_root: Path) -> PreparedJob:
        recipe = self._recipes.get(spec.recipe_id)
        if recipe is None or recipe.kind != spec.kind:
            raise ValueError("Unknown recipe for job kind")
        if set(spec.inputs) != set(recipe.input_names):
            raise ValueError("Recipe inputs do not match the contract")
        if set(spec.parameters) != set(recipe.parameter_names):
            raise ValueError("Recipe parameters do not match the contract")

        view = self.catalog.resolve(spec.inputs["training_view"], "training_view")
        if spec.kind == "tokenizer":
            vocab = spec.parameters["vocab_size"]
            if type(vocab) is not int or vocab not in (256, 512, 1024):
                raise ValueError("vocab_size must be one of 256, 512, 1024")
            return PreparedJob(
                command=[
                    self.python,
                    "-m",
                    "aster.training.tokenizer_run",
                    "--root",
                    str(job_root),
                    "--view",
                    str(view),
                    "--vocab-size",
                    str(vocab),
                ],
                timeout_seconds=1800,
            )

        tokenizer = self.catalog.resolve(spec.inputs["tokenizer"], "tokenizer")
        if self.catalog.tokenizer_view_id(spec.inputs["tokenizer"]) != view.name:
            raise ValueError("Tokenizer and training view do not match")
        config = self.root / "configs" / "tinylm-overfit-v0.json"
        if not config.is_file() or config.is_symlink():
            raise ValueError("Pretrain recipe config is missing")
        return PreparedJob(
            command=[
                self.python,
                "-m",
                "aster.training.pretrain",
                "--root",
                str(job_root),
                "--view",
                str(view),
                "--tokenizer",
                str(tokenizer),
                "--config",
                str(config),
            ],
            timeout_seconds=1800,
        )
