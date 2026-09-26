"""Persistent local jobs for the Aster Sites control plane."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any
from uuid import uuid4

from aster.service.artifacts import ArtifactCatalog
from aster.service.contracts import JobSpec
from aster.service.recipes import RecipeRegistry

_JOB_ID = re.compile(r"[0-9a-f]{32}")
_ACTIVE = frozenset({"accepted", "running"})


class JobBusyError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class JobManager:
    def __init__(
        self,
        root: Path,
        *,
        python: str | None = None,
        process_runner: Callable[..., Any] | None = None,
    ):
        self.root = root.resolve()
        self.base = self.root / "runs" / "workbench"
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.catalog = ArtifactCatalog(self.root)
        self.recipes = RecipeRegistry(self.root, self.catalog, python or sys.executable)
        self.process_runner = process_runner or subprocess.run
        self._recover_interrupted()

    def legacy_tokenizer_spec(self, view_digest: str, vocab_size: int) -> JobSpec:
        return JobSpec(
            kind="tokenizer",
            recipe_id="tokenizer-bpe-v0",
            inputs={"training_view": self.catalog.make_id("training_view", view_digest)},
            parameters={"vocab_size": vocab_size},
        )

    def start(self, spec: JobSpec) -> str:
        job_id = uuid4().hex
        folder = self.base / job_id
        prepared = self.recipes.prepare(spec, folder)
        with self.lock:
            if self._has_active_job():
                raise JobBusyError("Aster compute job is already running")
            folder.mkdir()
            job: dict[str, object] = {
                "schema_version": "aster-service-job-0",
                "job_id": job_id,
                "run_id": None,
                "status": "accepted",
                "kind": spec.kind,
                "recipe_id": spec.recipe_id,
                "spec": spec.to_dict(),
                "accepted_at": now(),
            }
            write_json(folder / "job.json", job)
        thread = threading.Thread(
            target=self._work,
            args=(folder, job, prepared.command, prepared.timeout_seconds),
            daemon=True,
        )
        thread.start()
        return job_id

    def read(self, job_id: str) -> dict[str, object]:
        folder = self._folder(job_id)
        job = self._read_job(folder / "job.json")
        run, events, outputs = self._read_run(folder)
        visible_job = dict(job)
        if run is not None and visible_job.get("run_id") is None:
            visible_job["run_id"] = run.get("run_id")
        bundle: dict[str, object] = {
            "schema_version": "aster-service-job-bundle-0",
            "job": visible_job,
            "run": run,
            "events": events,
            "outputs": outputs,
        }
        if visible_job.get("kind") == "tokenizer":
            bundle["tokenizer"] = outputs.get("tokenizer")
        if visible_job.get("kind") == "pretrain":
            bundle["training"] = outputs.get("training")
        return bundle

    def summaries(self) -> list[dict[str, object]]:
        jobs = []
        for path in self.base.glob("*/job.json"):
            try:
                jobs.append(self._read_job(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        jobs.sort(key=lambda item: str(item.get("accepted_at", "")), reverse=True)
        return jobs[:50]

    def status(self) -> dict[str, object]:
        artifacts = self.catalog.all()
        return {
            "schema_version": "aster-service-status-0",
            "version": "aster-service-0.1",
            "capabilities": ["tokenizer.train", "model.pretrain"],
            "recipes": self.recipes.list(),
            "artifacts": artifacts,
            "jobs": self.summaries(),
            "views": [item["digest"] for item in artifacts["training_views"]],
        }

    def _work(self, folder: Path, job: dict[str, object], command: list[str], timeout: int) -> None:
        job["status"] = "running"
        job["started_at"] = now()
        write_json(folder / "job.json", job)
        environment = dict(os.environ, PYTHONPATH=str(self.root / "src"))
        try:
            with (folder / "process.log").open("w", encoding="utf-8") as log:
                result = self.process_runner(
                    command,
                    cwd=self.root,
                    env=environment,
                    stdout=log,
                    stderr=log,
                    timeout=timeout,
                )
            run, _, _ = self._read_run(folder)
            if run is None:
                raise RuntimeError("Aster run record was not created")
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not _JOB_ID.fullmatch(run_id):
                raise RuntimeError("Aster run ID is invalid")
            job["run_id"] = run_id
            returncode = getattr(result, "returncode", None)
            if returncode != 0 or run.get("status") != "completed":
                tail = (folder / "process.log").read_text(encoding="utf-8")[-3000:]
                raise RuntimeError(tail or "Aster process did not complete")
            if job.get("kind") == "tokenizer":
                self._promote_tokenizer(folder, run)
            job["status"] = "completed"
        except subprocess.TimeoutExpired:
            job.update(status="failed", error="Aster process timed out")
        except Exception as error:
            job.update(status="failed", error=str(error))
        job["ended_at"] = now()
        write_json(folder / "job.json", job)
        write_json(folder / "run-bundle.json", self.read(str(job["job_id"])))

    def _promote_tokenizer(self, folder: Path, run: dict[str, object]) -> None:
        raw_output = run.get("output")
        if not isinstance(raw_output, str):
            raise RuntimeError("Tokenizer run did not record its output artifact")
        source = Path(raw_output).resolve()
        source_base = (folder / "artifacts" / "tokenizers").resolve()
        if not source.is_relative_to(source_base) or not source.is_dir() or source.is_symlink():
            raise RuntimeError("Tokenizer output escaped the job artifact directory")
        if not re.fullmatch(r"[0-9a-f]{64}", source.name):
            raise RuntimeError("Tokenizer output identity is invalid")
        destination_base = self.root / "artifacts" / "tokenizers"
        destination_base.mkdir(parents=True, exist_ok=True)
        destination = destination_base / source.name
        if destination.exists():
            if not self._same_tree(source, destination):
                raise RuntimeError("Existing tokenizer artifact differs from completed job")
            return
        with tempfile.TemporaryDirectory(dir=destination_base, prefix=".promoting-") as temporary:
            stage = Path(temporary) / source.name
            shutil.copytree(source, stage)
            os.rename(stage, destination)

    @staticmethod
    def _same_tree(first: Path, second: Path) -> bool:
        first_files = {p.relative_to(first) for p in first.rglob("*") if p.is_file()}
        second_files = {p.relative_to(second) for p in second.rglob("*") if p.is_file()}
        return first_files == second_files and all(
            (first / name).read_bytes() == (second / name).read_bytes() for name in first_files
        )

    def _read_run(self, folder: Path) -> tuple[dict[str, object] | None, list[object], dict[str, object]]:
        paths = sorted((folder / "runs").glob("*/run.json"))
        if len(paths) > 1:
            raise ValueError("Job produced more than one Aster run")
        if not paths:
            return None, [], {}
        run_path = paths[0]
        run = self._read_object(run_path)
        events: list[object] = []
        events_path = run_path.parent / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        outputs: dict[str, object] = {}
        for name, key in (("tokenizer-bundle.json", "tokenizer"), ("training-bundle.json", "training")):
            path = run_path.parent / name
            if path.exists():
                try:
                    outputs[key] = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
        return run, events, outputs

    def _recover_interrupted(self) -> None:
        for path in self.base.glob("*/job.json"):
            try:
                job = self._read_job(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if job.get("status") in _ACTIVE:
                job.update(
                    status="interrupted",
                    ended_at=now(),
                    error="Service restarted before the job reached a terminal state",
                )
                write_json(path, job)

    def _has_active_job(self) -> bool:
        for path in self.base.glob("*/job.json"):
            try:
                if self._read_job(path).get("status") in _ACTIVE:
                    return True
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return False

    def _folder(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ValueError("Invalid job ID")
        folder = self.base / job_id
        if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(self.base.resolve()):
            raise ValueError("Job does not exist")
        return folder

    def _read_job(self, path: Path) -> dict[str, object]:
        data = self._read_object(path)
        if data.get("schema_version") != "aster-service-job-0":
            raise ValueError("Unsupported job record")
        return data

    @staticmethod
    def _read_object(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} must contain a JSON object")
        return dict(value)
