import json
from pathlib import Path
import subprocess
import time

import pytest

from aster.service.artifacts import ArtifactCatalog
from aster.service.contracts import JobSpec
from aster.service.jobs import JobManager
from aster.service.recipes import RecipeRegistry


def _artifacts(root: Path) -> tuple[str, str, str]:
    view_digest = "a" * 64
    view = root / "data" / "training" / view_digest
    view.mkdir(parents=True)
    (view / "manifest.json").write_text("{}", encoding="utf-8")

    tokenizer_digest = "b" * 64
    tokenizer = root / "artifacts" / "tokenizers" / tokenizer_digest
    tokenizer.mkdir(parents=True)
    for name in ("manifest.json", "vocab.json", "merges.json", "special_tokens.json"):
        (tokenizer / name).write_text("{}", encoding="utf-8")
    (tokenizer / "evaluation.json").write_text(
        json.dumps({"experiment": {"view_id": view_digest}}), encoding="utf-8"
    )
    catalog = ArtifactCatalog(root)
    return (
        view_digest,
        catalog.make_id("training_view", view_digest),
        catalog.make_id("tokenizer", tokenizer_digest),
    )


def test_job_spec_is_versioned_and_rejects_unknown_fields():
    spec = JobSpec.from_dict(
        {
            "schema_version": "aster-job-spec-0",
            "kind": "pretrain",
            "recipe_id": "tinylm-overfit-v0",
            "inputs": {"training_view": "training_view:" + "a" * 64, "tokenizer": "tokenizer:" + "b" * 64},
            "parameters": {},
        }
    )
    assert spec.kind == "pretrain"
    with pytest.raises(ValueError, match="Unknown job fields"):
        JobSpec.from_dict({**spec.to_dict(), "command": "rm -rf /"})


def test_recipe_constructs_fixed_pretrain_command_and_checks_provenance(tmp_path):
    view_digest, view_id, tokenizer_id = _artifacts(tmp_path)
    config = tmp_path / "configs" / "tinylm-overfit-v0.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    registry = RecipeRegistry(tmp_path, python="python-test")
    spec = JobSpec("pretrain", "tinylm-overfit-v0", {"training_view": view_id, "tokenizer": tokenizer_id}, {})
    prepared = registry.prepare(spec, tmp_path / "runs" / "workbench" / "job")
    assert prepared.command[:3] == ["python-test", "-m", "aster.training.pretrain"]
    assert "--config" in prepared.command
    assert not any("shell" in part for part in prepared.command)

    evaluation = tmp_path / "artifacts" / "tokenizers" / ("b" * 64) / "evaluation.json"
    evaluation.write_text(json.dumps({"experiment": {"view_id": "c" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        registry.prepare(spec, tmp_path / "other-job")
    assert view_digest == "a" * 64


def test_job_and_run_have_distinct_ids_and_live_outputs(tmp_path):
    _, view_id, tokenizer_id = _artifacts(tmp_path)
    config = tmp_path / "configs" / "tinylm-overfit-v0.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    run_id = "1" * 32

    def fake_runner(command, **kwargs):
        root = Path(command[command.index("--root") + 1])
        run = root / "runs" / run_id
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps({"schema_version": "aster-run-0", "run_id": run_id, "kind": "pretrain", "status": "completed"}),
            encoding="utf-8",
        )
        (run / "events.jsonl").write_text(
            json.dumps({"schema_version": "aster-event-0", "run_id": run_id, "seq": 1, "kind": "started"}) + "\n",
            encoding="utf-8",
        )
        (run / "training-bundle.json").write_text(
            json.dumps({"schema_version": "aster-training-bundle-0", "status": "completed"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    manager = JobManager(tmp_path, python="python-test", process_runner=fake_runner)
    spec = JobSpec("pretrain", "tinylm-overfit-v0", {"training_view": view_id, "tokenizer": tokenizer_id}, {})
    job_id = manager.start(spec)
    deadline = time.monotonic() + 2
    bundle = manager.read(job_id)
    while bundle["job"]["status"] in ("accepted", "running") and time.monotonic() < deadline:
        time.sleep(0.01)
        bundle = manager.read(job_id)
    assert bundle["job"]["status"] == "completed"
    assert bundle["job"]["job_id"] == job_id
    assert bundle["job"]["run_id"] == run_id
    assert job_id != run_id
    assert bundle["run"]["run_id"] == run_id
    assert bundle["training"]["status"] == "completed"
    assert bundle["events"][0]["seq"] == 1


def test_restart_marks_active_job_interrupted(tmp_path):
    base = tmp_path / "runs" / "workbench" / ("d" * 32)
    base.mkdir(parents=True)
    (base / "job.json").write_text(
        json.dumps(
            {
                "schema_version": "aster-service-job-0",
                "job_id": "d" * 32,
                "run_id": None,
                "status": "running",
                "kind": "pretrain",
                "recipe_id": "tinylm-overfit-v0",
                "spec": {},
                "accepted_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    manager = JobManager(tmp_path)
    assert manager.read("d" * 32)["job"]["status"] == "interrupted"
