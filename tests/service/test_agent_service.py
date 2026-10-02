import json
from pathlib import Path
import subprocess
import time

import pytest

from aster.service.contracts import JobSpec
from aster.service.jobs import JobManager
from aster.service.recipes import RecipeRegistry


def test_agent_job_spec_and_recipe_are_fixed(tmp_path):
    spec = JobSpec.from_dict(
        {
            "schema_version": "aster-job-spec-0",
            "kind": "agent",
            "recipe_id": "agent-calculate-store-v0",
            "inputs": {},
            "parameters": {},
        }
    )
    registry = RecipeRegistry(tmp_path, python="python-test")
    prepared = registry.prepare(spec, tmp_path / "runs" / "workbench" / "job")

    assert prepared.command[:3] == ["python-test", "-m", "aster.runtime.service_agent"]
    assert prepared.command[-2] == "--root"
    assert prepared.timeout_seconds == 60


def test_agent_job_rejects_extra_inputs_and_parameters(tmp_path):
    registry = RecipeRegistry(tmp_path, python="python-test")

    extra_input = JobSpec(
        "agent",
        "agent-calculate-store-v0",
        {"task": "arbitrary"},
        {},
    )
    try:
        registry.prepare(extra_input, tmp_path / "job-a")
    except ValueError as error:
        assert "inputs" in str(error)
    else:
        raise AssertionError("agent recipe accepted an extra input")

    extra_parameter = JobSpec(
        "agent",
        "agent-calculate-store-v0",
        {},
        {"module": "arbitrary.module"},
    )
    try:
        registry.prepare(extra_parameter, tmp_path / "job-b")
    except ValueError as error:
        assert "parameters" in str(error)
    else:
        raise AssertionError("agent recipe accepted an extra parameter")


def test_job_manager_exposes_agent_bundle_without_local_paths(tmp_path):
    run_id = "2" * 32

    def fake_runner(command, **kwargs):
        root = Path(command[command.index("--root") + 1])
        run = root / "runs" / run_id
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps(
                {
                    "schema_version": "aster-run-0",
                    "run_id": run_id,
                    "kind": "agent",
                    "status": "completed",
                    "agent": "agent-bundle.json",
                    "trajectory": "trajectory.jsonl",
                    "evaluation": "evaluation.json",
                }
            ),
            encoding="utf-8",
        )
        (run / "events.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "aster-event-0",
                    "run_id": run_id,
                    "seq": 1,
                    "producer": "runtime",
                    "kind": "started",
                    "data": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "agent-bundle.json").write_text(
            json.dumps(
                {
                    "schema_version": "aster-agent-bundle-0",
                    "recipe_id": "agent-calculate-store-v0",
                    "policy_id": "rule-calculate-store-v0",
                    "trajectory": [],
                    "evaluation": {"summary": {"task_success": True}},
                    "final_memory": {"total": 42},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    manager = JobManager(tmp_path, python="python-test", process_runner=fake_runner)
    spec = JobSpec("agent", "agent-calculate-store-v0", {}, {})
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
    assert bundle["agent"]["schema_version"] == "aster-agent-bundle-0"
    assert bundle["agent"]["final_memory"] == {"total": 42}
    assert bundle["outputs"]["agent"] == bundle["agent"]
    assert str(tmp_path) not in json.dumps(bundle)
    assert "agent.run" in manager.status()["capabilities"]


def test_learned_agent_recipe_accepts_only_registered_decision_model(tmp_path):
    digest = "a" * 64
    model_root = tmp_path / "artifacts" / "decision_models" / digest
    tokenizer_root = model_root / "tokenizer"
    tokenizer_root.mkdir(parents=True)
    (model_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": f"decision_model:{digest}",
                "candidate_builder_id": "calculate-and-store-v0",
            }
        ),
        encoding="utf-8",
    )
    (model_root / "model.pt").write_bytes(b"placeholder")
    for name in ("manifest.json", "vocab.json", "merges.json", "special_tokens.json"):
        (tokenizer_root / name).write_text("{}", encoding="utf-8")

    registry = RecipeRegistry(tmp_path, python="python-test")
    spec = JobSpec(
        "agent",
        "agent-decision-model-v0",
        {"decision_model": f"decision_model:{digest}"},
        {},
    )
    prepared = registry.prepare(spec, tmp_path / "job")

    assert prepared.command[:3] == [
        "python-test",
        "-m",
        "aster.runtime.service_learned_agent",
    ]
    assert prepared.command[3:5] == ["--root", str(tmp_path / "job")]
    assert prepared.command[5] == "--artifact"
    assert prepared.command[6] == str(model_root)
    assert prepared.timeout_seconds == 60

    with pytest.raises(ValueError, match="inputs"):
        registry.prepare(
            JobSpec("agent", "agent-decision-model-v0", {"model_path": "/tmp/x"}, {}),
            tmp_path / "bad-job",
        )
