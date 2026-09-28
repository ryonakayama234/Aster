import json

from aster.runtime.service_agent import run_calculate_store_recipe


def test_service_agent_emits_replayable_calculate_store_bundle(tmp_path):
    run_path = run_calculate_store_recipe(tmp_path)

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    bundle = json.loads((run_path / "agent-bundle.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_path / "evaluation.json").read_text(encoding="utf-8"))
    trajectory = [
        json.loads(line)
        for line in (run_path / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert run["status"] == "completed"
    assert run["kind"] == "agent"
    assert run["agent"] == "agent-bundle.json"
    assert bundle["schema_version"] == "aster-agent-bundle-0"
    assert bundle["recipe_id"] == "agent-calculate-store-v0"
    assert bundle["policy_id"] == "rule-calculate-store-v0"
    assert bundle["task"] == {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }
    assert [item["action"]["name"] or item["action"]["kind"] for item in trajectory] == [
        "calculator",
        "memory.put",
        "memory.get",
        "stop",
    ]
    assert trajectory[0]["observation"]["output"] == 42
    assert trajectory[0]["state_after"]["memory"] == {}
    assert trajectory[1]["state_after"]["memory"] == {"total": 42}
    assert bundle["final_memory"] == {"total": 42}
    assert evaluation["summary"]["task_success"] is True
    assert evaluation["summary"]["steps"] == 4
    assert evaluation["summary"]["goal_verified"] is True
    assert evaluation["summary"]["first_goal_verified_step"] == 2
    assert len(evaluation["trajectory_sha256"]) == 64
