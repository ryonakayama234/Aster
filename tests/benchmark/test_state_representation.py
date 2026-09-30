from dataclasses import replace
import json
import subprocess
import sys

import pytest

from aster.benchmark.state_representation import build_suite, audit_suite, compact_input, replay, run_preflight
from aster.records.trajectory import Trajectory


def test_executed_pairs_and_candidate_shortcuts():
    suite, pairs = build_suite()
    report, rows = audit_suite(suite, pairs)
    assert report["cases"] == report["replay_verified"] == report["rule_continuation_success"] == 48
    assert report["relations"] == {"same_step_different_target": 32, "same_target_different_step": 32}
    assert report["compact_target_collisions"] == 0
    assert report["candidate_only_ambiguous_sets"] > 0
    assert report["step_only_correct"] == 16
    assert report["first_candidate_correct"] == 12
    assert report["candidate_only_in_sample_oracle_upper_correct"] == 36
    assert report["compact_bytes"] < report["raw_bytes"]
    for row in rows:
        steps = [t["step"] for t in row["rule_continuation"]["trajectory"]]
        assert steps == list(range(len(steps)))


def test_replay_rejects_forged_prefix_observation_and_state():
    suite, _ = build_suite()
    example = suite.cases[3].example  # calculated, no delay
    transition = example.trajectory.transitions[0]
    forged = replace(transition, observation=replace(transition.observation, output=999))
    with pytest.raises(ValueError, match="Prefix replay"):
        replay(replace(example, trajectory=Trajectory((forged,))))
    with pytest.raises(ValueError, match="State replay"):
        replay(replace(example, state=replace(example.state, step=999)))
    with pytest.raises(ValueError, match="Teacher replay"):
        replay(replace(example, target_index=0))


def test_projection_accepts_observations_only_and_preserves_numeric_values():
    suite, _ = build_suite()
    example = suite.cases[3].example
    payload = json.loads(compact_input(example.state, example.trajectory, example.candidates[0]))
    assert payload["events"][0][1]["output"] == 5
    assert set(payload) == {"schema", "task", "memory", "events", "last_failure", "candidate"}
    assert "phase" not in payload and "step" not in payload and "teacher" not in payload
    other_label = replace(example, target_index=0, teacher="different")
    assert compact_input(other_label.state, other_label.trajectory, example.candidates[0]) == compact_input(example.state, example.trajectory, example.candidates[0])


def test_suite_roundtrip_and_preflight_run(tmp_path):
    suite, pairs = build_suite()
    assert suite.to_dict() == build_suite()[0].to_dict()
    assert not suite.cases_for("train") and not suite.cases_for("test")
    path = run_preflight(tmp_path, source_git_sha="test")
    report = json.loads((path / "preflight.json").read_text())
    assert report["model_scored_cases"] == report["test_scored_cases"] == 0
    assert json.loads((path / "run.json").read_text())["status"] == "completed"
    wrong = replace(suite, cases=(replace(suite.cases[0], split="test"),))
    with pytest.raises(ValueError, match="dev only"):
        audit_suite(wrong, [])
    with pytest.raises(ValueError, match="relation"):
        audit_suite(suite, [dict(relation="unknown", left=pairs[0]["left"], right=pairs[0]["right"])])


def test_preflight_import_does_not_load_optional_torch():
    subprocess.run([sys.executable, "-c", "import sys; from aster.benchmark.state_representation import build_suite; build_suite(); assert 'torch' not in sys.modules"], check=True)
