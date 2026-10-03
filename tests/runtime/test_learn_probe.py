"""Fixed LEARN-v0 development probe lineage and split invariants."""

import json

from aster.training.learn_probe import (
    ACT_RECIPE_ID,
    ACT_TASK_ID,
    CORRECTION_TASK,
    DEFAULT_ACT_RUN_ID,
    PROBE_FAMILY_ID,
    PROBE_ID,
    SIBLING_TASK,
    TRAIN_CONFIG,
    build_learn_dev_probe_suite,
    resolve_act_parent_artifact,
)


def _write_dummy_registered_artifact(root, digest: str):
    path = root / "artifacts" / "decision_models" / digest
    (path / "tokenizer").mkdir(parents=True)
    for name in (
        "model.pt",
        "tokenizer/manifest.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.json",
        "tokenizer/special_tokens.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    artifact_id = f"decision_model:{digest}"
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "candidate_builder_id": "calculate-and-store-v0",
            }
        ),
        encoding="utf-8",
    )
    return artifact_id, path


def test_resolve_act_parent_artifact_uses_logged_logical_id(tmp_path):
    artifact_id, artifact_path = _write_dummy_registered_artifact(tmp_path, "a" * 64)
    run_dir = tmp_path / "runs" / DEFAULT_ACT_RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "aster-run-0",
                "run_id": DEFAULT_ACT_RUN_ID,
                "kind": "agent",
                "status": "completed",
                "inputs": {
                    "recipe_id": ACT_RECIPE_ID,
                    "policy_mode": "model_only",
                    "task_id": ACT_TASK_ID,
                    "decision_model_artifact_id": artifact_id,
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_id, resolved_path = resolve_act_parent_artifact(tmp_path)

    assert resolved_id == artifact_id
    assert resolved_path == artifact_path.resolve()


def test_learn_dev_probe_is_fixed_key_shift_and_not_confirmatory():
    assert TRAIN_CONFIG.seed == 42
    assert TRAIN_CONFIG.steps == 100
    assert TRAIN_CONFIG.learning_rate == 3e-3

    correction_visible = {
        key: value for key, value in CORRECTION_TASK.items() if key not in {"task_id", "store_as"}
    }
    sibling_visible = {
        key: value for key, value in SIBLING_TASK.items() if key not in {"task_id", "store_as"}
    }
    assert correction_visible == sibling_visible
    assert CORRECTION_TASK["store_as"] == "diagnostic_total"
    assert SIBLING_TASK["store_as"] == "sibling_total"

    suite = build_learn_dev_probe_suite()
    assert suite.suite_id == PROBE_ID
    sibling_cases = suite.cases_for("test")
    assert len(sibling_cases) == 4
    assert {case.slice_name for case in sibling_cases} == {"uncorrected_sibling"}
    assert {case.leakage_group for case in sibling_cases} == {PROBE_FAMILY_ID}
    assert all(case.example.state.task["store_as"] == "sibling_total" for case in sibling_cases)
    assert suite.cases_for("calibration")
