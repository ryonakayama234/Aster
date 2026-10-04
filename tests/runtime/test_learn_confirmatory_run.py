"""Gate 4 confirmatory runner invariants that do not require the local ACT artifact."""

import json
from pathlib import Path

import pytest

from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
)
from aster.training.learn_confirmatory_run import (
    build_confirmatory_sibling_suite,
    discover_completed_confirmatory_units,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_confirmatory_sibling_suite_keeps_frozen_family_out_of_training():
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    family = families[0]
    assert isinstance(family, dict)
    family_id = family["family_id"]
    sibling_task = family["sibling_task"]
    assert isinstance(family_id, str)
    assert isinstance(sibling_task, dict)

    suite = build_confirmatory_sibling_suite(manifest, family_id)

    assert suite.training_examples() == ()
    assert suite.cases_for("calibration")
    test_cases = suite.cases_for("test")
    assert len(test_cases) == 4
    assert {case.leakage_group for case in test_cases} == {family_id}
    assert {
        case.example.state.task["store_as"]
        for case in test_cases
    } == {sibling_task["store_as"]}


def test_failed_attempt_without_primary_endpoint_is_retryable(tmp_path):
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_id = _first_family_id(manifest)
    _write_attempt(
        tmp_path,
        manifest,
        run_id="a" * 32,
        family_id=family_id,
        seed=42,
        status="failed",
        endpoint_emitted=False,
    )

    assert discover_completed_confirmatory_units(tmp_path, manifest) == {}


def test_failed_attempt_after_primary_endpoint_blocks_retry(tmp_path):
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_id = _first_family_id(manifest)
    _write_attempt(
        tmp_path,
        manifest,
        run_id="b" * 32,
        family_id=family_id,
        seed=42,
        status="failed",
        endpoint_emitted=True,
    )

    with pytest.raises(RuntimeError, match="already emitted primary endpoint"):
        discover_completed_confirmatory_units(tmp_path, manifest)


def test_running_attempt_blocks_duplicate_measurement(tmp_path):
    manifest = load_confirmatory_manifest(_ROOT / DEFAULT_MANIFEST_PATH)
    family_id = _first_family_id(manifest)
    _write_attempt(
        tmp_path,
        manifest,
        run_id="c" * 32,
        family_id=family_id,
        seed=42,
        status="running",
        endpoint_emitted=False,
    )

    with pytest.raises(RuntimeError, match="still marked running"):
        discover_completed_confirmatory_units(tmp_path, manifest)


def _first_family_id(manifest):
    design = manifest["family_design"]
    assert isinstance(design, dict)
    families = design["families"]
    assert isinstance(families, list)
    first = families[0]
    assert isinstance(first, dict)
    family_id = first["family_id"]
    assert isinstance(family_id, str)
    return family_id


def _write_attempt(
    root,
    manifest,
    *,
    run_id,
    family_id,
    seed,
    status,
    endpoint_emitted,
):
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    run = {
        "schema_version": "aster-run-0",
        "run_id": run_id,
        "kind": "correction_transfer_experiment",
        "status": status,
        "inputs": {
            "provenance": {
                "scope": "confirmatory_measurement",
                "confirmatory_evidence": True,
                "protocol_id": manifest["protocol_id"],
                "manifest_sha256": manifest_sha256,
                "family_id": family_id,
                "seed": seed,
            }
        },
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if endpoint_emitted:
        for arm in ("replay", "correction"):
            benchmark = run_dir / f"benchmark-{arm}"
            benchmark.mkdir()
            (benchmark / "benchmark.json").write_text("{}\n", encoding="utf-8")
            (benchmark / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
