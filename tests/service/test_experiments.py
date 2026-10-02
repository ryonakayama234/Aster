import hashlib
import json
from pathlib import Path

import pytest

from aster.service.experiments import ExperimentCatalog, FAILURE_AUDIT_ID


def test_failure_audit_projection_matches_committed_evidence():
    root = Path.cwd().resolve()
    catalog = ExperimentCatalog(root)
    bundle = catalog.read(FAILURE_AUDIT_ID)

    assert bundle["schema_version"] == "aster-experiment-bundle-0"
    evidence = bundle["evidence"]
    assert isinstance(evidence, dict)
    report = root / "reports" / "decision-failure-audit-v0.json"
    assert evidence["source_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()

    arms = bundle["arms"]
    assert isinstance(arms, list)
    fixed = next(arm for arm in arms if arm["arm_id"] == "seed-42--eight-fixed")
    case = next(item for item in fixed["cases"] if item["case_id"] == "subtract/numbers/phase-0/delay-0")

    assert case["correct"] is False
    assert case["target_action"]["name"] == "calculator"
    assert case["selected_action"]["name"] == "memory.get"
    assert case["target_tokenization"]["length"] == 23
    assert case["candidate_scores"] is None
    assert case["candidate_scores_status"] == "unavailable_from_committed_evidence"
    assert fixed["learning_curve"] is None
    assert fixed["learning_curve_status"] == "unavailable_from_committed_evidence"
    assert str(root) not in json.dumps(bundle)


def test_experiment_catalog_rejects_arbitrary_ids():
    catalog = ExperimentCatalog(Path.cwd())
    with pytest.raises(ValueError, match="Unknown experiment ID"):
        catalog.read("../../private")
