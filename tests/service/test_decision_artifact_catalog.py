"""DecisionModel registration keeps inference artifacts immutable and path-free for Service callers."""

import json

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.model.decision_artifact import save_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.service.artifacts import ArtifactCatalog
from aster.tokenizer.artifact import train_aster_tokenizer


def _artifact(tmp_path):
    suite = build_calculate_and_store_suite()
    train_cases = suite.cases_for("train")
    texts = [
        serialize_decision_input(case.example.state, case.example.trajectory, candidate)
        for case in train_cases
        for candidate in case.example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=320,
        min_pair_frequency=1,
        name="DecisionCatalogTestTokenizer",
        version="0",
    )
    torch.manual_seed(23)
    model = DecisionModel(
        TinyLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=2048,
                width=8,
                heads=2,
                layers=1,
            )
        )
    )
    source = tmp_path / "source-model"
    manifest = save_decision_artifact(
        source,
        model,
        tokenizer,
        model_id="decision-catalog-test",
        candidate_builder_id="calculate-and-store-v0",
        suite_id=suite.suite_id,
        suite_sha256=digest(json_bytes(suite.to_dict())),
        train_config={"steps": 0, "seed": 23},
        temperature=1.0,
        autonomous_threshold=0.8,
        fallback_threshold=0.6,
        source_git_sha="catalog-test",
    )
    return source, manifest


def test_register_decision_model_returns_logical_id_and_lists_metadata(tmp_path):
    root = tmp_path / "aster-root"
    root.mkdir()
    source, manifest = _artifact(tmp_path)
    catalog = ArtifactCatalog(root)

    ref = catalog.register_decision_model(source)
    resolved = catalog.resolve(ref.artifact_id, "decision_model")

    assert ref.artifact_id == manifest["artifact_id"]
    assert ref.kind == "decision_model"
    assert resolved == root / "artifacts" / "decision_models" / ref.digest
    assert resolved != source
    listed = catalog.all()["decision_models"]
    assert listed == [
        {
            "artifact_id": ref.artifact_id,
            "kind": "decision_model",
            "digest": ref.digest,
            "model_id": "decision-catalog-test",
            "candidate_builder_id": "calculate-and-store-v0",
            "suite_id": "calculate-and-store-v0",
            "source_git_sha": "catalog-test",
            "provenance_valid": True,
        }
    ]
    assert str(source) not in json.dumps(listed)


def test_register_decision_model_is_idempotent(tmp_path):
    root = tmp_path / "aster-root"
    root.mkdir()
    source, _ = _artifact(tmp_path)
    catalog = ArtifactCatalog(root)

    first = catalog.register_decision_model(source)
    second = catalog.register_decision_model(source)

    assert second == first


def test_register_decision_model_rejects_tampered_weights(tmp_path):
    root = tmp_path / "aster-root"
    root.mkdir()
    source, _ = _artifact(tmp_path)
    with (source / "model.pt").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="state digest mismatch"):
        ArtifactCatalog(root).register_decision_model(source)


def test_register_decision_model_rejects_symlinked_content(tmp_path):
    root = tmp_path / "aster-root"
    root.mkdir()
    source, _ = _artifact(tmp_path)
    link = source / "unexpected-link"
    try:
        link.symlink_to(source / "manifest.json")
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        ArtifactCatalog(root).register_decision_model(source)
