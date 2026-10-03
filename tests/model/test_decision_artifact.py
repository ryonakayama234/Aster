"""DecisionModel artifacts keep weights, tokenizer, calibration, and lineage together."""

import json

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.inference.decide import score_candidates
from aster.model.decision_artifact import load_decision_artifact, save_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.tokenizer.artifact import train_aster_tokenizer


def _fixture_model():
    suite = build_calculate_and_store_suite()
    train_cases = suite.cases_for("train")
    texts = [
        serialize_decision_input(
            case.example.state,
            case.example.trajectory,
            candidate,
        )
        for case in train_cases
        for candidate in case.example.candidates
    ]
    tokenizer = train_aster_tokenizer(
        texts,
        target_vocab_size=320,
        min_pair_frequency=1,
        name="DecisionArtifactTestTokenizer",
        version="0",
    )
    torch.manual_seed(17)
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
    return suite, model, tokenizer


def test_decision_artifact_round_trip_preserves_scores_and_lineage(tmp_path):
    suite, model, tokenizer = _fixture_model()
    case = suite.cases_for("dev")[0]
    with torch.no_grad():
        expected = score_candidates(
            model,
            tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()

    suite_sha256 = digest(json_bytes(suite.to_dict()))
    manifest = save_decision_artifact(
        tmp_path / "artifact",
        model,
        tokenizer,
        model_id="decision-artifact-test",
        candidate_builder_id="calculate-and-store-v0",
        suite_id=suite.suite_id,
        suite_sha256=suite_sha256,
        train_config={"steps": 0, "seed": 17},
        temperature=1.25,
        autonomous_threshold=0.8,
        fallback_threshold=0.6,
        source_git_sha="deadbeef",
    )
    loaded, loaded_tokenizer, loaded_manifest = load_decision_artifact(
        tmp_path / "artifact"
    )
    with torch.no_grad():
        actual = score_candidates(
            loaded,
            loaded_tokenizer,
            case.example.state,
            case.example.trajectory,
            case.example.candidates,
        ).detach().cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded_manifest["artifact_id"] == manifest["artifact_id"]
    assert loaded_manifest["suite_sha256"] == suite_sha256
    assert loaded_manifest["candidate_builder_id"] == "calculate-and-store-v0"
    assert loaded_manifest["resume_supported"] is False
    assert loaded_manifest["calibration"]["temperature"] == 1.25
    assert loaded.training is False

    persisted = json.loads(
        (tmp_path / "artifact" / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["model_state_sha256"]
    assert persisted["tokenizer_sha256"]


def test_decision_artifact_rejects_tampered_model_state(tmp_path):
    suite, model, tokenizer = _fixture_model()
    save_decision_artifact(
        tmp_path / "artifact",
        model,
        tokenizer,
        model_id="decision-artifact-test",
        candidate_builder_id="calculate-and-store-v0",
        suite_id=suite.suite_id,
        suite_sha256=digest(json_bytes(suite.to_dict())),
        train_config={"steps": 0},
        temperature=1.0,
        autonomous_threshold=0.8,
        fallback_threshold=0.6,
    )
    with (tmp_path / "artifact" / "model.pt").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="state digest mismatch"):
        load_decision_artifact(tmp_path / "artifact")


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_decision_artifact_rejects_invalid_temperature(tmp_path, temperature):
    suite, model, tokenizer = _fixture_model()
    with pytest.raises(ValueError, match="positive and finite"):
        save_decision_artifact(
            tmp_path / "artifact",
            model,
            tokenizer,
            model_id="decision-artifact-test",
            candidate_builder_id="calculate-and-store-v0",
            suite_id=suite.suite_id,
            suite_sha256=digest(json_bytes(suite.to_dict())),
            train_config={"steps": 0},
            temperature=temperature,
            autonomous_threshold=0.8,
            fallback_threshold=0.6,
        )
