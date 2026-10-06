import json
from pathlib import Path

import pytest

from aster.corpus.pipeline import digest, json_bytes
from aster.tokenizer.artifact import save_tokenizer, train_aster_tokenizer
from aster.training.dataset import load_training_tokenizer


ROOT = Path(__file__).resolve().parents[2]


def make_lang_artifact(tmp_path, view_id="view-123"):
    tokenizer = train_aster_tokenizer(
        ["こんにちは Aster。こんにちは Aster。"],
        target_vocab_size=280,
        version="lang-v0",
    )
    artifact = tmp_path / "artifact"
    save_tokenizer(tokenizer, artifact)

    provenance = {
        "schema_version": "aster-tokenizer-provenance-0",
        "training_view_id": view_id,
        "fit_split": "train",
        "sealed_test_used_for_fit": False,
    }
    (artifact / "provenance.json").write_bytes(json_bytes(provenance))

    files = {
        path.name: digest(path.read_bytes())
        for path in sorted(artifact.iterdir())
        if path.is_file()
    }
    identity = {
        "schema_version": "aster-tokenizer-artifact-0",
        "training_view_id": view_id,
        "files": files,
    }
    (artifact / "artifact.json").write_bytes(json_bytes(identity))
    return artifact


def test_lang_v0_tokenizer_artifact_loads_for_matching_view(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    tokenizer, payload = load_training_tokenizer(artifact, "view-123")

    assert tokenizer.vocab_size == len(payload["vocab"]) + 2
    probe = "こんにちは Aster。"
    assert tokenizer.decode(tokenizer.encode(probe)) == probe


def test_lang_v0_tokenizer_artifact_rejects_wrong_view(tmp_path):
    artifact = make_lang_artifact(tmp_path)

    with pytest.raises(ValueError, match="training view"):
        load_training_tokenizer(artifact, "other-view")


def test_lang_v0_tokenizer_artifact_detects_tampering(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    vocab = artifact / "vocab.json"
    vocab.write_text(vocab.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file hash mismatch"):
        load_training_tokenizer(artifact, "view-123")


def test_lang_v0_wiring_spec_pins_gate1_outputs_and_overfit_config():
    spec = json.loads((ROOT / "configs/lang-v0-wiring.json").read_text(encoding="utf-8"))

    assert spec["training_view_id"] == (
        "6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d"
    )
    assert spec["tokenizer_artifact_id"] == (
        "37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513"
    )
    assert spec["tiny_lm_config"] == "configs/tinylm-overfit-v0.json"
