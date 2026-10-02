"""ACT v0 persists model-only decisions separately from executed runtime truth."""

import json

import pytest


torch = pytest.importorskip("torch")

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.corpus.pipeline import digest, json_bytes
from aster.model.decision_artifact import save_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.decision import serialize_decision_input
from aster.runtime.service_learned_agent import (
    CANDIDATE_BUILDER_ID,
    SERIALIZER_ID,
    TASK_ID,
    run_learned_agent_recipe,
)
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
        name="ActRunnerTestTokenizer",
        version="0",
    )
    torch.manual_seed(29)
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
    artifact = tmp_path / "decision-model"
    manifest = save_decision_artifact(
        artifact,
        model,
        tokenizer,
        model_id="act-runner-test",
        candidate_builder_id=CANDIDATE_BUILDER_ID,
        suite_id=suite.suite_id,
        suite_sha256=digest(json_bytes(suite.to_dict())),
        train_config={"steps": 0, "seed": 29},
        temperature=1.0,
        autonomous_threshold=0.8,
        fallback_threshold=0.6,
    )
    return artifact, manifest


def test_learned_agent_act_wiring_persists_aligned_decisions(tmp_path):
    artifact, manifest = _artifact(tmp_path)
    run_path = run_learned_agent_recipe(tmp_path / "job", artifact, max_steps=4)

    bundle = json.loads((run_path / "agent-bundle.json").read_text(encoding="utf-8"))
    decision = bundle["decision"]
    trajectory = bundle["trajectory"]
    traces = decision["traces"]

    assert bundle["schema_version"] == "aster-agent-bundle-0"
    assert bundle["recipe_id"] == "agent-decision-model-v0"
    assert bundle["task"]["task_id"] == TASK_ID
    assert decision["decision_model_artifact_id"] == manifest["artifact_id"]
    assert decision["policy_mode"] == "model_only"
    assert decision["serializer_id"] == SERIALIZER_ID
    assert decision["candidate_builder_id"] == CANDIDATE_BUILDER_ID
    assert decision["model_update"] is False
    assert decision["fallback_enabled"] is False
    assert decision["rule_intervention_enabled"] is False
    assert decision["wiring_gate"]["passed"] is True
    assert decision["wiring_gate"]["weights_unchanged"] is True
    assert decision["wiring_gate"]["decision_transition_alignment"] is True
    assert len(traces) == len(trajectory) >= 1

    for trace, transition in zip(traces, trajectory, strict=True):
        selected = trace["candidates"][trace["selected_index"]]
        assert trace["step"] == transition["step"]
        assert selected == transition["action"]

    assert (run_path / "decision-traces.jsonl").is_file()
    assert (run_path / "trajectory.jsonl").is_file()
    assert (run_path / "evaluation.json").is_file()
    assert isinstance(decision["capability_gate"]["passed"], bool)
    assert decision["capability_gate"]["task_id"] == TASK_ID
