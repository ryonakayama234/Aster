"""OBS v0a fixture regression: test only structural capability, never broad generalization."""
from __future__ import annotations

from json import loads

import pytest

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.records.decision import serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.transition import Action
from aster.runtime.observation_recovery import (
    BRANCHES, FIXED_CANDIDATES, RETRY, STOP, ObservationBlindPolicy,
    ReadRetryCandidates, ReferenceReadPolicy, _append_action,
    assert_observation_only_pair, continue_episode, create_episode,
    model_preflight, run_reference_checks,
)


def test_g0_true_observation_only_pair_and_replay(tmp_path):
    success, fault = (create_episode(branch) for branch in BRANCHES)
    assert_observation_only_pair(success, fault)
    for episode in (success, fault):
        assert episode.trajectory.transitions[0].action == RETRY
        assert episode.trajectory.transitions[0].state_before["memory"] == {"answer": 42}
        assert episode.trajectory.transitions[0].state_after["memory"] == {"answer": 42}
        assert ReadRetryCandidates().build(
            episode.context.snapshot(1), episode.trajectory, ("memory.get", "stop"),
        ) == FIXED_CANDIDATES
        path = tmp_path / (episode.branch + ".jsonl")
        episode.recorder.write_jsonl(path)
        assert TrajectoryRecorder.read_jsonl(path) == episode.trajectory

    assert success.trajectory.last.observation.ok is True
    assert success.trajectory.last.evaluation.goal_satisfied is True
    assert fault.trajectory.last.observation.ok is False
    assert fault.trajectory.last.observation.error.code == "transient_read_error"
    assert fault.trajectory.last.evaluation.goal_satisfied is False
    for candidate in FIXED_CANDIDATES:
        left = loads(serialize_decision_input(success.context.snapshot(1), success.trajectory, candidate))
        right = loads(serialize_decision_input(fault.context.snapshot(1), fault.trajectory, candidate))
        assert left["candidate"] == right["candidate"]
        assert left["state"] == right["state"]
        assert left["history"][0]["action"] == right["history"][0]["action"]
        assert left["history"][0]["observation"] != right["history"][0]["observation"]
        assert "evaluation" not in left["history"][0]
        assert "evaluation" not in right["history"][0]


def test_g1_reference_verifies_both_while_blind_chooses_same_first_action():
    result = run_reference_checks()
    assert result["G0_observation_only"] == "PASS"
    assert result["G1_reference"] == "PASS"
    assert all(result["reference"][branch]["task_success"] for branch in BRANCHES)
    assert result["observation_blind"]["success"]["task_success"] is True
    assert result["observation_blind"]["transient_failure"]["task_success"] is False
    assert result["observation_blind"]["success"]["first_continuation_action"] == (
        result["observation_blind"]["transient_failure"]["first_continuation_action"]
    )

    success = create_episode("success")
    success_actions = continue_episode(success, ReferenceReadPolicy())
    assert [step.action for step in success_actions.transitions] == [RETRY, STOP]

    fault = create_episode("transient_failure")
    fault_actions = continue_episode(fault, ReferenceReadPolicy())
    assert [step.action for step in fault_actions.transitions] == [RETRY, RETRY, STOP]
    assert fault_actions.transitions[1].observation.ok
    assert fault_actions.transitions[1].evaluation.goal_satisfied
    assert fault_actions.transitions[-1].evaluation.goal_satisfied


def test_no_old_v0_builder_mutation():
    # An old ACT artifact with its old builder still fail-stops, untouched.
    episode = create_episode("transient_failure")
    candidates = CalculateAndStoreCandidates().build(
        episode.context.snapshot(1), episode.trajectory, ("memory.get", "stop"),
    )
    assert candidates == (Action.stop("tool_failure"),)


def test_reject_missing_or_wrong_prefix():
    with pytest.raises(ValueError, match="Unknown"):
        create_episode("corrupt")
    fault = create_episode("transient_failure")
    with pytest.raises(ValueError, match="continuation"):
        continue_episode(fault, ReferenceReadPolicy(), max_continuation=1)
    _append_action(fault, RETRY)
    with pytest.raises(ValueError, match="one scripted prefix"):
        continue_episode(fault, ReferenceReadPolicy())


def test_preflight_checks_all_three_decision_positions_without_truncation():
    class Model:
        class Backbone:
            class Config:
                context_length = 999999
            config = Config()
        backbone = Backbone()

    class Tokenizer:
        def encode(self, text, *, add_bos, add_eos):
            assert add_bos and add_eos
            return list(text.encode("utf-8")) + [256, 257]

    result = model_preflight(Model(), Tokenizer())
    assert result["status"] == "PASS"
    assert set(result["input_lengths"]) == {
        "success_postprefix", "transient_failure_postprefix",
        "transient_failure_after_retry",
    }
    assert result["input_lengths"]["transient_failure_after_retry"][0] > (
        result["input_lengths"]["transient_failure_postprefix"][0]
    )
    Model.Backbone.Config.context_length = 10
    result = model_preflight(Model(), Tokenizer())
    assert result["status"] == "BLOCKED_CONTEXT"
    assert result["max_input_length"] > result["context_length"]


def test_g2_fake_model_adapter_success_failure_and_context_block(monkeypatch):
    """Exercise the G2 adapter/traces without claiming that a real model succeeded."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from aster.records.decision import DecisionTrace
    from aster.runtime.observation_recovery import run_frozen_model
    import aster.inference.decide as decide

    class FakeModel:
        backbone = SimpleNamespace(config=SimpleNamespace(context_length=999999))

        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class FakeTokenizer:
        def encode(self, text, *, add_bos, add_eos):
            assert add_bos and add_eos
            return list(text.encode("utf-8")) + [256, 257]

    class FakeObservationPolicy:
        def __init__(self, model, tokenizer, *, candidate_builder, model_id):
            self.candidate_builder = candidate_builder
            self.traces = []

        def decide(self, state, trajectory, available_actions):
            candidates = self.candidate_builder.build(state, trajectory, available_actions)
            choice = STOP if trajectory.last.evaluation.goal_satisfied else RETRY
            index = candidates.index(choice)
            self.traces.append(DecisionTrace(
                step=state.step, model_id="synthetic-test",
                candidates=candidates, scores=(0.0, 0.0),
                probabilities=(0.5, 0.5), selected_index=index,
            ))
            return choice

    monkeypatch.setattr(decide, "ModelPolicy", FakeObservationPolicy)
    model = FakeModel()
    result = run_frozen_model(model, FakeTokenizer(), model_id="synthetic-test")
    assert result["G2"] == "DEV_PAIR_SUCCESS_ONLY"
    assert result["weights_unchanged"]
    assert len(result["branches"]["success"]["decision_traces"]) == 1
    assert len(result["branches"]["transient_failure"]["decision_traces"]) == 2

    class FakeBlindModelPolicy(FakeObservationPolicy):
        def decide(self, state, trajectory, available_actions):
            candidates = self.candidate_builder.build(state, trajectory, available_actions)
            self.traces.append(DecisionTrace(
                step=state.step, model_id="synthetic-blind",
                candidates=candidates, scores=(0.0, 1.0),
                probabilities=(0.1, 0.9), selected_index=1,
            ))
            return STOP

    monkeypatch.setattr(decide, "ModelPolicy", FakeBlindModelPolicy)
    blind = run_frozen_model(model, FakeTokenizer(), model_id="synthetic-blind")
    assert blind["G2"] == "NOT_SUPPORTED_ON_DEV_PAIR"

    model.backbone.config.context_length = 10
    blocked = run_frozen_model(model, FakeTokenizer(), model_id="synthetic-test")
    assert blocked["G2"] == "BLOCKED_CONTEXT"
    assert "branches" not in blocked
