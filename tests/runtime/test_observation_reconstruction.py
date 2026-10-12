"""OBS v0c: structural / synthetic-policy tests, NOT real-model capability evidence."""
from __future__ import annotations

from json import loads
from types import SimpleNamespace

import pytest

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.evaluator.episode import evaluate_episode
from aster.records.decision import DecisionTrace, serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.transition import Action
from aster.runtime.observation_reconstruction import (
    AVAILABLE, CALCULATE, MAX_MODEL_ACTIONS, PREFIX, STOP,
    BlindStopPolicy, MissingKeyCandidates, ReferenceReconstructionPolicy,
    _append_action, continue_episode, create_episode, model_preflight,
    run_frozen_model, run_reference_checks,
)
from aster.tools.builtin.calculator import CALCULATOR_SPEC
from aster.tools.builtin.memory import (
    MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put,
)
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


def test_truthful_missing_key_prefix_and_serialization(tmp_path):
    episode = create_episode()
    assert len(episode.trajectory) == 1
    prefix = episode.trajectory.last
    assert prefix is not None
    assert prefix.action == PREFIX
    assert prefix.observation.accepted and not prefix.observation.ok
    assert prefix.observation.error is not None
    assert prefix.observation.error.code == "memory_key_not_found"
    assert prefix.state_before["memory"] == {}
    assert prefix.state_after["memory"] == {}
    assert not prefix.evaluation.goal_satisfied

    state = episode.context.snapshot(1)
    candidates = MissingKeyCandidates().build(state, episode.trajectory, AVAILABLE)
    assert candidates == (CALCULATE, PREFIX, STOP)
    for candidate in candidates:
        payload = loads(serialize_decision_input(state, episode.trajectory, candidate))
        assert payload["state"]["memory"] == {}
        assert set(payload["history"][0]) == {"action", "observation"}
        assert payload["history"][0]["observation"]["error"]["code"] == "memory_key_not_found"
        assert payload["candidate"] == candidate.to_dict()
        assert '"value":42' not in serialize_decision_input(
            state, episode.trajectory, candidate,
        )
    path = tmp_path / "trajectory.jsonl"
    episode.recorder.write_jsonl(path)
    assert TrajectoryRecorder.read_jsonl(path) == episode.trajectory


def test_put_value_comes_from_real_calculator_observation():
    episode = create_episode()
    _append_action(episode, CALCULATE)
    assert episode.trajectory.last is not None
    assert episode.trajectory.last.observation.output == 42
    candidates = MissingKeyCandidates().build(
        episode.context.snapshot(2), episode.trajectory, AVAILABLE,
    )
    assert candidates == (
        CALCULATE, Action.tool("memory.put", key="answer", value=42), PREFIX, STOP,
    )

    # A test-only non-oracle calculator output: candidate MUST NOT secretly use
    # task[23+19] or the independent TaskEvaluator expected value.
    altered = create_episode()
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, lambda _ctx, _args: 7)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    altered.executor = ToolExecutor(registry)
    _append_action(altered, CALCULATE)
    options = MissingKeyCandidates().build(
        altered.context.snapshot(2), altered.trajectory, AVAILABLE,
    )
    assert Action.tool("memory.put", key="answer", value=7) in options
    assert Action.tool("memory.put", key="answer", value=42) not in options
    _append_action(altered, Action.tool("memory.put", key="answer", value=7))
    _append_action(altered, PREFIX)
    _append_action(altered, STOP)
    assert altered.context.memory == {"answer": 7}
    assert not evaluate_episode(altered.trajectory).task_success


def test_reference_uses_real_tools_and_blind_fails(tmp_path):
    result = run_reference_checks()
    assert result["G0_structure"] == "PASS"
    assert result["G1_reference"] == "PASS"
    assert result["reference_task_success"] is True
    assert result["blind_task_success"] is False

    episode = create_episode()
    trajectory = continue_episode(episode, ReferenceReconstructionPolicy())
    assert [t.action for t in trajectory.transitions] == [
        PREFIX, CALCULATE, Action.tool("memory.put", key="answer", value=42),
        PREFIX, STOP,
    ]
    assert [t.step for t in trajectory.transitions] == list(range(5))
    assert [t.evaluation.goal_satisfied for t in trajectory.transitions] == [
        False, False, False, True, True,
    ]
    assert episode.context.memory == {"answer": 42}
    report = evaluate_episode(trajectory)
    assert report.task_success and report.terminal_reached
    assert report.failed_executions == 1  # The truthful scripted prefix.
    assert report.rejected_actions == 0

    path = tmp_path / "trajectory.jsonl"
    episode.recorder.write_jsonl(path)
    assert TrajectoryRecorder.read_jsonl(path) == trajectory

    blind = create_episode()
    trajectory = continue_episode(blind, BlindStopPolicy())
    assert [t.action for t in trajectory.transitions] == [PREFIX, STOP]
    assert not evaluate_episode(trajectory).task_success


def test_old_act_builder_still_fail_stops_and_new_builder_needs_prefix():
    episode = create_episode()
    assert CalculateAndStoreCandidates().build(
        episode.context.snapshot(1), episode.trajectory, AVAILABLE,
    ) == (Action.stop("tool_failure"),)
    empty = TrajectoryRecorder().trajectory()
    with pytest.raises(ValueError, match="scripted prefix"):
        MissingKeyCandidates().build(episode.context.snapshot(0), empty, AVAILABLE)
    with pytest.raises(ValueError, match="task/tool"):
        MissingKeyCandidates().build(episode.context.snapshot(1), episode.trajectory, ("stop",))


def test_bad_actions_and_budget_are_rejected():
    class InvalidPolicy:
        def decide(self, state, trajectory, available_actions):
            return Action.tool("memory.put", key="answer", value=42)

    episode = create_episode()
    with pytest.raises(ValueError, match="outside"):
        continue_episode(episode, InvalidPolicy())

    with pytest.raises(ValueError, match="budget"):
        continue_episode(create_episode(), BlindStopPolicy(), max_actions=5)
    completed = create_episode()
    continue_episode(completed, BlindStopPolicy())
    with pytest.raises(ValueError, match="scripted prefix"):
        continue_episode(completed, BlindStopPolicy())


class FakeTokenizer:
    def encode(self, text, *, add_bos, add_eos):
        assert add_bos and add_eos
        return list(text.encode("utf-8")) + [256, 257]


class FakeModel:
    def __init__(self, context_length=999999):
        self.backbone = SimpleNamespace(config=SimpleNamespace(context_length=context_length))

    def state_dict(self):
        torch = pytest.importorskip("torch")
        return {"weight": torch.tensor([1.0])}


def test_preflight_includes_all_reachable_wrong_action_histories():
    result = model_preflight(FakeModel(), FakeTokenizer())
    assert result["status"] == "PASS"
    # At most 3 non-stop options after the first action.
    assert result["decision_nodes_checked"] > MAX_MODEL_ACTIONS
    assert set(result["max_token_lengths_by_model_action"]) == {"1", "2", "3", "4"}
    assert result["max_token_lengths_by_model_action"]["4"] > (
        result["max_token_lengths_by_model_action"]["1"]
    )
    blocked = model_preflight(FakeModel(context_length=10), FakeTokenizer())
    assert blocked["status"] == "BLOCKED_CONTEXT"
    assert blocked["max_input_length"] > blocked["context_length"]


def test_synthetic_g2_traces_and_failures_are_not_model_evidence(monkeypatch):
    pytest.importorskip("torch")
    import aster.inference.decide as decide

    class FakeReferenceModelPolicy:
        def __init__(self, model, tokenizer, *, candidate_builder, model_id):
            self.builder = candidate_builder
            self.traces = []
            self.delegate = ReferenceReconstructionPolicy()

        def decide(self, state, trajectory, available_actions):
            candidates = self.builder.build(state, trajectory, available_actions)
            selected = self.delegate.decide(state, trajectory, available_actions)
            index = candidates.index(selected)
            self.traces.append(DecisionTrace(
                step=state.step, model_id="synthetic-test",
                candidates=candidates, scores=tuple(float(i == index) for i in range(len(candidates))),
                probabilities=tuple(float(i == index) for i in range(len(candidates))),
                selected_index=index,
            ))
            return selected

    monkeypatch.setattr(decide, "ModelPolicy", FakeReferenceModelPolicy)
    result = run_frozen_model(FakeModel(), FakeTokenizer(), model_id="synthetic-test")
    assert result["G2"] == "DEV_CASE_SUCCESS_ONLY"
    assert result["weights_unchanged"] is True
    assert len(result["decision_traces"]) == 4
    assert result["episode"].task_success

    class FakeBlindModelPolicy(FakeReferenceModelPolicy):
        def decide(self, state, trajectory, available_actions):
            candidates = self.builder.build(state, trajectory, available_actions)
            index = candidates.index(STOP)
            self.traces.append(DecisionTrace(
                step=state.step, model_id="synthetic-blind",
                candidates=candidates, scores=tuple(float(i == index) for i in range(len(candidates))),
                probabilities=tuple(float(i == index) for i in range(len(candidates))),
                selected_index=index,
            ))
            return STOP

    monkeypatch.setattr(decide, "ModelPolicy", FakeBlindModelPolicy)
    failed = run_frozen_model(FakeModel(), FakeTokenizer(), model_id="synthetic-blind")
    assert failed["G2"] == "NOT_SUPPORTED_ON_DEV_CASE"
    assert not failed["episode"].task_success
    assert len(failed["decision_traces"]) == 1

    blocked = run_frozen_model(FakeModel(context_length=10), FakeTokenizer(), model_id="synthetic")
    assert blocked["G2"] == "BLOCKED_CONTEXT"
    assert "trajectory" not in blocked
