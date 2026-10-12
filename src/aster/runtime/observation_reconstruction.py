"""OBS v0c: truthful missing-key reconstruction, isolated from production ACT.

The first memory.get is scripted, never a model decision. All continuation
actions are selected from a versioned experimental candidate builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from json import loads

from aster.agent.policy import Policy
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, JsonValue, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.runtime.state import RuntimeState
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import (
    MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put,
)
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry

PROTOCOL_ID = "obs-missing-key-reconstruction-v0c"
CANDIDATE_BUILDER_ID = "obs-missing-key-reconstruction-v0c-experimental"
SOURCE_BUILDER_ID = "calculate-and-store-v0"
PINNED_ARTIFACT_ID = (
    "decision_model:0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca"
)
MAX_MODEL_ACTIONS = 4
TASK: dict[str, JsonValue] = {
    "kind": "calculate_and_store", "operation": "add",
    "left": 23, "right": 19, "store_as": "answer",
}
PREFIX = Action.tool("memory.get", key="answer")
CALCULATE = Action.tool("calculator", operation="add", left=23, right=19)
STOP = Action.stop("policy_stop")
AVAILABLE = ("calculator", "memory.put", "memory.get", "stop")


class _Missing:
    pass


_MISSING = _Missing()


def _calculation(trajectory: Trajectory) -> JsonValue | _Missing:
    """Only a real, successful calculator observation supplies a put value."""
    for transition in reversed(trajectory.transitions):
        if transition.action == CALCULATE and transition.observation.ok:
            return transition.observation.output
    return _MISSING


def _has_successful_put(trajectory: Trajectory, value: JsonValue) -> bool:
    return any(
        transition.action == Action.tool("memory.put", key="answer", value=value)
        and transition.observation.ok
        for transition in trajectory.transitions
    )


def _has_verified_read(trajectory: Trajectory, value: JsonValue) -> bool:
    return any(
        transition.action == PREFIX
        and transition.observation.ok
        and isinstance(transition.observation.output, dict)
        and transition.observation.output.get("key") == "answer"
        and transition.observation.output.get("value") == value
        for transition in trajectory.transitions
    )


class MissingKeyCandidates:
    """Offer recovery and erroneous alternatives, without using evaluator labels."""

    def build(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> tuple[Action, ...]:
        if state.task != TASK or not set(AVAILABLE).issubset(available_actions):
            raise ValueError("OBS v0c task/tool contract mismatch")
        if not trajectory.transitions or trajectory.transitions[0].action != PREFIX:
            raise ValueError("OBS v0c requires a recorded scripted prefix")
        candidates = [CALCULATE]
        calculated = _calculation(trajectory)
        if not isinstance(calculated, _Missing):
            candidates.append(Action.tool("memory.put", key="answer", value=calculated))
        candidates.extend((PREFIX, STOP))
        return tuple(candidates)


class ReferenceReconstructionPolicy:
    """Tool-grounded reference, NOT a learned policy or model fallback."""

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        candidates = MissingKeyCandidates().build(state, trajectory, available_actions)
        value = _calculation(trajectory)
        if isinstance(value, _Missing):
            choice = CALCULATE
        elif not _has_successful_put(trajectory, value):
            choice = Action.tool("memory.put", key="answer", value=value)
        elif not _has_verified_read(trajectory, value):
            choice = PREFIX
        else:
            choice = STOP
        if choice not in candidates:
            raise AssertionError("Reference action missing from experimental candidates")
        return choice


class BlindStopPolicy:
    """Predeclared observation-independent negative control."""

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        candidates = MissingKeyCandidates().build(state, trajectory, available_actions)
        if STOP not in candidates:
            raise AssertionError("Blind action unavailable")
        return STOP


@dataclass
class Episode:
    context: RuntimeContext
    executor: ToolExecutor
    evaluator: TaskEvaluator
    recorder: TrajectoryRecorder

    @property
    def trajectory(self) -> Trajectory:
        return self.recorder.trajectory()


def _append_action(episode: Episode, action: Action) -> Transition:
    history = episode.trajectory
    step = len(history)
    before = episode.context.snapshot(step)
    if action.kind == "stop":
        observation = Observation(
            accepted=True, ok=True,
            output={"stopped": True, "reason": action.arguments.get("reason", "policy_stop")},
        )
    else:
        observation = episode.executor.execute(action, episode.context)
    after = episode.context.snapshot(step + 1)
    evaluation = episode.evaluator.evaluate(
        episode.context.task, after, action, observation, history,
    )
    transition = Transition(
        step=step, state_before=before.to_dict(),
        available_actions=AVAILABLE, action=action, observation=observation,
        state_after=after.to_dict(), evaluation=evaluation,
    )
    episode.recorder.record(transition)
    return transition


def create_episode() -> Episode:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    episode = Episode(
        context=RuntimeContext(task=TASK, memory={}),
        executor=ToolExecutor(registry),
        evaluator=TaskEvaluator(),
        recorder=TrajectoryRecorder(),
    )
    prefix = _append_action(episode, PREFIX)
    if (
        prefix.observation.ok or not prefix.observation.accepted
        or prefix.observation.error is None
        or prefix.observation.error.code != "memory_key_not_found"
        or prefix.evaluation.goal_satisfied or episode.context.memory
    ):
        raise AssertionError("Prefix is not a truthful, non-mutating missing-key read")
    return episode


def continue_episode(episode: Episode, policy: Policy, *, max_actions: int = 4) -> Trajectory:
    if len(episode.trajectory) != 1 or episode.trajectory.transitions[0].action != PREFIX:
        raise ValueError("Continuation requires exactly one scripted prefix")
    if max_actions != MAX_MODEL_ACTIONS:
        raise ValueError("OBS v0c model action budget must remain fixed")
    builder = MissingKeyCandidates()
    for _ in range(max_actions):
        state = episode.context.snapshot(len(episode.trajectory))
        history = episode.trajectory
        candidates = builder.build(state, history, AVAILABLE)
        action = policy.decide(state, history, AVAILABLE)
        if action not in candidates:
            raise ValueError("Policy selected an action outside the fixed builder")
        _append_action(episode, action)
        if action.kind == "stop":
            break
    return episode.trajectory


def expected_actions_from_reference() -> tuple[Action, ...]:
    episode = create_episode()
    trajectory = continue_episode(episode, ReferenceReconstructionPolicy())
    return tuple(transition.action for transition in trajectory.transitions[1:])


def run_reference_checks() -> dict:
    """G0/G1 synthetic plumbing evidence; never evidence about frozen weights."""
    episode = create_episode()
    first_candidates = MissingKeyCandidates().build(
        episode.context.snapshot(1), episode.trajectory, AVAILABLE,
    )
    if first_candidates != (CALCULATE, PREFIX, STOP):
        raise AssertionError("Unexpected initial candidate set")
    for candidate in first_candidates:
        payload = loads(
            serialize_decision_input(episode.context.snapshot(1), episode.trajectory, candidate)
        )
        if (
            payload["state"]["memory"] != {}
            or payload["history"][0]["observation"]["error"]["code"] != "memory_key_not_found"
            or set(payload["history"][0]) != {"action", "observation"}
            or payload["candidate"] != candidate.to_dict()
        ):
            raise AssertionError("Decision input leaked labels or misrepresented the prefix")

    reference = create_episode()
    ref_traj = continue_episode(reference, ReferenceReconstructionPolicy())
    ref_eval = evaluate_episode(ref_traj)
    actions = [transition.action for transition in ref_traj.transitions]
    expected = [
        PREFIX, CALCULATE, Action.tool("memory.put", key="answer", value=42), PREFIX, STOP,
    ]
    if (
        actions != expected or len(actions) != 5
        or not ref_eval.task_success or not ref_eval.goal_verified
        or not ref_eval.terminal_reached or ref_eval.rejected_actions != 0
        or ref_eval.failed_executions != 1 or reference.context.memory != {"answer": 42}
        or not ref_traj.transitions[1].observation.ok
        or ref_traj.transitions[1].observation.output != 42
        or not ref_traj.transitions[3].observation.ok
    ):
        raise AssertionError("G1 real-tool reference reconstruction failed")

    blind = create_episode()
    blind_traj = continue_episode(blind, BlindStopPolicy())
    blind_eval = evaluate_episode(blind_traj)
    if blind_eval.task_success or blind_eval.goal_verified or len(blind_traj) != 2:
        raise AssertionError("Observation-blind stop unexpectedly succeeded")
    return {
        "protocol_id": PROTOCOL_ID,
        "G0_structure": "PASS",
        "G1_reference": "PASS",
        "reference_actions": [action.to_dict() for action in actions],
        "reference_task_success": ref_eval.task_success,
        "blind_actions": [step.action.to_dict() for step in blind_traj.transitions],
        "blind_task_success": blind_eval.task_success,
        "note": "One scripted-prefix development case; no learned-model claim",
    }


def model_preflight(model, tokenizer) -> dict:
    """Encode EVERY reachable decision node within the fixed four-action horizon.

    This includes erroneous continuations (repeated reads/calculations/writes).
    Stops terminate a path, so they cannot generate new decision nodes.
    All nodes are replayed through the unmodified real ToolExecutor.
    """
    limit = model.backbone.config.context_length
    builder = MissingKeyCandidates()
    frontier: list[tuple[Action, ...]] = [()]
    max_length = 0
    nodes = 0
    per_depth: dict[str, int] = {}
    for depth in range(MAX_MODEL_ACTIONS):
        next_frontier: list[tuple[Action, ...]] = []
        depth_max = 0
        for path in frontier:
            episode = create_episode()
            for action in path:
                _append_action(episode, action)
            state = episode.context.snapshot(len(episode.trajectory))
            candidates = builder.build(state, episode.trajectory, AVAILABLE)
            for candidate in candidates:
                serialized = serialize_decision_input(state, episode.trajectory, candidate)
                length = len(tokenizer.encode(serialized, add_bos=True, add_eos=True))
                depth_max = max(depth_max, length)
                max_length = max(max_length, length)
            nodes += 1
            if depth + 1 < MAX_MODEL_ACTIONS:
                next_frontier.extend(path + (candidate,) for candidate in candidates
                                     if candidate.kind != "stop")
        per_depth[str(depth + 1)] = depth_max
        frontier = next_frontier
    return {
        "status": "PASS" if max_length <= limit else "BLOCKED_CONTEXT",
        "context_length": limit,
        "max_input_length": max_length,
        "decision_nodes_checked": nodes,
        "max_token_lengths_by_model_action": per_depth,
    }


def run_frozen_model(model, tokenizer, *, model_id: str) -> dict:
    """G2 diagnostic adapter: strictly no training, teacher, or fallback."""
    import torch
    from aster.inference.decide import ModelPolicy

    run_reference_checks()
    preflight = model_preflight(model, tokenizer)
    if preflight["status"] != "PASS":
        return {"G2": "BLOCKED_CONTEXT", "preflight": preflight}

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    episode = create_episode()
    policy = ModelPolicy(
        model, tokenizer, candidate_builder=MissingKeyCandidates(), model_id=model_id,
    )
    trajectory = continue_episode(episode, policy)
    model_transitions = trajectory.transitions[1:]
    if (
        len(policy.traces) != len(model_transitions)
        or any(trace.step != transition.step or trace.selected != transition.action
               for trace, transition in zip(policy.traces, model_transitions, strict=True))
    ):
        raise AssertionError("Model DecisionTrace does not match executed actions")
    if any(not torch.equal(before[key], value)
           for key, value in model.state_dict().items()):
        raise AssertionError("Frozen DecisionModel weights changed")

    evaluation = evaluate_episode(trajectory)
    full_sequence = tuple(t.action for t in model_transitions)
    passed = (
        full_sequence == expected_actions_from_reference()
        and len(trajectory) == 5
        and evaluation.task_success and evaluation.goal_verified
        and evaluation.terminal_reached and evaluation.rejected_actions == 0
        and evaluation.failed_executions == 1
    )
    return {
        "G2": "DEV_CASE_SUCCESS_ONLY" if passed else "NOT_SUPPORTED_ON_DEV_CASE",
        "preflight": preflight,
        "trajectory": trajectory,
        "decision_traces": policy.traces,
        "episode": evaluation,
        "weights_unchanged": True,
        "selected_recovery_actions": [t.action.to_dict() for t in model_transitions],
    }
