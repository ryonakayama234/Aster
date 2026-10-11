"""OBS v0a: isolated, non-mutating read fault and observation-conditioned continuation.

Experiment-only fixture. Never replaces the ACT v0 candidate builder or service.
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
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.runtime.state import RuntimeState
from aster.tools.builtin.memory import MEMORY_GET_SPEC, memory_get
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.tools.schema import ToolExecutionError

PROTOCOL_ID = "obs-read-retry-v0a"
CANDIDATE_BUILDER_ID = "obs-read-retry-v0a-experimental"
SOURCE_BUILDER_ID = "calculate-and-store-v0"
TASK = {"kind": "calculate_and_store", "operation": "add",
        "left": 23, "right": 19, "store_as": "answer"}
START_MEMORY = {"answer": 42}
RETRY = Action.tool("memory.get", key="answer")
STOP = Action.stop("policy_stop")
FIXED_CANDIDATES = (RETRY, STOP)
BRANCHES = ("success", "transient_failure")


class ReadRetryCandidates:
    """Identical choices and order regardless of prior tool outcome."""

    def build(self, state: RuntimeState, trajectory: Trajectory,
              available_actions: tuple[str, ...]) -> tuple[Action, ...]:
        if state.task != TASK or "memory.get" not in available_actions or "stop" not in available_actions:
            raise ValueError("OBS v0a fixture/action contract mismatch")
        return FIXED_CANDIDATES


class ReferenceReadPolicy:
    """Diagnostic oracle only; never described as a learned model."""

    def decide(self, state: RuntimeState, trajectory: Trajectory,
               available_actions: tuple[str, ...]) -> Action:
        ReadRetryCandidates().build(state, trajectory, available_actions)
        if not trajectory.transitions:
            raise ValueError("Scripted prefix missing")
        return STOP if trajectory.last.evaluation.goal_satisfied else RETRY


class ObservationBlindPolicy:
    """Always stop, regardless of Observation; controlled lower baseline."""

    def decide(self, state: RuntimeState, trajectory: Trajectory,
               available_actions: tuple[str, ...]) -> Action:
        ReadRetryCandidates().build(state, trajectory, available_actions)
        return STOP


@dataclass
class Episode:
    branch: str
    context: RuntimeContext
    executor: ToolExecutor
    evaluator: TaskEvaluator
    recorder: TrajectoryRecorder

    @property
    def trajectory(self) -> Trajectory:
        return self.recorder.trajectory()


def _executor(branch: str) -> ToolExecutor:
    calls = 0

    def possibly_fail_once(context: RuntimeContext, arguments: dict):
        nonlocal calls
        calls += 1
        if branch == "transient_failure" and calls == 1:
            raise ToolExecutionError("transient_read_error", "temporary read failure")
        return memory_get(context, arguments)

    registry = ToolRegistry()
    registry.register(MEMORY_GET_SPEC, possibly_fail_once)
    return ToolExecutor(registry)


def _append_action(episode: Episode, action: Action) -> Transition:
    """Record a real tool/stop transition with contiguous steps and truthful eval."""
    history = episode.recorder.trajectory()
    step = len(history)
    state_before = episode.context.snapshot(step)
    available_actions = episode.executor.registry.names() + ("stop",)
    if action.kind == "stop":
        observation = Observation(
            accepted=True, ok=True,
            output={"stopped": True, "reason": action.arguments.get("reason", "policy_stop")},
        )
    else:
        observation = episode.executor.execute(action, episode.context)
    state_after = episode.context.snapshot(step + 1)
    evaluation = episode.evaluator.evaluate(
        episode.context.task, state_after, action, observation, history,
    )
    transition = Transition(
        step=step, state_before=state_before.to_dict(),
        available_actions=available_actions, action=action, observation=observation,
        state_after=state_after.to_dict(), evaluation=evaluation,
    )
    episode.recorder.record(transition)
    return transition


def create_episode(branch: str) -> Episode:
    if branch not in BRANCHES:
        raise ValueError("Unknown OBS v0a branch")
    context = RuntimeContext(task=TASK, memory=START_MEMORY)
    episode = Episode(
        branch=branch, context=context, executor=_executor(branch),
        evaluator=TaskEvaluator(), recorder=TrajectoryRecorder(),
    )
    # Prefix is environmental scaffolding, never credited as a model choice.
    transition = _append_action(episode, RETRY)
    if branch == "success":
        if not transition.observation.ok or not transition.evaluation.goal_satisfied:
            raise AssertionError("Success prefix failed")
    else:
        if transition.observation.ok or transition.evaluation.goal_satisfied:
            raise AssertionError("Fault prefix was unexpectedly successful")
        if transition.observation.error is None or transition.observation.error.code != "transient_read_error":
            raise AssertionError("Fault was not a truthful one-shot read error")
    if episode.context.memory != START_MEMORY:
        raise AssertionError("Scripted prefix mutated memory")
    return episode


def assert_observation_only_pair(success: Episode, failed: Episode) -> None:
    """The *entire model-visible* paired decision differs in Observation only."""
    if (success.branch, failed.branch) != BRANCHES:
        raise AssertionError("Wrong branch order")
    left, right = success.trajectory.last, failed.trajectory.last
    if left is None or right is None or len(success.trajectory) != 1 or len(failed.trajectory) != 1:
        raise AssertionError("A single scripted prefix is required")
    if left.observation == right.observation:
        raise AssertionError("Observations must actually differ")
    if left.state_before != right.state_before or left.state_after != right.state_after:
        raise AssertionError("Paired state differs")
    if left.action != right.action or left.available_actions != right.available_actions:
        raise AssertionError("Paired executed actions differ")
    for candidate in FIXED_CANDIDATES:
        l = loads(serialize_decision_input(success.context.snapshot(1), success.trajectory, candidate))
        r = loads(serialize_decision_input(failed.context.snapshot(1), failed.trajectory, candidate))
        if l["history"][0]["observation"] == r["history"][0]["observation"]:
            raise AssertionError("Serialized observations do not differ")
        l["history"][0]["observation"] = None
        r["history"][0]["observation"] = None
        if l != r:
            raise AssertionError("A model-visible field besides Observation differs")


def continue_episode(episode: Episode, policy: Policy, *, max_continuation: int = 2) -> Trajectory:
    if len(episode.trajectory) != 1:
        raise ValueError("Continuation requires one scripted prefix")
    if max_continuation != 2:
        raise ValueError("OBS v0a fixes exactly two possible continuation steps")
    for _ in range(max_continuation):
        state = episode.context.snapshot(len(episode.trajectory))
        action = policy.decide(
            state, episode.trajectory, episode.executor.registry.names() + ("stop",)
        )
        if action not in FIXED_CANDIDATES:
            raise ValueError("Policy selected outside experiment candidate tuple")
        _append_action(episode, action)
        if action.kind == "stop":
            break
    return episode.trajectory


def run_reference_checks() -> dict:
    """G0 and G1 development fixture; no learned model is invoked."""
    pair = {branch: create_episode(branch) for branch in BRANCHES}
    assert_observation_only_pair(pair["success"], pair["transient_failure"])
    oracle: dict[str, dict] = {}
    blind: dict[str, dict] = {}
    for branch in BRANCHES:
        reference = create_episode(branch)
        traj = continue_episode(reference, ReferenceReadPolicy())
        oracle[branch] = {
            "actions": [t.action.to_dict() for t in traj.transitions],
            "goal_verified": evaluate_episode(traj).goal_verified,
            "task_success": evaluate_episode(traj).task_success,
        }
        baseline = create_episode(branch)
        trace = continue_episode(baseline, ObservationBlindPolicy())
        blind[branch] = {
            "first_continuation_action": trace.transitions[1].action.to_dict(),
            "task_success": evaluate_episode(trace).task_success,
        }
    valid_reference = (
        oracle["success"]["task_success"] and oracle["transient_failure"]["task_success"]
        and len(oracle["success"]["actions"]) == 2
        and len(oracle["transient_failure"]["actions"]) == 3
    )
    if not valid_reference:
        raise AssertionError("G1 reference oracle/verification failed")
    if (blind["success"]["first_continuation_action"]
            != blind["transient_failure"]["first_continuation_action"]):
        raise AssertionError("Observation-blind baseline is not blind")
    return {
        "protocol_id": PROTOCOL_ID, "G0_observation_only": "PASS",
        "G1_reference": "PASS",
        "reference": oracle, "observation_blind": blind,
        "note": "Scripted prefix, single development pair; no model capability claim",
    }


def model_preflight(model, tokenizer) -> dict:
    """Check every model decision input before inference, never silently truncate."""
    inputs = {}
    for branch in BRANCHES:
        episode = create_episode(branch)
        positions = [(branch + "_postprefix", episode)]
        if branch == "transient_failure":
            # Truthful tool retry, solely to preflight the later model input.
            _append_action(episode, RETRY)
            positions.append((branch + "_after_retry", episode))
        for name, point in positions:
            state = point.context.snapshot(len(point.trajectory))
            lengths = [
                len(tokenizer.encode(
                    serialize_decision_input(state, point.trajectory, candidate),
                    add_bos=True, add_eos=True,
                ))
                for candidate in FIXED_CANDIDATES
            ]
            inputs[name] = lengths
    limit = model.backbone.config.context_length
    maximum = max(max(values) for values in inputs.values())
    return {
        "status": "PASS" if maximum <= limit else "BLOCKED_CONTEXT",
        "context_length": limit, "input_lengths": inputs,
        "max_input_length": maximum,
    }


def run_frozen_model(model, tokenizer, *, model_id: str) -> dict:
    """G2 only after G0/G1, with immutable weights and no reference intervention."""
    import torch

    from aster.inference.decide import ModelPolicy

    preflight = model_preflight(model, tokenizer)
    if preflight["status"] != "PASS":
        return {"G2": "BLOCKED_CONTEXT", "preflight": preflight}
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    results = {}
    for branch in BRANCHES:
        episode = create_episode(branch)
        policy = ModelPolicy(
            model, tokenizer,
            candidate_builder=ReadRetryCandidates(), model_id=model_id,
        )
        trajectory = continue_episode(episode, policy)
        assert len(policy.traces) == len(trajectory.transitions) - 1
        assert all(
            trace.selected == transition.action and trace.step == transition.step
            for trace, transition in zip(policy.traces, trajectory.transitions[1:], strict=True)
        )
        results[branch] = {
            "trajectory": trajectory, "decision_traces": policy.traces,
            "episode": evaluate_episode(trajectory),
        }
    if not all(torch.equal(before[key], current)
               for key, current in model.state_dict().items()):
        raise AssertionError("Frozen model weights changed")
    success = results["success"]
    fault = results["transient_failure"]
    g2 = (
        [t.action for t in success["trajectory"].transitions[1:]] == [STOP]
        and [t.action for t in fault["trajectory"].transitions[1:]] == [RETRY, STOP]
        and success["episode"].task_success and success["episode"].goal_verified
        and fault["episode"].task_success and fault["episode"].goal_verified
    )
    return {
        "G2": "DEV_PAIR_SUCCESS_ONLY" if g2 else "NOT_SUPPORTED_ON_DEV_PAIR",
        "preflight": preflight, "branches": results, "weights_unchanged": True,
    }
