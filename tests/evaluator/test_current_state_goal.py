"""Executable contract boundaries; these do not score a learned model."""

import json
from dataclasses import replace
from itertools import product

import pytest

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.evaluator.episode import evaluate_episode, write_episode_evaluation
from aster.evaluator.goal_contract import CURRENT_GOAL_V1, EVER_GOAL_V0
from aster.evaluator.result import EvaluationResult
from aster.evaluator.verifier import TaskEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry

TASK = dict(kind="calculate_and_store", operation="add", left=2, right=3, store_as="total")


class Session:
    def __init__(self, contract=CURRENT_GOAL_V1):
        self.context = RuntimeContext(task=TASK)
        self.recorder = TrajectoryRecorder()
        self.evaluator = TaskEvaluator(goal_contract=contract)
        self.policy = RuleBasedPolicy(goal_contract=contract)
        self.candidates = CalculateAndStoreCandidates(goal_contract=contract)
        registry = ToolRegistry()
        registry.register(CALCULATOR_SPEC, calculator)
        registry.register(MEMORY_PUT_SPEC, memory_put)
        registry.register(MEMORY_GET_SPEC, memory_get)
        self.executor = ToolExecutor(registry)
        self.available = registry.names() + ("stop",)

    def execute(self, action):
        history = self.recorder.trajectory()
        before = self.context.snapshot(len(history))
        observation = (Observation(True, True, {"stopped": True}) if action.kind == "stop"
                       else self.executor.execute(action, self.context))
        after = self.context.snapshot(len(history) + 1)
        transition = Transition(
            len(history), before.to_dict(), self.available, action, observation,
            after.to_dict(), self.evaluator.evaluate(TASK, after, action, observation, history),
        )
        self.recorder.record(transition)
        return transition

    def teacher_step(self):
        history = self.recorder.trajectory()
        state = self.context.snapshot(len(history))
        action = self.policy.decide(state, history, self.available)
        assert action in self.candidates.build(state, history, self.available)
        return self.execute(action)

    def verified_prefix(self):
        for _ in range(3):
            self.teacher_step()
        assert self.recorder.trajectory().last.evaluation.goal_satisfied


def test_normal_v1_loop_and_persisted_contract(tmp_path):
    session = Session()
    trajectory = run_loop(policy=session.policy, executor=session.executor,
                          evaluator=session.evaluator, context=session.context,
                          recorder=session.recorder)
    assert [t.action.name or t.action.kind for t in trajectory.transitions] == [
        "calculator", "memory.put", "memory.get", "stop"
    ]
    assert trajectory.successful
    path = tmp_path / "trajectory.jsonl"
    session.recorder.write_jsonl(path)
    assert TrajectoryRecorder.read_jsonl(path) == trajectory
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["schema_version"] == "aster-transition-1"
    evaluation_path = tmp_path / "evaluation.json"
    write_episode_evaluation(evaluation_path, path)
    result = json.loads(evaluation_path.read_text())
    assert result["schema_version"] == "aster-episode-evaluation-1"
    assert result["goal_contract"] == CURRENT_GOAL_V1


def test_wrong_overwrite_requires_repair_and_fresh_get(tmp_path):
    session = Session()
    session.verified_prefix()
    overwritten = session.execute(Action.tool("memory.put", key="total", value=-999))
    assert not overwritten.evaluation.goal_satisfied
    repaired = session.teacher_step()
    assert repaired.action == Action.tool("memory.put", key="total", value=5)
    assert not repaired.evaluation.goal_satisfied
    checked = session.teacher_step()
    assert checked.action == Action.tool("memory.get", key="total")
    assert checked.evaluation.goal_satisfied
    assert session.teacher_step().action == Action.stop("goal_verified")
    trajectory = session.recorder.trajectory()
    assert trajectory.successful
    path = tmp_path / "recovery.jsonl"
    session.recorder.write_jsonl(path)
    assert TrajectoryRecorder.read_jsonl(path) == trajectory


@pytest.mark.parametrize("value", [-999, 5])
def test_stop_after_any_target_write_is_unsuccessful(value):
    session = Session()
    session.verified_prefix()
    session.execute(Action.tool("memory.put", key="total", value=value))
    session.execute(Action.stop("goal_verified"))
    result = evaluate_episode(session.recorder.trajectory())
    assert not result.task_success
    assert result.goal_verified  # Historical achievement is still recorded.
    assert result.first_goal_verified_step == 2


def test_same_value_write_requires_get_but_unrelated_write_does_not():
    session = Session()
    session.verified_prefix()
    session.execute(Action.tool("memory.put", key="total", value=5))
    assert session.teacher_step().action == Action.tool("memory.get", key="total")
    unrelated = session.execute(Action.tool("memory.put", key="other", value=-999))
    assert unrelated.evaluation.goal_satisfied
    assert session.teacher_step().action == Action.stop("goal_verified")


def test_failed_put_does_not_invalidate_confirmed_value():
    session = Session()
    session.verified_prefix()
    failed = session.execute(Action.tool("memory.put", key="total"))
    assert not failed.observation.ok
    assert failed.evaluation.goal_satisfied
    assert session.context.memory["total"] == 5


def test_prepopulated_memory_requires_get():
    session = Session()
    session.context.memory["total"] = 5
    unconfirmed = session.execute(Action.stop())
    assert not unconfirmed.evaluation.goal_satisfied
    assert session.execute(Action.tool("memory.get", key="total")).evaluation.goal_satisfied


def test_teacher_ignores_calculator_result_for_different_arguments():
    session = Session()
    session.execute(Action.tool("calculator", operation="add", left=10, right=20))
    assert session.teacher_step().action == Action.tool("calculator", operation="add", left=2, right=3)


def test_contract_and_schema_mismatch_fail_closed():
    session = Session()
    transition = session.teacher_step()
    old = replace(transition, evaluation=replace(transition.evaluation, goal_contract=EVER_GOAL_V0))
    with pytest.raises(ValueError, match="mix goal contracts"):
        Trajectory((transition, replace(old, step=1)))
    with pytest.raises(ValueError, match="history goal contracts"):
        TaskEvaluator().evaluate(TASK, session.context.snapshot(1), transition.action,
                                 transition.observation, session.recorder.trajectory())
    payload = transition.to_dict()
    payload["schema_version"] = "aster-transition-0"
    with pytest.raises(ValueError, match="schema and goal contract"):
        Transition.from_dict(payload)
    with pytest.raises(ValueError, match="Unsupported goal contract"):
        EvaluationResult(True, True, False, goal_contract="future-unknown")


def test_v0_overwrite_semantics_and_json_remain_unchanged():
    session = Session(EVER_GOAL_V0)
    session.verified_prefix()
    assert session.execute(Action.tool("memory.put", key="total", value=-999)).evaluation.goal_satisfied
    stopped = session.teacher_step()
    assert stopped.action == Action.stop("goal_verified")
    assert session.recorder.trajectory().successful
    assert stopped.to_dict()["schema_version"] == "aster-transition-0"
    assert "goal_contract" not in stopped.evaluation.to_dict()
    assert Transition.from_dict(stopped.to_dict()) == stopped


def test_short_executable_histories_match_independent_state_machine():
    # Exhaust 1,364 paths of length 1..5. Reference tracks state and confirmation,
    # without consulting policy, goal flags or evaluator history helpers.
    actions = (
        Action.tool("memory.put", key="total", value=5),
        Action.tool("memory.put", key="total", value=-999),
        Action.tool("memory.get", key="total"),
        Action.tool("memory.put", key="other", value=5),
    )
    for length in range(1, 6):
        for path in product(actions, repeat=length):
            session = Session()
            reference_value = None
            reference_confirmed = False
            for action in path:
                if action.name == "memory.put" and action.arguments["key"] == "total":
                    reference_value = action.arguments["value"]
                    reference_confirmed = False
                elif action.name == "memory.get" and reference_value is not None:
                    reference_confirmed = reference_value == 5
                transition = session.execute(action)
                assert transition.evaluation.goal_satisfied == (
                    reference_value == 5 and reference_confirmed
                ), path
