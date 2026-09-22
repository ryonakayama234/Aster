from aster.agent.policy import RuleBasedPolicy
from aster.evaluator.verifier import TaskEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.transition import Action
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def test_executor_rejects_unknown_tool_and_invalid_arguments():
    executor = build_executor()
    context = RuntimeContext(task={})

    unknown = executor.execute(Action.tool("missing"), context)
    assert not unknown.accepted
    assert not unknown.ok
    assert unknown.error is not None
    assert unknown.error.code == "tool_not_found"

    invalid = executor.execute(Action.tool("calculator", operation="add", left=1), context)
    assert not invalid.accepted
    assert not invalid.ok
    assert invalid.error is not None
    assert invalid.error.code == "invalid_arguments"


def test_agent_kernel_records_calculate_store_verify_stop(tmp_path):
    task = {
        "kind": "calculate_and_store",
        "operation": "add",
        "left": 40,
        "right": 2,
        "store_as": "total",
    }
    context = RuntimeContext(task=task)
    recorder = TrajectoryRecorder()

    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=context,
        recorder=recorder,
    )

    assert [transition.action.name or transition.action.kind for transition in trajectory.transitions] == [
        "calculator",
        "memory.put",
        "memory.get",
        "stop",
    ]
    assert trajectory.transitions[0].observation.output == 42
    assert trajectory.transitions[0].state_after["memory"] == {}
    assert trajectory.transitions[1].state_before["memory"] == {}
    assert trajectory.transitions[1].state_after["memory"] == {"total": 42}
    assert trajectory.transitions[2].observation.output == {"key": "total", "value": 42}
    assert not trajectory.transitions[1].evaluation.goal_satisfied
    assert trajectory.transitions[2].evaluation.goal_satisfied
    assert trajectory.stopped
    assert trajectory.successful
    assert context.memory == {"total": 42}

    for transition in trajectory.transitions:
        assert transition.available_actions == ("calculator", "memory.put", "memory.get", "stop")

    path = tmp_path / "trajectory.jsonl"
    recorder.write_jsonl(path)
    restored = TrajectoryRecorder.read_jsonl(path)
    assert restored == trajectory


def test_tool_failure_becomes_observation_then_policy_stops():
    task = {
        "kind": "calculate_and_store",
        "operation": "divide",
        "left": 1,
        "right": 0,
        "store_as": "total",
    }
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=build_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task),
    )

    assert len(trajectory.transitions) == 2
    failed = trajectory.transitions[0]
    assert failed.action.name == "calculator"
    assert failed.observation.accepted
    assert not failed.observation.ok
    assert failed.observation.error is not None
    assert failed.observation.error.code == "division_by_zero"
    assert trajectory.transitions[1].action == Action.stop("tool_failure")
    assert not trajectory.successful
