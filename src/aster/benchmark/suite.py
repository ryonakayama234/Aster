"""Deterministic synthetic suites for the current Agent Kernel task family."""

from dataclasses import replace

from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import DecisionExample
from aster.records.transition import Action, JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry
from aster.training.decision import examples_from_teacher_trajectory


def build_calculate_and_store_suite() -> BenchmarkSuite:
    """Build the v0 split used to distinguish memorization from simple transfer.

    The suite is deliberately small and synthetic. It validates benchmark plumbing and
    exposes named distribution shifts; it is not evidence of broad agent capability.
    """
    cases: list[BenchmarkCase] = []
    specs = (
        ("train", "in_distribution", "train-add-small", _task("add", 2, 3, "total"), {}, None),
        (
            "train",
            "in_distribution",
            "train-subtract-small",
            _task("subtract", 9, 4, "result"),
            {},
            None,
        ),
        (
            "calibration",
            "in_distribution",
            "cal-add-small",
            _task("add", 11, 7, "total"),
            {},
            None,
        ),
        (
            "calibration",
            "in_distribution",
            "cal-subtract-small",
            _task("subtract", 15, 6, "result"),
            {},
            None,
        ),
        ("dev", "interpolation", "dev-add", _task("add", 8, 13, "total"), {}, None),
        (
            "dev",
            "interpolation",
            "dev-subtract",
            _task("subtract", 17, 5, "result"),
            {},
            None,
        ),
        (
            "test",
            "range_shift",
            "test-range",
            _task("add", 142, 387, "total"),
            {},
            None,
        ),
        (
            "test",
            "composition",
            "test-composition",
            _task("multiply", 6, 7, "product"),
            {},
            None,
        ),
        (
            "test",
            "key_shift",
            "test-key",
            _task("add", 4, 5, "unseen_key"),
            {},
            None,
        ),
        (
            "test",
            "history_shift",
            "test-history",
            _task("add", 3, 9, "total"),
            {"noise": 99, "status": "warm"},
            None,
        ),
        (
            "test",
            "tool_failure",
            "test-failure",
            _task("divide", 1, 0, "total"),
            {},
            None,
        ),
        (
            "test",
            "candidate_set_shift",
            "test-candidate-set",
            _task("add", 6, 8, "total"),
            {},
            "distractor",
        ),
        (
            "test",
            "candidate_permutation",
            "test-permutation",
            _task("subtract", 12, 5, "result"),
            {},
            "reverse",
        ),
    )

    for split, slice_name, group, task, memory, transform in specs:
        examples = _teacher_examples(task, memory)
        for step, example in enumerate(examples):
            if transform == "distractor":
                example = _add_distractor(example)
            elif transform == "reverse":
                example = _reverse_candidates(example)
            cases.append(
                BenchmarkCase(
                    case_id=f"{group}/step-{step}",
                    split=split,
                    slice_name=slice_name,
                    leakage_group=group,
                    example=example,
                )
            )

    return BenchmarkSuite("calculate-and-store-v0", tuple(cases))


def _task(operation: str, left: int, right: int, store_as: str) -> dict[str, JsonValue]:
    return {
        "kind": "calculate_and_store",
        "operation": operation,
        "left": left,
        "right": right,
        "store_as": store_as,
    }


def _executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def _teacher_examples(
    task: dict[str, JsonValue], memory: dict[str, JsonValue]
) -> list[DecisionExample]:
    trajectory = run_loop(
        policy=RuleBasedPolicy(),
        executor=_executor(),
        evaluator=TaskEvaluator(),
        context=RuntimeContext(task=task, memory=memory),
    )
    return examples_from_teacher_trajectory(trajectory, teacher="rule-v0:benchmark")


def _add_distractor(example: DecisionExample) -> DecisionExample:
    distractor = Action.stop("benchmark_distractor")
    if distractor in example.candidates:
        return example
    return replace(example, candidates=example.candidates + (distractor,))


def _reverse_candidates(example: DecisionExample) -> DecisionExample:
    candidates = tuple(reversed(example.candidates))
    return replace(example, candidates=candidates, target_index=candidates.index(example.target))
