"""Executed dev counterexamples and target-free input projection; no model training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry

SCHEMA = "aster-state-representation-preflight-0"


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def execute(context, recorder, tool_executor, action) -> Transition:
    """Execute with absolute prefix steps; run_loop currently starts its steps at zero."""
    history = recorder.trajectory()
    step = len(history.transitions)
    before = context.snapshot(step)
    observation = (Observation(True, True, {"stopped": True, "reason": action.arguments.get("reason", "policy_stop")})
                   if action.kind == "stop" else tool_executor.execute(action, context))
    after = context.snapshot(step + 1)
    evaluation = TaskEvaluator().evaluate(context.task, after, action, observation, history)
    transition = Transition(step, before.to_dict(), tool_executor.registry.names() + ("stop",),
                            action, observation, after.to_dict(), evaluation)
    recorder.record(transition)
    return transition


def make_example(task, phase: int, delay: int) -> DecisionExample:
    if phase not in range(4) or delay not in range(3):
        raise ValueError("phase must be 0..3; delay must be 0..2")
    context, recorder, tools = RuntimeContext(task), TrajectoryRecorder(), executor()
    available = tools.registry.names() + ("stop",)
    teacher = RuleBasedPolicy()
    for _ in range(phase):
        action = teacher.decide(context.snapshot(len(recorder.trajectory().transitions)),
                                recorder.trajectory(), available)
        execute(context, recorder, tools, action)
    for action in (Action.tool("memory.put", key="representation_noise", value=99),
                   Action.tool("memory.get", key="representation_noise"))[:delay]:
        execute(context, recorder, tools, action)
    history = recorder.trajectory()
    state = context.snapshot(len(history.transitions))
    candidates = CalculateAndStoreCandidates().build(state, history, available)
    target = teacher.decide(state, history, available)
    return DecisionExample(state, history, candidates, candidates.index(target), teacher="rule-v0:representation-dev")


def build_suite() -> tuple[BenchmarkSuite, list[dict]]:
    """48 dev probes, two parent groups; do not construct or score sealed tests."""
    cases, pairs = [], []
    specs = (("add", "anchor", 2, 3, "total"), ("subtract", "anchor", 9, 4, "result"),
             ("add", "joint", 6, 10, "diagnostic_total"),
             ("subtract", "joint", 20, 7, "diagnostic_result"))
    for operation, variant, left, right, key in specs:
        task = dict(kind="calculate_and_store", operation=operation, left=left, right=right, store_as=key)
        def case_id(phase, delay):
            return f"{operation}/{variant}/phase-{phase}/delay-{delay}"
        for phase in range(4):
            for delay in range(3):
                cases.append(BenchmarkCase(case_id(phase, delay), "dev", variant,
                    f"train-{operation}-small-descendants", make_example(task, phase, delay)))
            for delay in (1, 2):
                pairs.append(dict(relation="same_target_different_step", left=case_id(phase, 0), right=case_id(phase, delay)))
        for p in range(4):
            for q in range(p + 1, 4):
                for d in range(3):
                    e = p + d - q
                    if e in range(3):
                        pairs.append(dict(relation="same_step_different_target", left=case_id(p, d), right=case_id(q, e)))
    return BenchmarkSuite("calculate-store-state-representation-dev-v0", tuple(cases)), pairs


def compact_input(state, history, candidate: Action) -> str:
    """Project observable task facts and relevant raw events, never teacher labels.

    This is domain-specific engineering, not learned perception or generic memory.
    Keep all calculator events and target-key memory events, plus the last status.
    No phase, step, evaluation, target_index, teacher or expected-answer field.
    """
    key = state.task.get("store_as")
    events = []
    for transition in history.transitions:
        action = transition.action
        if action.name == "calculator" or (action.name in ("memory.put", "memory.get") and action.arguments.get("key") == key):
            events.append([action.to_dict(), transition.observation.to_dict()])
    return canonical({"schema": "aster-decision-compact-input-0", "task": state.task,
        "memory": {"present": key in state.memory, "value": state.memory.get(key) if isinstance(key, str) else None},
        "events": events, "last_failure": history.last.observation.to_dict() if history.last and not history.last.observation.ok else None,
        "candidate": candidate.to_dict()})


def replay(example: DecisionExample):
    """Reject tampered state, observations, evaluation, candidates or teacher labels."""
    context, recorder, tools = RuntimeContext(example.state.task), TrajectoryRecorder(), executor()
    for saved in example.trajectory.transitions:
        actual = execute(context, recorder, tools, saved.action)
        if actual.to_dict() != saved.to_dict():
            raise ValueError("Prefix replay mismatch")
    state = context.snapshot(len(recorder.trajectory().transitions))
    available = tools.registry.names() + ("stop",)
    if state != example.state:
        raise ValueError("State replay mismatch")
    if CalculateAndStoreCandidates().build(state, recorder.trajectory(), available) != example.candidates:
        raise ValueError("Candidate replay mismatch")
    if RuleBasedPolicy().decide(state, recorder.trajectory(), available) != example.target:
        raise ValueError("Teacher replay mismatch")
    return context, recorder, tools


def continue_rule(example: DecisionExample) -> dict:
    context, recorder, tools = replay(example)
    prefix_length = len(recorder.trajectory().transitions)
    for _ in range(8):
        history = recorder.trajectory()
        action = RuleBasedPolicy().decide(context.snapshot(len(history.transitions)), history,
                                          tools.registry.names() + ("stop",))
        last = execute(context, recorder, tools, action)
        if last.evaluation.terminal:
            break
    history = recorder.trajectory()
    last = history.last
    return {"prefix_length": prefix_length, "success": bool(last and last.evaluation.terminal and last.evaluation.goal_satisfied),
            "trajectory": [t.to_dict() for t in history.transitions]}


def audit_suite(suite: BenchmarkSuite, pairs: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {case.case_id: case for case in suite.cases}
    if any(case.split != "dev" for case in suite.cases):
        raise ValueError("Preflight accepts dev only")
    rows, projections, candidate_groups = [], defaultdict(set), defaultdict(list)
    # Labels follow the original eight training decisions; no dev fitting.
    step_labels = {0: "calculator", 1: "memory.put", 2: "memory.get", 3: "stop:goal_verified"}
    def label(action):
        return action.name if action.kind == "tool" else "stop:" + str(action.arguments.get("reason"))
    for case in suite.cases:
        example = case.example
        continuation = continue_rule(example)
        if not continuation["success"]:
            raise ValueError("Rule continuation failed")
        compact = [compact_input(example.state, example.trajectory, a) for a in example.candidates]
        raw = [serialize_decision_input(example.state, example.trajectory, a) for a in example.candidates]
        projections[canonical(compact)].add(canonical(example.target.to_dict()))
        candidate_groups[canonical([a.to_dict() for a in example.candidates])].append(case)
        rows.append({"case_id": case.case_id, "case_sha256": digest(case.to_dict()),
            "step_only_correct": any(a == example.target and label(a) == step_labels.get(example.state.step) for a in example.candidates),
            "first_candidate_correct": example.candidates[0] == example.target,
            "raw_inputs": raw, "compact_inputs": compact,
            "raw_bytes": [len(s.encode()) for s in raw], "compact_bytes": [len(s.encode()) for s in compact],
            "rule_continuation": continuation})
    for pair in pairs:
        left, right = (by_id[pair[name]].example for name in ("left", "right"))
        if pair["relation"] == "same_step_different_target":
            valid = left.state.step == right.state.step and left.target != right.target
        elif pair["relation"] == "same_target_different_step":
            valid = left.state.step != right.state.step and left.target == right.target
            valid = valid and [compact_input(left.state, left.trajectory, a) for a in left.candidates] == [compact_input(right.state, right.trajectory, a) for a in right.candidates]
        else:
            raise ValueError("Unknown pair relation")
        if not valid:
            raise ValueError("Counterexample relation failed")
    collisions = sum(len(targets) > 1 for targets in projections.values())
    if collisions:
        raise ValueError("Compact input loses necessary target information")
    ambiguous = [group for group in candidate_groups.values() if len({canonical(c.example.target.to_dict()) for c in group}) > 1]
    raw_bytes = sum(sum(r["raw_bytes"]) for r in rows)
    compact_bytes = sum(sum(r["compact_bytes"]) for r in rows)
    return {"schema_version": SCHEMA, "suite_sha256": digest(suite.to_dict()), "pairs_sha256": digest(pairs),
        "cases": len(rows), "parent_groups": len({c.leakage_group for c in suite.cases}), "independent_holdout": False,
        "relations": {name: sum(p["relation"] == name for p in pairs) for name in ("same_step_different_target", "same_target_different_step")},
        "replay_verified": len(rows), "rule_continuation_success": len(rows), "candidate_coverage": len(rows),
        "compact_target_collisions": collisions, "candidate_only_ambiguous_sets": len(ambiguous),
        "candidate_only_ambiguous_cases": sum(map(len, ambiguous)),
        "candidate_only_in_sample_oracle_upper_correct": sum(max(Counter(canonical(c.example.target.to_dict()) for c in group).values()) for group in candidate_groups.values()),
        "step_only_correct": sum(r["step_only_correct"] for r in rows),
        "first_candidate_correct": sum(r["first_candidate_correct"] for r in rows),
        "raw_bytes": raw_bytes, "compact_bytes": compact_bytes, "compact_to_raw_ratio": compact_bytes / raw_bytes,
        "model_scored_cases": 0, "test_scored_cases": 0, "scope": "development preflight; no learned capability measured"}, rows


def run_preflight(root: str | Path, *, source_git_sha: str | None = None, source_provenance: dict | None = None) -> Path:
    suite, pairs = build_suite()
    run = RunLog(Path(root), "state_representation_preflight", {"source_git_sha": source_git_sha,
                 "source_provenance": source_provenance, "suite_sha256": digest(suite.to_dict()), "model_update": False}, producer="evaluator")
    try:
        report, rows = audit_suite(suite, pairs)
        report.update(source_git_sha=source_git_sha, source_provenance=source_provenance)
        for name, value in (("suite.json", suite.to_dict()), ("pairs.json", pairs), ("preflight.json", report)):
            (run.path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (run.path / "cases.jsonl").write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
        run.finish("completed", report="preflight.json", model_scored_cases=0, test_scored_cases=0)
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    return run.path
