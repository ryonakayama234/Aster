"""Read-only state/shortcut diagnostics for saved train-fit checkpoints.

These dev probes descend from the two seen training tasks. They diagnose specific
changes, not independent unseen task families or a new training corpus.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.suite import build_calculate_and_store_suite, _executor, _task
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import ModelPolicy, score_candidates
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.trajectory import Trajectory
from aster.records.transition import Action, Transition
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.training.decision_fit import load_decision_fit_checkpoint


SCHEMA = "aster-decision-state-diagnostics-0"


def action_label(action: Action) -> str:
    """Action type, excluding numeric/key arguments; stop reasons remain distinct."""
    return str(action.name) if action.kind == "tool" else "stop:" + str(action.arguments.get("reason"))


def _task_specs():
    # Derived numeric/key variants deliberately share their original task group.
    return (
        ("add", "seen", _task("add", 2, 3, "total")),
        ("subtract", "seen", _task("subtract", 9, 4, "result")),
        ("add", "numbers", _task("add", 6, 10, "total")),
        ("subtract", "numbers", _task("subtract", 20, 7, "result")),
        ("add", "key", _task("add", 2, 3, "diagnostic_total")),
        ("subtract", "key", _task("subtract", 9, 4, "diagnostic_result")),
    )


def reachable_example(task: dict, phase: int, delay: int) -> DecisionExample:
    """Execute a teacher prefix then benign real tool calls, without editing step."""
    if phase not in range(4) or delay < 0 or delay > 2:
        raise ValueError("phase must be 0..3 and delay 0..2")
    context = RuntimeContext(task=task)
    executor = _executor()
    evaluator = TaskEvaluator()
    recorder = TrajectoryRecorder()
    teacher = RuleBasedPolicy()
    available = executor.registry.names() + ("stop",)

    def execute(action: Action) -> None:
        history = recorder.trajectory()
        step = len(history.transitions)
        before = context.snapshot(step)
        observation = executor.execute(action, context)
        if not observation.ok:
            raise ValueError("Diagnostic prefix tool execution failed")
        after = context.snapshot(step + 1)
        evaluation = evaluator.evaluate(task, after, action, observation, history)
        recorder.record(Transition(
            step=step, state_before=before.to_dict(), available_actions=available,
            action=action, observation=observation, state_after=after.to_dict(),
            evaluation=evaluation,
        ))

    for _ in range(phase):
        execute(teacher.decide(context.snapshot(len(recorder.trajectory().transitions)),
                               recorder.trajectory(), available))
    if delay:
        execute(Action.tool("memory.put", key="diagnostic_noise", value=99))
    if delay == 2:
        execute(Action.tool("memory.get", key="diagnostic_noise"))
    history = recorder.trajectory()
    state = context.snapshot(len(history.transitions))
    candidates = CalculateAndStoreCandidates().build(state, history, available)
    target = teacher.decide(state, history, available)
    if target not in candidates:
        raise ValueError("Teacher target absent from diagnostic candidates")
    # Execute the target on this local context and independently verify the final
    # phase's goal. Every persisted history observation already came from execution.
    if phase == 3:
        if not history.last or not history.last.evaluation.goal_satisfied:
            raise ValueError("Stop teacher target has no verified goal")
    else:
        execute(target)
    return DecisionExample(state, history, candidates, candidates.index(target),
                           teacher="rule-v0:reachable-diagnostic")


def build_state_diagnostic_suite() -> BenchmarkSuite:
    cases = []
    for operation, variant, task in _task_specs():
        for phase in range(4):
            delays = (0, 1, 2) if variant == "seen" else (0,)
            for delay in delays:
                example = reachable_example(task, phase, delay)
                slice_name = "seen_control" if variant == "seen" and not delay else (
                    "reachable_delay" if delay else "numeric_shift" if variant == "numbers" else "key_shift"
                )
                case_id = f"{operation}/{variant}/phase-{phase}/delay-{delay}"
                cases.append(BenchmarkCase(case_id, "dev", slice_name,
                                           f"train-{operation}-small-descendants", example))
                if variant == "seen" and delay == 0:
                    reversed_candidates = tuple(reversed(example.candidates))
                    reversed_example = replace(example, candidates=reversed_candidates,
                                               target_index=reversed_candidates.index(example.target))
                    cases.append(BenchmarkCase(case_id + "/reverse", "dev", "candidate_permutation",
                                               f"train-{operation}-small-descendants", reversed_example))
    return BenchmarkSuite("calculate-and-store-state-diagnostics-v0", tuple(cases))


def fit_step_lookup(train_cases: tuple[BenchmarkCase, ...]) -> dict[int, str]:
    """Learn step -> action type from original train only; never read dev targets."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for case in train_cases:
        if case.split != "train":
            raise ValueError("Step lookup may only fit on train")
        counts[case.example.state.step][action_label(case.example.target)] += 1
    return {step: count.most_common(1)[0][0] for step, count in counts.items()}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _summary(rows: list[dict]) -> dict:
    covered = [row for row in rows if row["candidate_coverage"]]
    return {
        "cases": len(rows), "groups": len({r["leakage_group"] for r in rows}),
        "candidate_coverage": len(covered) / len(rows),
        "model_correct": sum(r["model_correct"] for r in rows),
        "model_accuracy": sum(r["model_correct"] for r in rows) / len(rows),
        "step_only_correct": sum(r["step_only_correct"] for r in rows),
        "first_candidate_correct": sum(r["first_candidate_correct"] for r in rows),
        "nll_on_covered": sum(r["nll"] for r in covered) / len(covered) if covered else None,
        "min_target_margin": min(r["target_margin"] for r in covered) if covered else None,
    }


def run_state_diagnostics(root: str | Path, checkpoint: str | Path, *, source_git_sha: str | None = None) -> Path:
    """Load frozen weights; write dev-only probes and fresh model-only episodes."""
    model, tokenizer, manifest = load_decision_fit_checkpoint(checkpoint)
    # This runner's seen-control and step-only training anchor is this exact suite.
    original = build_calculate_and_store_suite()
    if manifest.get("suite_sha256") != _suite_digest(original):
        raise ValueError("Checkpoint training suite does not match diagnostic anchor")
    arm = manifest.get("arm")
    if not isinstance(arm, dict) or arm.get("arm_id") not in ("eight-fixed", "eight-shuffle"):
        raise ValueError("Diagnostics require an eight-example fit checkpoint")
    suite = build_state_diagnostic_suite()
    lookup = fit_step_lookup(original.cases_for("train"))
    run = RunLog(Path(root), "decision_state_diagnostics", {
        "checkpoint_id": manifest["checkpoint_id"], "suite_id": suite.suite_id,
        "suite_sha256": _suite_digest(suite), "source_git_sha": source_git_sha,
        "model_update": False, "temperature_fit": False, "test": "sealed",
    }, producer="evaluator")
    try:
        report, rows, episodes = measure_state_diagnostics(model, tokenizer, suite=suite, step_lookup=lookup)
        report.update(checkpoint_id=manifest["checkpoint_id"], checkpoint_manifest=manifest)
        _write(run.path / "diagnostic-suite.json", suite.to_dict())
        _write(run.path / "diagnostics.json", report)
        for name, records in (("predictions.jsonl", rows), ("episodes.jsonl", episodes)):
            (run.path / name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))
        run.finish("completed", diagnostics="diagnostics.json", predictions="predictions.jsonl",
                   episodes="episodes.jsonl", test_status="sealed")
    except Exception as error:
        run.finish("failed", error=str(error))
        raise
    return run.path


def measure_state_diagnostics(model, tokenizer, *, suite=None, task_specs=None, step_lookup=None):
    """Evaluate supplied frozen model on explicit dev probes; never load test cases.

    The checkpoint wrapper keeps its original anchor validation. New research
    recipes can share this measurement without pretending to be eight-fit runs.
    """
    suite = suite if suite is not None else build_state_diagnostic_suite()
    if any(case.split != "dev" for case in suite.cases):
        raise ValueError("State diagnostics accepts dev probes only")
    task_specs = tuple(task_specs) if task_specs is not None else _task_specs()
    lookup = step_lookup if step_lookup is not None else fit_step_lookup(
        build_calculate_and_store_suite().cases_for("train"))
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.eval()
    started = time.perf_counter()
    rows = []
    for case in suite.cases:
        example = case.example
        with torch.no_grad():
            scores = score_candidates(model, tokenizer, example.state,
                                      example.trajectory, example.candidates)
        logits = [float(v) for v in scores.tolist()]
        target = example.target_index
        nll = float(torch.logsumexp(scores.double(), dim=0).item()) - logits[target]
        predicted = int(scores.argmax().item())
        label = lookup.get(example.state.step)
        step_action = next((a for a in example.candidates if action_label(a) == label), None)
        texts = [serialize_decision_input(example.state, example.trajectory, a)
                 for a in example.candidates]
        rows.append({
            "case_id": case.case_id, "slice": case.slice_name, "split": "dev",
            "leakage_group": case.leakage_group, "step": example.state.step,
            "candidate_coverage": example.target in example.candidates,
            "candidates": [a.to_dict() for a in example.candidates],
            "target_action": example.target.to_dict(), "target_index": target,
            "model_action": example.candidates[predicted].to_dict(),
            "model_correct": predicted == target, "logits": logits, "nll": nll,
            "target_probability": math.exp(-nll),
            "target_margin": logits[target] - max(v for i, v in enumerate(logits) if i != target),
            "step_only_label": label, "step_only_abstained": step_action is None,
            "step_only_correct": step_action == example.target,
            "first_candidate_correct": example.candidates[0] == example.target,
            "case_sha256": _digest(case.to_dict()),
            "input_sha256": [_digest(t) for t in texts],
            "token_lengths": [len(tokenizer.encode(t, add_bos=True, add_eos=True)) for t in texts],
        })
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_slice[row["slice"]].append(row)
    by_id = {row["case_id"]: row for row in rows}
    permutations = []
    for row in by_slice["candidate_permutation"]:
        base = by_id[row["case_id"].removesuffix("/reverse")]
        permutations.append({"case_id": row["case_id"],
                             "same_selected_action": row["model_action"] == base["model_action"],
                             "max_reordered_logit_difference": max(abs(a-b) for a, b in zip(
                                 row["logits"], reversed(base["logits"]), strict=True))})
    episodes = []
    for operation, variant, task in task_specs:
        for policy_name, policy in (("rule", RuleBasedPolicy()), ("model_only", ModelPolicy(model, tokenizer))):
            coverage = []

            class AuditedPolicy:
                def decide(self, state, trajectory, available_actions):
                    teacher_action = RuleBasedPolicy().decide(state, trajectory, available_actions)
                    candidates = CalculateAndStoreCandidates().build(state, trajectory, available_actions)
                    coverage.append(teacher_action in candidates)
                    return policy.decide(state, trajectory, available_actions)

            trajectory = run_loop(policy=AuditedPolicy(), executor=_executor(), evaluator=TaskEvaluator(),
                                  context=RuntimeContext(task=task), max_steps=8)
            last = trajectory.last
            success = bool(last and last.evaluation.terminal and last.evaluation.goal_satisfied)
            episodes.append({"task_id": f"{operation}/{variant}", "variant": variant,
                             "policy": policy_name, "success": success, "steps": len(trajectory.transitions),
                             "teacher_candidate_coverage": sum(coverage) / len(coverage),
                             "trajectory": [t.to_dict() for t in trajectory.transitions],
                             "failure": None if success else "step_limit" if last and not last.evaluation.terminal else "stopped_without_goal"})
    unchanged = all(torch.equal(state_before[name], tensor) for name, tensor in model.state_dict().items())
    if not unchanged:
        raise RuntimeError("Read-only diagnostics changed model weights")
    report = {
        "schema_version": SCHEMA, "suite_id": suite.suite_id, "suite_sha256": _suite_digest(suite),
        "overall": _summary(rows), "slices": {name: _summary(rs) for name, rs in by_slice.items()},
        "step_lookup": lookup, "permutation_checks": permutations,
        "episodes": [{k: v for k, v in ep.items() if k != "trajectory"} for ep in episodes],
        "weights_unchanged": unchanged, "model_update": False, "temperature_fit": False,
        "test": {"status": "sealed"}, "wall_seconds": time.perf_counter() - started,
        "resources": {"python_torch_version": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"},
        "scope": "dev probes descended from two seen train task templates; not independent held-out families",
        "candidate_shortcut_limit": "first-candidate is a weak control; candidate builder itself supplies arguments and goal_verified",
    }
    return report, rows, episodes


def _suite_digest(suite: BenchmarkSuite) -> str:
    # Match the existing fit runner's canonical JSON, including its trailing newline.
    from aster.corpus.pipeline import digest, json_bytes
    return digest(json_bytes(suite.to_dict()))


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
