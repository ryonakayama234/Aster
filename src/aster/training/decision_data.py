"""Equal-budget training-data interventions, separate from frozen dev evaluation."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import platform
import time

import torch

from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.state_diagnostics import (
    _task_specs, build_state_diagnostic_suite, fit_step_lookup,
    measure_state_diagnostics, reachable_example,
)
from aster.benchmark.suite import _executor, _task, _teacher_examples
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision import DecisionExample
from aster.records.decision import serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Action, Transition
from aster.runtime.context import RuntimeContext
from aster.tokenizer.artifact import save_tokenizer, train_aster_tokenizer
from aster.training.decision_fit import (
    DecisionFitArm, DecisionFitStudyConfig, _audit_train_cases, _directory_digest,
    _new_model, _run_arm, _serialized_texts, _state_digest, _write_json,
    _write_jsonl, load_decision_fit_checkpoint,
)

SCHEMA = "aster-decision-data-intervention-0"
ARMS = ("repeat-control", "numbers-only", "keys-only", "numbers-and-keys")
# Deliberately different from the existing dev probes and sealed old suite.
TRAIN_NUMBERS = {"add": ((2, 3), (4, 7)), "subtract": ((9, 4), (12, 3))}
TRAIN_KEYS = {"add": ("total", "sum_slot"), "subtract": ("result", "difference_slot")}


def example_signature(example: DecisionExample) -> str:
    """Compare model-visible content, ignoring record labels such as teacher."""
    return digest(json_bytes({"inputs": [serialize_decision_input(example.state, example.trajectory, a)
                                         for a in example.candidates],
                              "target": example.target.to_dict()}))


def anchor_cases() -> tuple[BenchmarkCase, ...]:
    """Only the original eight train decisions; no old test construction."""
    cases = []
    for op in TRAIN_NUMBERS:
        task = _task(op, *TRAIN_NUMBERS[op][0], TRAIN_KEYS[op][0])
        cases.extend(BenchmarkCase(f"train-{op}-small/step-{i}", "train", "anchor",
                                   f"train-{op}-small-descendants", example)
                     for i, example in enumerate(_teacher_examples(task, {})))
    return tuple(cases)


def build_training_intervention() -> tuple[BenchmarkSuite, tuple[DecisionFitArm, ...]]:
    """32 slots/arm; 8/16/16/32 unique decisions, same operation/phase weights."""
    cases = []
    arms = []
    for arm_name in ARMS:
        ids = []
        for op in TRAIN_NUMBERS:
            for number_slot in range(2):
                for key_slot in range(2):
                    ni = number_slot if arm_name in ("numbers-only", "numbers-and-keys") else 0
                    ki = key_slot if arm_name in ("keys-only", "numbers-and-keys") else 0
                    task = _task(op, *TRAIN_NUMBERS[op][ni], TRAIN_KEYS[op][ki])
                    for phase, example in enumerate(_teacher_examples(task, {})):
                        case_id = f"{arm_name}/{op}/n{number_slot}-k{key_slot}/phase-{phase}"
                        ids.append(case_id)
                        cases.append(BenchmarkCase(case_id, "train", arm_name,
                                                   f"train-{op}-small-descendants", example))
        arms.append(DecisionFitArm(arm_name, tuple(ids), "shuffle"))
    return BenchmarkSuite("calculate-store-data-intervention-train-v0", tuple(cases)), tuple(arms)


def development_tasks():
    return _task_specs() + (
        ("add", "joint", _task("add", 6, 10, "diagnostic_total")),
        ("subtract", "joint", _task("subtract", 20, 7, "diagnostic_result")),
    )


def build_development_probes() -> BenchmarkSuite:
    base = build_state_diagnostic_suite()
    cases = list(base.cases)
    for op, variant, task in development_tasks():
        if variant == "joint":
            for phase in range(4):
                cases.append(BenchmarkCase(f"{op}/joint/phase-{phase}", "dev", "joint_shift",
                                           f"train-{op}-small-descendants", reachable_example(task, phase, 0)))
    return BenchmarkSuite("calculate-store-data-intervention-dev-probes-v0", tuple(cases))


def build_sealed_prefix_suite() -> BenchmarkSuite:
    """Entire redundant-calculation prefix generator family is reserved for test.

    Generate verified records to commit their digests, never score them here. The
    underlying calculator/store task family is still shared; this is no broad
    task-family holdout. All repetition/phase variants stay in their test group.
    """
    cases = []
    for op, left, right, key in (("add", 31, 18, "reserved_sum"),
                                 ("subtract", 46, 11, "reserved_difference")):
        for repetitions in (2, 3):
            context = RuntimeContext(task=_task(op, left, right, key))
            executor = _executor()
            evaluator = TaskEvaluator()
            recorder = TrajectoryRecorder()
            available = executor.registry.names() + ("stop",)

            def execute(action):
                history = recorder.trajectory()
                step = len(history.transitions)
                before = context.snapshot(step)
                obs = executor.execute(action, context)
                if not obs.ok:
                    raise ValueError("Sealed prefix execution failed")
                after = context.snapshot(step + 1)
                evaluation = evaluator.evaluate(context.task, after, action, obs, history)
                recorder.record(Transition(step=step, state_before=before.to_dict(), available_actions=available,
                    action=action, observation=obs, state_after=after.to_dict(), evaluation=evaluation))

            for _ in range(repetitions):
                execute(Action.tool("calculator", operation=op, left=left, right=right))
            from aster.agent.candidates import CalculateAndStoreCandidates
            for phase in (1, 2, 3):
                history = recorder.trajectory()
                state = context.snapshot(len(history.transitions))
                target = RuleBasedPolicy().decide(state, history, available)
                candidates = CalculateAndStoreCandidates().build(state, history, available)
                if target not in candidates:
                    raise ValueError("Sealed teacher candidate missing")
                example = DecisionExample(state, history, candidates, candidates.index(target),
                                          teacher="rule-v0:reserved-redundant-calculation")
                cases.append(BenchmarkCase(f"reserved-{op}/repeat-{repetitions}/phase-{phase}",
                    "test", "redundant_calculation_prefix", f"reserved-repeat-calculator-{op}", example))
                if phase < 3:
                    execute(target)
                elif not history.last or not history.last.evaluation.goal_satisfied:
                    raise ValueError("Sealed teacher stop not goal verified")
    return BenchmarkSuite("calculate-store-reserved-prefix-v0", tuple(cases))


def validate_partition(train: BenchmarkSuite, probes: BenchmarkSuite, sealed: BenchmarkSuite) -> dict:
    """Probe/train ancestry is explicit; test generator groups cannot cross."""
    if any(c.split != "train" for c in train.cases) or any(c.split != "dev" for c in probes.cases):
        raise ValueError("Training/probe split mismatch")
    if any(c.split != "test" for c in sealed.cases):
        raise ValueError("Reserved cases must be test")
    train_groups = {c.leakage_group for c in train.cases}
    probe_groups = {c.leakage_group for c in probes.cases}
    test_groups = {c.leakage_group for c in sealed.cases}
    if test_groups & (train_groups | probe_groups):
        raise ValueError("Test generator family leaks across partitions")
    train_examples = {example_signature(c.example) for c in train.cases}
    for case in probes.cases:
        if case.slice_name not in ("seen_control", "candidate_permutation") and example_signature(case.example) in train_examples:
            raise ValueError("Changed development probe appears in training")
    return {
        "train": {"suite_sha256": digest(json_bytes(train.to_dict())), "groups": sorted(train_groups)},
        "development_probes": {"suite_sha256": digest(json_bytes(probes.to_dict())), "groups": sorted(probe_groups),
            "ancestry_overlap_with_train": sorted(train_groups & probe_groups), "independent_holdout": False},
        "test": {"suite_sha256": digest(json_bytes(sealed.to_dict())), "groups": sorted(test_groups),
            "status": "sealed", "generator": "redundant_calculation_prefix", "decision_cases": len(sealed.cases),
            "scored_cases": 0, "scope": "reserved procedural-prefix family; underlying task family shared"},
        "old_test": "sealed/not_evaluated", "calibration": "not_run",
    }


def run_data_intervention(root: str | Path, *, config: DecisionFitStudyConfig | None = None,
                          source_git_sha: str | None = None) -> Path:
    config = config if config is not None else DecisionFitStudyConfig(epochs=50, learning_rate=1e-3)
    train, arms = build_training_intervention()
    probes = build_development_probes()
    sealed = build_sealed_prefix_suite()
    partition = validate_partition(train, probes, sealed)
    anchors = anchor_cases()
    tokenizer = train_aster_tokenizer(_serialized_texts(anchors), target_vocab_size=config.target_vocab_size,
        min_pair_frequency=config.min_pair_frequency, name="AsterDecisionFitTokenizer", version="0")
    audit = _audit_train_cases(train.cases, tokenizer, config.context_length)
    run = RunLog(Path(root), "decision_data_intervention", {"config": asdict(config),
        "source_git_sha": source_git_sha, "partition": partition, "tokenizer_training": "eight_original_train_decisions_only",
        "budget_per_arm": {"optimizer_updates": 32*config.epochs, "example_exposures": 32*config.epochs}}, producer="trainer")
    started = time.perf_counter()
    try:
        _write_json(run.path / "training-suite.json", train.to_dict())
        _write_json(run.path / "development-probes.json", probes.to_dict())
        _write_json(run.path / "sealed-test-suite.json", sealed.to_dict())
        _write_json(run.path / "split-manifest.json", partition)
        _write_jsonl(run.path / "input-audit.jsonl", audit)
        save_tokenizer(tokenizer, run.path / "tokenizer")
        tokenizer_sha = _directory_digest(run.path / "tokenizer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = _new_model(tokenizer, config)
        initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        initial_sha = _state_digest(initial_state)
        torch.save(initial_state, run.path / "initial-model.pt")
        results = []
        for arm in arms:
            cases = tuple(c for c in train.cases if c.case_id in arm.case_ids)
            summary = _run_arm(run.path, train, partition["train"]["suite_sha256"], tokenizer,
                initial_state, initial_sha, cases, arm, config, source_git_sha=source_git_sha)
            checkpoint = run.path / "arms" / arm.arm_id / "checkpoint"
            frozen_model, frozen_tokenizer, manifest = load_decision_fit_checkpoint(checkpoint)
            report, predictions, episodes = measure_state_diagnostics(frozen_model, frozen_tokenizer,
                suite=probes, task_specs=development_tasks(), step_lookup=fit_step_lookup(anchors))
            report.update(checkpoint_id=manifest["checkpoint_id"], partition=partition)
            evaluation = checkpoint.parent / "dev-evaluation"
            evaluation.mkdir()
            _write_json(evaluation / "diagnostics.json", report)
            _write_jsonl(evaluation / "predictions.jsonl", predictions)
            _write_jsonl(evaluation / "episodes.jsonl", episodes)
            unique = {example_signature(c.example) for c in cases}
            results.append({"fit": summary, "unique_train_decisions": len(unique), "dev": report})
            run.event("arm_measured", {"arm_id": arm.arm_id, "train_fit_target_met": summary["fit_target_met"],
                "checkpoint_id": manifest["checkpoint_id"], "weights_unchanged": report["weights_unchanged"], "test_scored_cases": 0})
        report = {"schema_version": SCHEMA, "run_id": run.id, "source_git_sha": source_git_sha,
            "config": asdict(config), "partition": partition, "initial_model_sha256": initial_sha,
            "tokenizer_sha256": tokenizer_sha, "tokenizer_anchor": "original eight train decisions",
            "arms": results, "resources": {"wall_seconds": time.perf_counter()-started,
                "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"},
            "historical_budget": {"exposures": 400, "comparable_budget": False},
            "scope": "same-task-family development probes; no independent task-family generalization claim"}
        _write_json(run.path / "study.json", report)
        run.finish("completed", study="study.json", test_status="sealed")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    return run.path
