"""Fixed compact/byte experiment: reachable wrong-write training intervention."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import signal
import time

import torch

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.state_representation import (
    audit_suite, build_suite, canonical, compact_input, continue_rule, digest,
    execute, executor, make_example,
)
from aster.records.decision import DecisionExample
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Action
from aster.runtime.context import RuntimeContext
from aster.training.decision_data import build_development_probes, development_tasks
from aster.training.decision_fit import (
    DecisionFitArm, DecisionFitStudyConfig, _max_rss_kib, _new_model, _run_arm,
    _state_digest, _write_json, _write_jsonl, load_decision_fit_checkpoint,
)
from aster.training.decision_representation import audit_inputs, candidate_lookup, measure_dev
from aster.training.decision_tokenizer import build_byte_decision_tokenizer, build_comparison_training_suite

SCHEMA = "aster-decision-state-training-0"
SERIALIZER_ID = "aster-decision-compact-input-0"


def wrong_write_example(task, phase: int, value: int, *, double_write=False):
    """Successful but task-incorrect writes; never duplicate the reserved calculator family."""
    if phase not in (0, 1):
        raise ValueError("Wrong-write phase must be 0 or 1")
    context, recorder, tools = RuntimeContext(task), TrajectoryRecorder(), executor()
    if phase == 1:
        calculation = Action.tool("calculator", operation=task["operation"], left=task["left"], right=task["right"])
        transition = execute(context, recorder, tools, calculation)
        if not transition.observation.ok or transition.observation.output == value:
            raise ValueError("Wrong-write value must differ from calculation")
    for _ in range(2 if double_write else 1):
        transition = execute(context, recorder, tools, Action.tool("memory.put", key=task["store_as"], value=value))
        if not transition.observation.ok:
            raise ValueError("Wrong-write execution failed")
    history = recorder.trajectory()
    state = context.snapshot(len(history.transitions))
    available = tools.registry.names() + ("stop",)
    candidates = CalculateAndStoreCandidates().build(state, history, available)
    target = RuleBasedPolicy().decide(state, history, available)
    expected = "calculator" if phase == 0 else "memory.put"
    if target.name != expected or target not in candidates:
        raise ValueError("Wrong-write teacher mismatch")
    return DecisionExample(state, history, candidates, candidates.index(target), teacher="rule-v0:wrong-write")


def signature(example):
    """Candidate-order invariant exact model input identity, excluding record labels."""
    return digest(sorted(compact_input(example.state, example.trajectory, a) for a in example.candidates))


def candidate_signature(example):
    return canonical(sorted(canonical(a.to_dict()) for a in example.candidates))


def build_training():
    baseline, _ = build_comparison_training_suite()
    state_cases = []
    for case in baseline.cases:
        phase = int(case.case_id.rsplit("-", 1)[1])
        example = (wrong_write_example(case.example.state.task, phase, -999)
                   if phase < 2 else case.example)
        state_cases.append(BenchmarkCase(case.case_id, "train", "wrong_write_intervention",
                                        case.leakage_group, example))
    return {"baseline": baseline, "state": BenchmarkSuite("calculate-store-state-train-v0", tuple(state_cases))}


def build_new_dev():
    cases, pairs = [], []
    for operation, variant, task in development_tasks():
        examples = (wrong_write_example(task, 0, -777, double_write=True),
                    wrong_write_example(task, 1, -777), make_example(task, 2, 0), make_example(task, 3, 0))
        ids = [f"{operation}/{variant}/state-{i}" for i in range(4)]
        for i, example in enumerate(examples):
            cases.append(BenchmarkCase(ids[i], "dev", variant, f"train-{operation}-small-descendants", example))
        pairs.extend([
            {"relation": "same_step_same_candidates_different_target", "left": ids[1], "right": ids[2]},
            {"relation": "same_step_different_target", "left": ids[0], "right": ids[1]},
            {"relation": "same_step_different_target", "left": ids[0], "right": ids[2]},
        ])
    return BenchmarkSuite("calculate-store-wrong-write-dev-v0", tuple(cases)), pairs


def preflight(trains, new_dev, pairs, original, old):
    """Replay, diagnose actual visible change, and classify union-train overlap."""
    targets = defaultdict(set)
    for suite in list(trains.values()) + [new_dev, original, old]:
        if any(c.split == "test" for c in suite.cases):
            raise ValueError("Test is sealed")
        for case in suite.cases:
            example = case.example
            available = executor().registry.names() + ("stop",)
            canonical_candidates = CalculateAndStoreCandidates().build(example.state, example.trajectory, available)
            if Counter(canonical(a.to_dict()) for a in canonical_candidates) != Counter(canonical(a.to_dict()) for a in example.candidates):
                raise ValueError("Candidate replay mismatch")
            replay_example = replace(example, candidates=canonical_candidates, target_index=canonical_candidates.index(example.target))
            if not continue_rule(replay_example)["success"]:
                raise ValueError("Rule continuation failed")
            targets[signature(case.example)].add(canonical(case.example.target.to_dict()))
    if any(len(v) != 1 for v in targets.values()):
        raise ValueError("Compact teacher collision")
    counts = {arm: dict(Counter(c.example.target.name or "stop" for c in suite.cases))
              for arm, suite in trains.items()}
    if counts["baseline"] != counts["state"] or set(counts["state"].values()) != {8}:
        raise ValueError("Target frequency mismatch")
    a, b = trains["baseline"], trains["state"]
    if [c.case_id for c in a.cases] != [c.case_id for c in b.cases]:
        raise ValueError("Slot mismatch")
    changed = [c.case_id for c, d in zip(a.cases, b.cases, strict=True) if signature(c.example) != signature(d.example)]
    if len(changed) != 16:
        raise ValueError("Expected sixteen model-visible changed slots")
    union = {signature(c.example) for suite in trains.values() for c in suite.cases}
    overlaps = {name: {c.case_id: signature(c.example) in union for c in suite.cases}
                for name, suite in (("new_dev", new_dev), ("original_dev", original), ("old_dev", old))}
    by_id = {c.case_id: c.example for c in new_dev.cases}
    pair_rows = []
    for pair in pairs:
        left, right = (by_id[pair[k]] for k in ("left", "right"))
        if left.state.step != right.state.step or left.target == right.target:
            raise ValueError("State pair relation failed")
        if pair["relation"] == "same_step_same_candidates_different_target" and candidate_signature(left) != candidate_signature(right):
            raise ValueError("Candidate sets differ in main pair")
        pair_rows.append({**pair, "eligible_nonoverlap": not any(overlaps["new_dev"][pair[k]] for k in ("left", "right"))})
    # Prove that old irrelevant-delay augmentation leaves compact input unchanged.
    task = a.cases[0].example.state.task
    delay_alias = all(signature(make_example(task, phase, 0)) == signature(make_example(task, phase, 1)) for phase in range(4))
    if not delay_alias:
        raise ValueError("Expected irrelevant-delay projection identity")
    return {"schema_version": SCHEMA, "changed_slots": changed, "target_counts": counts,
            "input_collisions": 0, "irrelevant_delay_is_input_alias": delay_alias,
            "overlap": overlaps, "pairs": pair_rows, "independent_holdout": False,
            "rule_verified_cases": sum(len(s.cases) for s in list(trains.values()) + [new_dev, original, old]),
            "step_only_new_dev_correct": sum((c.example.target.name or "stop") == {0:"calculator", 1:"memory.put", 2:"memory.get", 3:"stop"}.get(c.example.state.step) for c in new_dev.cases),
            "first_candidate_new_dev_correct": sum(c.example.target_index == 0 for c in new_dev.cases),
            "ancestry": "two existing task parent groups; wrong-write family shared train/dev",
            "reserved_test_family": "redundant calculator prefixes untouched/not constructed/not scored",
            "test_scored_cases": 0}


def summarize_nonoverlap(output, audit):
    rows = [json.loads(line) for line in (output / "dev-predictions.jsonl").read_text().splitlines()]
    by_id = {r["case_id"]: r for r in rows}
    nonoverlap = [r for r in rows if not audit["overlap"]["new_dev"][r["case_id"]]]
    primary = [p for p in audit["pairs"] if p["relation"] == "same_step_same_candidates_different_target" and p["eligible_nonoverlap"]]
    episodes = [json.loads(line) for line in (output / "prefix-episodes.jsonl").read_text().splitlines()]
    selected = [e for e in episodes if not audit["overlap"]["new_dev"][e["case_id"]]]
    result = {"nonoverlap_decisions": {"count": len(nonoverlap), "correct": sum(r["correct"] for r in nonoverlap)},
              "primary_pairs": {"count": len(primary), "both_correct": sum(by_id[p["left"]]["correct"] and by_id[p["right"]]["correct"] for p in primary)},
              "nonoverlap_prefix": {"count": len(selected), "success": sum(e["success"] is True for e in selected), "invalid": sum(e["invalid_evaluation"] is not None for e in selected)},
              "retention_cases": len(rows)-len(nonoverlap)}
    _write_json(output / "nonoverlap.json", result)
    return result


def summarize_overlap_file(path, overlap):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {name: {"count": sum(overlap[r["case_id"]] == wanted for r in rows),
                   "correct": sum(overlap[r["case_id"]] == wanted and r["correct"] for r in rows)}
            for name, wanted in (("retention", True), ("nonoverlap_diagnostic", False))}


def run_state_training(root, *, config=None, source_git_sha=None, source_provenance=None, deadline_seconds=1800):
    config = config or DecisionFitStudyConfig(epochs=50, learning_rate=1e-3)
    trains = build_training()
    new_dev, pairs = build_new_dev()
    original, original_pairs = build_suite()
    old = build_development_probes()
    audit = preflight(trains, new_dev, pairs, original, old)
    original_audit, _ = audit_suite(original, original_pairs)
    tokenizer = build_byte_decision_tokenizer()
    input_audits = {name: audit_inputs(suite.cases, tokenizer, compact_input, config.context_length)
                    for name, suite in list(trains.items()) + [("new_dev", new_dev), ("original_dev", original), ("old_dev", old)]}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        initial_model = _new_model(tokenizer, config)
    initial = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    initial_hash = _state_digest(initial)
    exact = (config == DecisionFitStudyConfig(epochs=50, learning_rate=1e-3, seed=config.seed)
             and config.seed in (42, 43, 44) and torch.get_num_threads() == 2 and deadline_seconds == 1800)
    protocol = {"schema_version": SCHEMA, "config": asdict(config), "input_serializer_id": SERIALIZER_ID,
                "status": "preregistered-measurement" if exact else "debug/non-preregistered",
                "source_git_sha": source_git_sha, "source_provenance": source_provenance,
                "suite_digests": {k: digest(v.to_dict()) for k,v in list(trains.items()) + [("new_dev",new_dev),("original_dev",original),("old_dev",old)]},
                "initial_model_sha256": initial_hash, "preflight_sha256": digest(audit),
                "deadline_seconds_per_arm": deadline_seconds, "parent_groups": 2, "independent_holdout": False,
                "test": "sealed/not_constructed/not_scored", "calibration": "not_run"}
    run = RunLog(Path(root), "decision_state_training", protocol, producer="trainer")
    results, orders = [], []
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.perf_counter()
    def expire(signum, frame):
        raise TimeoutError("State training arm exceeded deadline")
    try:
        _write_json(run.path / "protocol.json", protocol)
        _write_json(run.path / "preflight.json", audit)
        _write_json(run.path / "original-preflight.json", original_audit)
        _write_json(run.path / "pairs.json", audit["pairs"])
        for name, suite in list(trains.items()) + [("new_dev",new_dev),("original_dev",original),("old_dev",old)]:
            _write_json(run.path / f"{name}-suite.json", suite.to_dict())
            _write_jsonl(run.path / f"{name}-inputs.jsonl", input_audits[name][0])
        signal.signal(signal.SIGALRM, expire)
        for name, train in trains.items():
            signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
            arm_started = time.perf_counter()
            ids = tuple(c.case_id for c in train.cases)
            arm = DecisionFitArm(name, ids, "shuffle")
            summary = _run_arm(run.path, train, digest(train.to_dict()), tokenizer, initial, initial_hash,
                               train.cases, arm, config, source_git_sha=source_git_sha,
                               serializer=compact_input, input_serializer_id=SERIALIZER_ID)
            if not summary["reload_metrics_match"]:
                raise RuntimeError("Reload metrics mismatch")
            model, saved_tokenizer, _ = load_decision_fit_checkpoint(run.path / "arms" / name / "checkpoint", expected_input_serializer_id=SERIALIZER_ID)
            output = run.path / "arms" / name
            original_output = output / "original"
            original_output.mkdir()
            measured = measure_dev(model, saved_tokenizer, new_dev, pairs, old, compact_input, output)
            original_measured = measure_dev(model, saved_tokenizer, original, original_pairs, old, compact_input, original_output)
            nonoverlap = summarize_nonoverlap(output, audit)
            _write_json(output / "candidate-lookup.json", candidate_lookup(train, new_dev))
            result = {"arm": name, "fit": summary, "new_dev": measured, "nonoverlap": nonoverlap,
                      "original_dev": original_measured, "train_token_budget": input_audits[name][1],
                      "original_overlap_metrics": summarize_overlap_file(original_output / "dev-predictions.jsonl", audit["overlap"]["original_dev"]),
                      "old_overlap_metrics": summarize_overlap_file(output / "old-dev-predictions.jsonl", audit["overlap"]["old_dev"]),
                      "training_semantic_tokens": input_audits[name][1]["semantic_tokens_per_pass"]*config.epochs,
                      "training_padded_tokens": input_audits[name][1]["padded_tokens_per_pass"]*config.epochs,
                      "wall_seconds": time.perf_counter()-arm_started}
            results.append(result)
            _write_json(output / "result.json", result)
            orders.append((output / "epoch-order.jsonl").read_bytes())
            run.event("arm_completed", {"arm": name, "fit": summary["fit_target_met"]})
            print(f"seed={config.seed} arm={name} fit={summary['fit_target_met']} main_pairs={nonoverlap['primary_pairs']}", flush=True)
            signal.setitimer(signal.ITIMER_REAL, 0)
        if orders[0] != orders[1]:
            raise RuntimeError("Paired slot permutation mismatch")
        study = {**protocol, "run_id": run.id, "arms": results, "same_epoch_orders": True,
                 "full_initial_weights_identical": True, "preflight": audit,
                 "parameter_count": sum(p.numel() for p in initial_model.parameters()),
                 "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
                 "wall_seconds": time.perf_counter()-started, "process_max_rss_kib": _max_rss_kib(),
                 "rss_scope": "shared process maximum, not per-arm comparison", "test_scored_cases": 0}
        _write_json(run.path / "study.json", study)
        run.finish("completed", study="study.json")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return run.path
