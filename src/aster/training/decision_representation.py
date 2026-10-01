"""Paired raw/compact diagnostics; fixed train data, dev-only state probes."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import signal
import time

import torch

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.state_representation import (
    audit_suite, build_suite, canonical, compact_input, digest, execute, replay,
)
from aster.inference.decide import score_candidates
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.records.runlog import RunLog
from aster.training.decision_data import build_development_probes, development_tasks
from aster.training.decision_fit import (
    DecisionFitArm, DecisionFitStudyConfig, _fit_metrics, _new_model, _predict_cases,
    _prediction_rows, _run_arm, _state_digest, _write_json, _write_jsonl, _max_rss_kib,
    load_decision_fit_checkpoint,
)
from aster.training.decision_tokenizer import build_byte_decision_tokenizer, build_comparison_training_suite
from aster.benchmark.state_representation import make_example

SCHEMA = "aster-decision-representation-comparison-0"
SERIALIZERS = {"raw": ("aster-decision-input-0", serialize_decision_input),
               "compact": ("aster-decision-compact-input-0", compact_input)}


def audit_inputs(cases, tokenizer, serializer, context_length):
    """Audit exact model inputs, preserving labels only outside the projection."""
    rows, signatures = [], defaultdict(set)
    for case in cases:
        if case.split == "test":
            raise ValueError("Test inputs are sealed")
        example = case.example
        texts = [serializer(example.state, example.trajectory, a) for a in example.candidates]
        lengths, candidates = [], []
        for action, text in zip(example.candidates, texts, strict=True):
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if tokenizer.decode(ids) != text:
                raise ValueError("Input does not round-trip")
            if len(ids) > context_length:
                raise ValueError(f"Input exceeds context: {case.case_id}")
            lengths.append(len(ids))
            candidates.append({"action": action.to_dict(), "text": text,
                               "input_sha256": digest(text), "token_ids": ids})
        signatures[canonical(texts)].add(canonical(example.target.to_dict()))
        rows.append({"case_id": case.case_id, "split": case.split, "candidates": candidates,
                     "target_index": example.target_index, "lengths": lengths})
    if any(len(targets) > 1 for targets in signatures.values()):
        raise ValueError("Model-visible inputs require conflicting targets")
    return rows, {"cases": len(rows), "max_tokens": max(max(r["lengths"]) for r in rows),
                  "semantic_tokens_per_pass": sum(sum(r["lengths"]) for r in rows),
                  "padded_tokens_per_pass": sum(len(r["lengths"])*max(r["lengths"]) for r in rows),
                  "serialization_sha256": digest(rows)}


def candidate_lookup(train, dev):
    """Train-only candidate-set/target frequency classifier; unknown sets abstain.

    Exact Action payloads are used, not state/history or labels from dev. Ties use
    canonical Action order, so this control is invariant to candidate permutation.
    This lookup classifier is not a neural candidate-only baseline.
    """
    counts = defaultdict(Counter)
    def key(example):
        return canonical(sorted(canonical(a.to_dict()) for a in example.candidates))
    for case in train.cases:
        if case.split != "train":
            raise ValueError("Candidate classifier fits train only")
        counts[key(case.example)][canonical(case.example.target.to_dict())] += 1
    predictions = {}
    for case in dev.cases:
        counter = counts.get(key(case.example), Counter())
        winner = min(counter, key=lambda a: (-counter[a], a)) if counter else None
        predictions[case.case_id] = {"covered": bool(counter),
            "predicted_action": json.loads(winner) if winner else None,
            "correct": winner == canonical(case.example.target.to_dict())}
    return {"kind": "train-only exact candidate-set frequency lookup; unknown=abstain",
            "fit_cases": len(train.cases), "dev_cases": len(dev.cases),
            "covered": sum(r["covered"] for r in predictions.values()),
            "correct": sum(r["correct"] for r in predictions.values()), "predictions": predictions}


def rollout(model, tokenizer, example, serializer):
    """Continue a real prefix, preserving absolute steps and model-visited states."""
    context, recorder, tools = replay(example)
    prefix_length = len(example.trajectory.transitions)
    decisions, first_error, invalid = [], None, None
    signatures = defaultdict(set)
    for _ in range(8):
        history = recorder.trajectory()
        state = context.snapshot(len(history.transitions))
        available = tools.registry.names() + ("stop",)
        target = RuleBasedPolicy().decide(state, history, available)
        candidates = CalculateAndStoreCandidates().build(state, history, available)
        inputs = [serializer(state, history, a) for a in candidates]
        signatures[canonical(inputs)].add(canonical(target.to_dict()))
        if any(len(v) > 1 for v in signatures.values()):
            raise ValueError("Visited input target collision")
        ids = [tokenizer.encode(s, add_bos=True, add_eos=True) for s in inputs]
        if any(len(v) > model.backbone.config.context_length for v in ids):
            invalid = "visited_input_exceeds_context"
            break
        with torch.no_grad():
            scores = score_candidates(model, tokenizer, state, history, candidates, serializer=serializer)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Nonfinite model scores")
        index = int(scores.argmax().item())
        selected = candidates[index]
        row = {"step": state.step, "candidates": [a.to_dict() for a in candidates],
               "scores": scores.tolist(), "target": target.to_dict(), "selected": selected.to_dict(),
               "correct": selected == target, "coverage": target in candidates,
               "inputs": inputs, "token_ids": ids,
               "state": state.to_dict()}
        decisions.append(row)
        if first_error is None and selected != target:
            first_error = row
        last = execute(context, recorder, tools, selected)
        if last.evaluation.terminal:
            break
    last = recorder.trajectory().last
    return {"prefix_length": prefix_length, "max_additional_actions": 8, "fallback": False,
            "invalid_evaluation": invalid, "additional_actions": len(decisions),
            "success": None if invalid else bool(last and len(decisions) and last.evaluation.terminal and last.evaluation.goal_satisfied),
            "first_error": first_error, "decisions": decisions,
            "trajectory": [t.to_dict() for t in recorder.trajectory().transitions]}


def measure_dev(model, tokenizer, suite, pairs, old_probes, serializer, output):
    model.eval()
    before = _state_digest(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    predictions = _predict_cases(model, tokenizer, suite.cases, serializer=serializer)
    rows = _prediction_rows(epoch=0, cases=suite.cases, predictions=predictions)
    if any(not math.isfinite(x) for p in predictions for x in p.logits):
        raise ValueError("Nonfinite dev scores")
    by_id = {p.case_id: p.correct for p in predictions}
    pair_rows = [{**pair, "both_correct": by_id[pair["left"]] and by_id[pair["right"]]} for pair in pairs]
    old_predictions = _predict_cases(model, tokenizer, old_probes.cases, serializer=serializer)
    old_rows = _prediction_rows(epoch=0, cases=old_probes.cases, predictions=old_predictions)
    episodes = [{"case_id": c.case_id, "slice": c.slice_name,
                 **rollout(model, tokenizer, c.example, serializer)} for c in suite.cases]
    initial_episodes = [{"task_id": f"{op}/{variant}", "variant": variant,
                         **rollout(model, tokenizer, make_example(task, 0, 0), serializer)}
                        for op, variant, task in development_tasks()]
    # Also audit sufficiency across different episodes, not merely within one path.
    signatures = defaultdict(set)
    for episode in episodes + initial_episodes:
        for row in episode["decisions"]:
            signatures[canonical(row["inputs"])].add(canonical(row["target"]))
    if any(len(v) > 1 for v in signatures.values()):
        raise ValueError("Cross-episode visited input target collision")
    unchanged = before == _state_digest(model.state_dict())
    rng_unchanged = torch.equal(rng_before, torch.get_rng_state())
    if not unchanged or not rng_unchanged:
        raise RuntimeError("Evaluation changed weights or torch RNG")
    _write_jsonl(output / "dev-predictions.jsonl", rows)
    _write_jsonl(output / "old-dev-predictions.jsonl", old_rows)
    _write_jsonl(output / "pair-predictions.jsonl", pair_rows)
    _write_jsonl(output / "prefix-episodes.jsonl", episodes)
    _write_jsonl(output / "initial-episodes.jsonl", initial_episodes)
    return {"teacher_prefix": _fit_metrics(predictions),
            "pairs": {r: {"count": sum(p["relation"] == r for p in pair_rows),
                          "both_correct": sum(p["relation"] == r and p["both_correct"] for p in pair_rows)}
                      for r in {p["relation"] for p in pairs}},
            "old_dev": _fit_metrics(old_predictions),
            "prefix_episodes": {"count": len(episodes), "success": sum(e["success"] is True for e in episodes),
                                "invalid": sum(e["invalid_evaluation"] is not None for e in episodes)},
            "initial_episodes": {variant: {"count": sum(e["variant"] == variant for e in initial_episodes),
                                            "success": sum(e["variant"] == variant and e["success"] is True for e in initial_episodes),
                                            "invalid": sum(e["variant"] == variant and e["invalid_evaluation"] is not None for e in initial_episodes)}
                                 for variant in {e["variant"] for e in initial_episodes}},
            "visited_teacher_coverage": all(d["coverage"] for e in episodes+initial_episodes for d in e["decisions"]),
            "weights_unchanged": unchanged, "rng_unchanged": rng_unchanged,
            "visited_input_collisions": 0, "test_scored_cases": 0}


def run_comparison(root, *, config=None, source_git_sha=None, source_provenance=None, deadline_seconds=1800):
    config = config or DecisionFitStudyConfig(epochs=50, learning_rate=1e-3)
    train, train_ids = build_comparison_training_suite()
    dev, pairs = build_suite()
    old_probes = build_development_probes()
    preflight, _ = audit_suite(dev, pairs)
    tokenizer = build_byte_decision_tokenizer()
    audits = {}
    # Stop the whole experiment on fixed-input overflow before either arm trains.
    for name, (_, serializer) in SERIALIZERS.items():
        audits[name] = {split: audit_inputs(cases, tokenizer, serializer, config.context_length)
                       for split, cases in (("train", train.cases), ("dev", dev.cases), ("old_dev", old_probes.cases))}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        initial_model = _new_model(tokenizer, config)
    initial = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    initial_hash = _state_digest(initial)
    exact = (config.epochs == 50 and config.learning_rate == 1e-3 and config.seed in (42,43,44)
             and config.context_length == 2048 and config.width == 16 and config.heads == 2
             and config.layers == 1 and config.stability_window == 3 and config.train_backbone
             and torch.get_num_threads() == 2 and deadline_seconds == 1800)
    protocol = {"schema_version": SCHEMA, "config": asdict(config), "source_git_sha": source_git_sha,
                "source_provenance": source_provenance,
                "status": "preregistered-measurement" if exact else "debug/non-preregistered",
                "training_suite_sha256": digest(train.to_dict()), "dev_suite_sha256": digest(dev.to_dict()),
                "pairs_sha256": digest(pairs), "old_dev_sha256": digest(old_probes.to_dict()),
                "initial_model_sha256": initial_hash, "deadline_seconds_per_arm": deadline_seconds,
                "parent_groups": 2, "independent_holdout": False,
                "test": "sealed/not_constructed/not_scored", "calibration": "not_run"}
    run = RunLog(Path(root), "decision_representation_comparison", protocol, producer="trainer")
    results, orders = [], []
    started = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expire(signum, frame):
        raise TimeoutError("Representation arm exceeded deadline")
    try:
        _write_json(run.path / "protocol.json", protocol)
        _write_json(run.path / "train-suite.json", train.to_dict())
        _write_json(run.path / "dev-suite.json", dev.to_dict())
        _write_json(run.path / "pairs.json", pairs)
        _write_json(run.path / "preflight.json", preflight)
        _write_json(run.path / "candidate-lookup.json", candidate_lookup(train, dev))
        signal.signal(signal.SIGALRM, expire)
        for name, (serializer_id, serializer) in SERIALIZERS.items():
            signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
            arm_started = time.perf_counter()
            for split, (rows, _) in audits[name].items():
                _write_jsonl(run.path / f"{name}-{split}-input-audit.jsonl", rows)
            arm = DecisionFitArm(name, train_ids, "shuffle")
            summary = _run_arm(run.path, train, digest(train.to_dict()), tokenizer, initial, initial_hash,
                               train.cases, arm, config, source_git_sha=source_git_sha,
                               serializer=serializer, input_serializer_id=serializer_id)
            model, saved_tokenizer, _ = load_decision_fit_checkpoint(
                run.path / "arms" / name / "checkpoint", expected_input_serializer_id=serializer_id)
            if not summary["reload_metrics_match"]:
                raise RuntimeError("Reload metrics mismatch")
            measured = measure_dev(model, saved_tokenizer, dev, pairs, old_probes, serializer, run.path / "arms" / name)
            result = {"arm": name, "fit": summary, "dev": measured,
                      "token_budget": {k: v[1] for k,v in audits[name].items()},
                      "wall_seconds": time.perf_counter()-arm_started,
                      "training_semantic_tokens": audits[name]["train"][1]["semantic_tokens_per_pass"]*config.epochs,
                      "training_padded_tokens": audits[name]["train"][1]["padded_tokens_per_pass"]*config.epochs}
            results.append(result)
            _write_json(run.path / "arms" / name / "result.json", result)
            orders.append((run.path / "arms" / name / "epoch-order.jsonl").read_bytes())
            run.event("arm_completed", {"arm": name, "fit": summary["fit_target_met"]})
            print(f"seed={config.seed} arm={name} fit={summary['fit_target_met']} dev={measured['teacher_prefix']['correct']}/48", flush=True)
            signal.setitimer(signal.ITIMER_REAL, 0)
        if orders[0] != orders[1]:
            raise RuntimeError("Paired epoch order mismatch")
        study = {**protocol, "run_id": run.id, "arms": results, "same_epoch_orders": True,
                 "full_initial_weights_identical": True, "parameter_count": sum(p.numel() for p in initial_model.parameters()),
                 "wall_seconds": time.perf_counter()-started, "python": platform.python_version(),
                 "process_max_rss_kib": _max_rss_kib(), "rss_scope": "process maximum across sequential arms, not arm memory comparison",
                 "torch": torch.__version__, "threads": torch.get_num_threads(), "test_scored_cases": 0}
        _write_json(run.path / "study.json", study)
        run.finish("completed", study="study.json")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return run.path
