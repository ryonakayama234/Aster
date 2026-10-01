"""Executed v1 train generation and dev partition audit; no model scoring."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
from importlib.metadata import version
from pathlib import Path
import platform
import time
from typing import Any

from aster.benchmark.fixed_v1_diagnostics import (
    Session, build_cases, candidate_signature, preflight, replay, signature,
)
from aster.benchmark.state_representation import canonical, compact_input, digest
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.records.decision import DecisionExample
from aster.records.runlog import RunLog
from aster.records.transition import Action
from aster.training.decision_data import TRAIN_KEYS, TRAIN_NUMBERS
from aster.training.decision_tokenizer import build_byte_decision_tokenizer

SCHEMA = "aster-decision-mixed-v1-preflight-0"
ARMS = ("normal-repeat-v1", "normal-plus-recovery-v1")


def build_training():
    cases, normal, recovery, get_stop = [], [], [], []
    for operation in TRAIN_NUMBERS:
        for ni, (left, right) in enumerate(TRAIN_NUMBERS[operation]):
            for ki, key in enumerate(TRAIN_KEYS[operation]):
                task = dict(kind="calculate_and_store", operation=operation,
                            left=left, right=right, store_as=key)
                task_id = f"{operation}/n{ni}-k{ki}"
                for group, phases in (("normal", range(4)), ("recovery", range(2))):
                    for phase in phases:
                        session = Session(task)
                        for _ in range(phase):
                            session.teacher_step()
                        if group == "recovery":
                            transition = session.execute(Action.tool("memory.put", key=key, value=-999))
                            if not transition.observation.ok:
                                raise ValueError("Training wrong-write failed")
                        case_id = f"train/{task_id}/{group}-{phase}"
                        cases.append(dict(case_id=case_id, task_id=task_id, split="train",
                                          group=group, phase=phase,
                                          leakage_group=f"train-{operation}-small-descendants",
                                          example=session.example()))
                        (normal if group == "normal" else recovery).append(case_id)
                        if group == "normal" and phase >= 2:
                            get_stop.append(case_id)
    return cases, {ARMS[0]: normal * 2, ARMS[1]: normal + recovery + get_stop}


def label(example):
    return example.target.name if example.target.kind == "tool" else "stop"


def independent_rule_check(example):
    """Replay, then check terminal memory and get ordering without goal flags."""
    session = replay(example)
    for _ in range(8):
        if session.teacher_step().evaluation.terminal:
            break
    trajectory = session.recorder.trajectory()
    task = example.state.task
    left, right = task["left"], task["right"]
    expected = left + right if task["operation"] == "add" else left - right
    key = task["store_as"]
    last_put, last_get = -1, -1
    for index, transition in enumerate(trajectory.transitions):
        action = transition.action
        if action.kind != "tool" or action.arguments.get("key") != key:
            continue
        if not transition.observation.ok:
            raise ValueError("Failed target-key Tool in rule continuation")
        if action.name == "memory.put":
            last_put = index
        elif (action.name == "memory.get" and isinstance(transition.observation.output, dict)
              and transition.observation.output.get("value") == expected):
            last_get = index
    stopped = bool(trajectory.last and trajectory.last.action.kind == "stop")
    valid = stopped and session.context.memory.get(key) == expected and 0 <= last_put < last_get
    if not valid or not evaluate_episode(trajectory).task_success:
        raise ValueError("Independent v1 terminal check failed")
    return len(trajectory.transitions)


def audit(cases, slots, dev, pairs, *, context_length=2048) -> dict[str, Any]:
    if len(cases) != 48 or len({c["case_id"] for c in cases}) != 48:
        raise ValueError("Expected 48 unique train case IDs")
    by_id = {c["case_id"]: c for c in cases}
    targets, union = defaultdict(set), set()
    tokenizer = build_byte_decision_tokenizer()
    manifest, lengths, transitions = [], [], 0
    for case in cases + dev:
        example = case["example"]
        if len(example.trajectory) and example.trajectory.goal_contract != CURRENT_GOAL_V1:
            raise ValueError("Non-v1 example")
        if example.target not in example.candidates:
            raise ValueError("Teacher candidate missing")
        transitions += independent_rule_check(example)
        # compact must ignore supervision/phase/evaluation metadata.
        altered = replace(example, teacher="not-the-teacher", target_index=(example.target_index + 1) % len(example.candidates))
        alternate_history = replace(example.trajectory, transitions=tuple(
            replace(t, evaluation=replace(t.evaluation, goal_satisfied=not t.evaluation.goal_satisfied))
            for t in example.trajectory.transitions))
        inputs = [compact_input(example.state, example.trajectory, a) for a in example.candidates]
        if inputs != [compact_input(altered.state, alternate_history, a) for a in altered.candidates]:
            raise ValueError("Supervision metadata leaks into compact input")
        lens = []
        for text in inputs:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if tokenizer.decode(ids) != text or len(ids) > context_length:
                raise ValueError("Byte roundtrip/context failure")
            lens.append(len(ids))
        lengths.extend(lens)
        sig = signature(example)
        target = canonical(example.target.to_dict())
        targets[sig].add(target)
        if case.get("split") == "train":
            union.add(sig)
        manifest.append(dict(case_id=case["case_id"], split=case.get("split", "dev"),
                             signature=sig, example_sha256=digest(example.to_dict()),
                             candidate_sha256=digest(candidate_signature(example)),
                             target=example.target.to_dict(), candidate_token_lengths=lens))
        # Saved representation must reproduce the same real state/teacher/candidates.
        restored = DecisionExample.from_dict(example.to_dict())
        if restored != example:
            raise ValueError("Example persistence mismatch")
        replay(restored)
    if any(len(v) != 1 for v in targets.values()) or len(union) != 48:
        raise ValueError("Model-visible input collision")
    arms = {}
    normal_ids = {c["case_id"] for c in cases if c["group"] == "normal"}
    for name in ARMS:
        ids = slots[name]
        if len(ids) != 64 or not set(ids) <= set(by_id) or not normal_ids <= set(ids):
            raise ValueError("Arm slots/normal support mismatch")
        counts = Counter(label(by_id[i]["example"]) for i in ids)
        if counts != Counter({"calculator": 16, "memory.put": 16, "memory.get": 16, "stop": 16}):
            raise ValueError("Action frequency mismatch")
        expected = 32 if name == ARMS[0] else 48
        if len(set(ids)) != expected:
            raise ValueError("Unique arm count mismatch")
        arms[name] = dict(slots=64, unique=expected, action_slots=dict(counts),
                          normal_slots=sum(i in normal_ids for i in ids),
                          case_exposures_25_epochs={i: n*25 for i,n in sorted(Counter(ids).items())},
                          slot_sha256=digest(ids))
    dev_audit = preflight(dev, pairs, union, {sig: targets[sig] for sig in union})
    train_pairs = []
    for normal_case in cases:
        if normal_case["group"] != "normal":
            continue
        for recovery_case in cases:
            if recovery_case["group"] != "recovery" or recovery_case["task_id"] != normal_case["task_id"]:
                continue
            n, r = normal_case["example"], recovery_case["example"]
            if candidate_signature(n) == candidate_signature(r) and n.target != r.target:
                train_pairs.append(dict(left=normal_case["case_id"], right=recovery_case["case_id"],
                                        same_step=n.state.step == r.state.step))
    return dict(schema_version=SCHEMA, goal_contract=CURRENT_GOAL_V1,
                input_serializer_id="aster-decision-compact-input-0", train_cases=48,
                train_unique_inputs=len(union), train_rule_success=48, dev_rule_success=128,
                train_candidate_coverage=48,
                tokenizer_spec_sha256=digest(dict(vocab={str(i):bytes([i]).hex() for i in range(256)},
                                                  merges=[], special_tokens={"<|bos|>":256,"<|eos|>":257})),
                independent_terminal_checks=len(cases)+len(dev), checked_transitions=transitions,
                input_collisions=0, metadata_leakage=False, max_candidate_tokens=max(lengths),
                candidate_sequence_count=len(lengths), token_length_sha256=digest(lengths),
                arms=arms, dev=dev_audit, dev_nonoverlap_pairs=sum(p["nonoverlap"] for p in pairs),
                train_pairs=train_pairs, case_manifest=manifest,
                train_suite_sha256=digest([{**{k:v for k,v in c.items() if k != "example"},
                                          "example":c["example"].to_dict()} for c in cases]),
                dev_suite_sha256=digest([{**{k:v for k,v in c.items() if k != "example"},
                                        "example":c["example"].to_dict()} for c in dev]),
                pairs_sha256=digest(pairs), model_scored=0, updates=0,
                test_constructed=0, test_scored=0, independent_holdout=False)


def run_preflight(root: Path, *, source=None):
    start = time.perf_counter()
    run = RunLog(root, "decision_mixed_v1_preflight", {"source": source}, producer="evaluator")
    try:
        cases, slots = build_training()
        dev, pairs = build_cases()
        report = audit(cases, slots, dev, pairs)
        report.update(source=source, run_id=run.id, python=platform.python_version(),
                      torch=version("torch"), environment="ChatGPT Linux CPU; no model computation",
                      wall_seconds=time.perf_counter()-start)
        for filename, value in (("summary.json", report), ("slots.json", slots), ("pairs.json", pairs)):
            (run.path/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        for filename, rows in (("train-cases.jsonl", cases), ("dev-cases.jsonl", dev)):
            (run.path/filename).write_text("".join(canonical({**{k:v for k,v in c.items() if k != "example"},
                                                             "example":c["example"].to_dict()})+"\n" for c in rows), encoding="utf-8")
        run.finish("completed", summary="summary.json")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    return run.path
