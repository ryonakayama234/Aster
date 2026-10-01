"""Paired compact/observed-relation diagnostic with fixed v1 mixed training."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import platform
import signal
import time

import torch

from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.fixed_v1_diagnostics import build_cases, score
from aster.benchmark.mixed_v1_preflight import ARMS, audit, build_training
from aster.benchmark.state_representation import compact_input, canonical, digest
from aster.records.decision_relations import RELATION_SERIALIZER_ID, relation_input
from aster.records.runlog import RunLog
from aster.training.decision_fit import DecisionFitStudyConfig, _new_model, _write_json, _write_jsonl
from aster.training.decision_mixed_v1 import (
    load_v1_checkpoint, measure_dev, train_arm, validate_registered,
)
from aster.training.decision_tokenizer import build_byte_decision_tokenizer

SERIALIZERS = {
    "compact-v1": ("aster-decision-compact-input-0", compact_input),
    "relations-v1": (RELATION_SERIALIZER_ID, relation_input),
}
SCHEMA = "aster-decision-relations-study-0"


def input_signature(example, serializer):
    return digest(sorted(serializer(example.state, example.trajectory, a) for a in example.candidates))


def prepare(registered):
    """Revalidate old generation and register new input overlap before training."""
    cases, slots = build_training()
    dev, pairs = build_cases()
    original = audit(cases, slots, dev, pairs)
    validate_registered(original, registered)
    tokenizer = build_byte_decision_tokenizer()
    representations = {}
    overlap = {}
    for name, (serializer_id, serializer) in SERIALIZERS.items():
        targets, union, manifest, lengths = defaultdict(set), set(), [], []
        for case in cases + dev:
            ex = case["example"]
            texts = [serializer(ex.state, ex.trajectory, a) for a in ex.candidates]
            changed_history = replace(ex.trajectory, transitions=tuple(
                replace(t, evaluation=replace(t.evaluation, goal_satisfied=not t.evaluation.goal_satisfied))
                for t in ex.trajectory.transitions))
            changed_ex = replace(ex, teacher="changed", target_index=(ex.target_index+1)%len(ex.candidates))
            if texts != [serializer(changed_ex.state, changed_history, a) for a in ex.candidates]:
                raise ValueError("Supervision leaks into serializer")
            sig = input_signature(ex, serializer)
            targets[sig].add(canonical(ex.target.to_dict()))
            if case.get("split") == "train":
                union.add(sig)
            lens = []
            for text in texts:
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if tokenizer.decode(ids) != text or len(ids)>2048:
                    raise ValueError("Fixed input roundtrip/context failure")
                lens.append(len(ids))
            lengths.extend(lens)
            manifest.append(dict(case_id=case["case_id"], signature=sig,
                                 target=ex.target.to_dict(), input_sha256=digest(texts), lengths=lens))
        if any(len(v)!=1 for v in targets.values()):
            raise ValueError("Serializer loses decision information")
        overlap[name] = {c["case_id"]:input_signature(c["example"],serializer) in union for c in dev}
        representations[name] = dict(serializer_id=serializer_id, train_unique_inputs=len(union),
            collisions=0, max_tokens=max(lengths), total_fixed_tokens=sum(lengths),
            manifest=manifest, overlap=overlap[name])
    common_overlap = {c["case_id"]:any(overlap[n][c["case_id"]] for n in SERIALIZERS) for c in dev}
    for case in dev:
        # A common exclusion denominator is used by both arms' primary summaries.
        case["overlap_union_train"] = common_overlap[case["case_id"]]
        case["overlap_by_serializer"] = {n:overlap[n][case["case_id"]] for n in SERIALIZERS}
    for pair in pairs:
        pair["nonoverlap"] = not any(common_overlap[pair[k]] for k in ("left","right"))
    report = dict(schema_version=SCHEMA, original_preflight=original,
                  representations=representations, slots=slots[ARMS[1]],
                  slot_sha256=digest(slots[ARMS[1]]), pairs=pairs,
                  common_nonoverlap_decisions=sum(not v for v in common_overlap.values()),
                  common_nonoverlap_pairs=sum(p["nonoverlap"] for p in pairs),
                  model_updates=0, model_scored=0, test_constructed=0, test_scored=0,
                  independent_holdout=False)
    return cases, dev, pairs, slots[ARMS[1]], report


def run_study(root, registered, relation_registration, *, source, debug=False, deadline_seconds=1800):
    cases, dev, pairs, slots, pre = prepare(registered)
    if pre != relation_registration:
        raise ValueError("Relation preflight registration changed")
    epochs, seeds = (1,(42,)) if debug else (25,(42,43,44))
    tokenizer = build_byte_decision_tokenizer()
    suite = BenchmarkSuite("calculate-store-relations-mixed-v1",tuple(
        BenchmarkCase(c["case_id"],"train",c["group"],c["leakage_group"],c["example"]) for c in cases))
    protocol = dict(schema_version=SCHEMA,source=source,debug=debug,epochs=epochs,seeds=seeds,
                    relation_registration_sha256=digest(pre),lr=0.001,threads=torch.get_num_threads(),
                    deadline_seconds=deadline_seconds,old_reserved_test="not_constructed/not_scored")
    run = RunLog(Path(root),"decision_relations_study",protocol,producer="trainer")
    _write_json(run.path/"protocol.json",protocol)
    _write_json(run.path/"preflight.json",pre)
    _write_json(run.path/"slots.json",slots)
    _write_json(run.path/"pairs.json",pairs)
    for filename, rows in (("train-cases.jsonl",cases),("dev-cases.jsonl",dev)):
        _write_jsonl(run.path/filename,[{**{k:v for k,v in c.items() if k!="example"},"example":c["example"].to_dict()} for c in rows])
    started = time.perf_counter()
    previous = signal.getsignal(signal.SIGALRM)
    def expire(signum, frame):
        raise TimeoutError("Relation arm deadline exceeded")
    results = []
    try:
        signal.signal(signal.SIGALRM,expire)
        for seed in seeds:
            config = DecisionFitStudyConfig(seed=seed,epochs=epochs,learning_rate=0.001)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                initial = {k:v.detach().clone() for k,v in _new_model(tokenizer,config).state_dict().items()}
            for name,(sid,serializer) in SERIALIZERS.items():
                path = run.path/f"seed-{seed}"/name
                arm_started = time.perf_counter()
                signal.setitimer(signal.ITIMER_REAL,deadline_seconds)
                try:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        model, tk, fit = train_arm(path,suite,slots,tokenizer,initial,config,source=source,
                                                  serializer=serializer,serializer_id=sid)
                    measured = measure_dev(path,model,tk,dev,pairs,suite.cases,slots,serializer=serializer)
                    re, rt, _ = load_v1_checkpoint(path/"checkpoint",serializer_id=sid)
                    cached = [json.loads(line) for line in (path/"dev-predictions.jsonl").read_text().splitlines()]
                    if any(score(re,rt,c["example"],serializer=serializer)["scores"]!=r["scores"] for c,r in zip(dev,cached,strict=True)):
                        raise ValueError("Relation dev reload mismatch")
                    result = dict(seed=seed,arm=name,status="completed",fit=fit,dev=measured,
                                  dev_reload_match=True,wall_seconds=time.perf_counter()-arm_started)
                except (TimeoutError,ValueError,RuntimeError) as error:
                    result = dict(seed=seed,arm=name,status="failed",error=str(error),wall_seconds=time.perf_counter()-arm_started)
                    path.mkdir(parents=True,exist_ok=True)
                    _write_json(path/"failure.json",result)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                results.append(result)
                _write_json(run.path/"partial-results.json",results)
                print(f"DONE seed={seed} arm={name} status={result['status']}",flush=True)
            completed = [r for r in results if r["seed"]==seed and r["status"]=="completed"]
            if len(completed)==2 and any(completed[0]["fit"][k]!=completed[1]["fit"][k] for k in ("initial_weight_sha256","order_sha256")):
                raise ValueError("Paired initialization/order mismatch")
        common = []
        for seed in seeds:
            if len([r for r in results if r["seed"]==seed and r["status"]=="completed"])!=2:
                continue
            episodes = [[json.loads(line) for line in (run.path/f"seed-{seed}"/n/"prefix-episodes.jsonl").read_text().splitlines()] for n in SERIALIZERS]
            valid = [i for i in range(len(dev)) if all(e[i]["invalid"] is None for e in episodes)]
            common.append(dict(seed=seed,common_valid=len(valid),all_started=len(dev),
                               success={n:sum(e[i]["success"] is True for i in valid) for n,e in zip(SERIALIZERS,episodes,strict=True)}))
        study = dict(**protocol,run_id=run.id,preflight=pre,models=results,common_prefix=common,
                     python=platform.python_version(),torch=torch.__version__,wall_seconds=time.perf_counter()-started,
                     independent_holdout=False,test_scored=0)
        _write_json(run.path/"study.json",study)
        run.finish("completed" if all(r["status"]=="completed" for r in results) else "failed",study="study.json")
    except BaseException as error:
        run.finish("failed",error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return run.path
