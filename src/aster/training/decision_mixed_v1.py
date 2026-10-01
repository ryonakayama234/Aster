"""Paired fresh training with normal support retained, under explicit v1."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import platform
import signal
import time
from typing import Any

import torch

from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.fixed_v1_diagnostics import candidate_signature, rollout, score, summarize
from aster.benchmark.mixed_v1_preflight import ARMS, audit, build_training
from aster.benchmark.fixed_v1_diagnostics import build_cases
from aster.benchmark.state_representation import canonical, compact_input, digest
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.records.runlog import RunLog
from aster.training.decision import decision_loss
from aster.training.decision_fit import (
    DecisionFitArm, DecisionFitStudyConfig, _fit_metrics, _max_rss_kib,
    _new_model, _predict_cases, _prediction_rows, _save_fit_checkpoint,
    _stable_fit, _state_digest, _write_json, _write_jsonl, load_decision_fit_checkpoint,
)
from aster.training.decision_tokenizer import build_byte_decision_tokenizer

SCHEMA = "aster-decision-mixed-v1-study-0"
SERIALIZER = "aster-decision-compact-input-0"


def load_v1_checkpoint(path):
    contract = json.loads((path.parent/"training-contract.json").read_text())
    if contract["goal_contract"] != CURRENT_GOAL_V1 or contract["input_serializer_id"] != SERIALIZER:
        raise ValueError("v1 training contract mismatch")
    model, tokenizer, manifest = load_decision_fit_checkpoint(path, expected_input_serializer_id=SERIALIZER)
    if contract["checkpoint_id"] != manifest["checkpoint_id"] or digest(contract["slots"]) != contract["slot_sha256"]:
        raise ValueError("v1 checkpoint/slots identity mismatch")
    return model, tokenizer, manifest


def validate_registered(actual, registered):
    keys = ("goal_contract", "input_serializer_id", "train_suite_sha256", "dev_suite_sha256",
            "pairs_sha256", "tokenizer_spec_sha256", "case_manifest", "train_pairs")
    if any(actual[k] != registered[k] for k in keys):
        raise ValueError("Registered preflight evidence mismatch")
    for arm in ARMS:
        if actual["arms"][arm] != registered["arms"][arm]:
            raise ValueError("Registered training slots mismatch")


def train_arm(path, suite, slots, tokenizer, initial, config, *, source):
    path.mkdir(parents=True)
    by_id = {c.case_id: c for c in suite.cases}
    normal_ids = {c.case_id for c in suite.cases if c.slice_name == "normal"}
    model = _new_model(tokenizer, config)
    model.load_state_dict(initial)
    initial_hash = _state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    curves, prediction_rows, orders = [], [], []
    updates, tokens, padded = 0, 0, 0
    update_wall, eval_wall = 0.0, 0.0
    arm = DecisionFitArm(path.name, tuple(by_id), "shuffle")

    def evaluate(epoch):
        nonlocal eval_wall
        start = time.perf_counter()
        rng = torch.get_rng_state().clone()
        before = _state_digest(model.state_dict())
        predictions = _predict_cases(model, tokenizer, suite.cases, serializer=compact_input)
        metrics = _fit_metrics(predictions)
        common = _fit_metrics([p for p in predictions if p.case_id in normal_ids])
        if before != _state_digest(model.state_dict()) or not torch.equal(rng, torch.get_rng_state()):
            raise ValueError("Train evaluation changed weights/RNG")
        curves.append(dict(epoch=epoch, optimizer_updates=updates, example_exposures=updates,
                           metrics=metrics, common_normal_metrics=common))
        prediction_rows.extend(_prediction_rows(epoch=epoch, cases=suite.cases, predictions=predictions))
        _write_jsonl(path/"learning-curve.jsonl", curves)
        _write_jsonl(path/"epoch-predictions.jsonl", prediction_rows)
        eval_wall += time.perf_counter()-start

    try:
        evaluate(0)
        for epoch in range(1, config.epochs+1):
            start = time.perf_counter()
            order = torch.randperm(len(slots), generator=generator).tolist()
            model.train()
            online_losses = []
            for index in order:
                example = by_id[slots[index]].example
                lengths = [len(tokenizer.encode(compact_input(example.state, example.trajectory, a),
                                                add_bos=True, add_eos=True)) for a in example.candidates]
                tokens += sum(lengths)
                padded += len(lengths)*max(lengths)
                optimizer.zero_grad(set_to_none=True)
                loss = decision_loss(model, tokenizer, example, serializer=compact_input)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                online_losses.append(float(loss.detach()))
                updates += 1
            update_wall += time.perf_counter()-start
            orders.append(dict(epoch=epoch, slot_indices=order, case_ids=[slots[i] for i in order]))
            _write_jsonl(path/"epoch-order.jsonl", orders)
            evaluate(epoch)
            curves[-1]["online_update_loss_mean"] = sum(online_losses)/len(online_losses)
            _write_jsonl(path/"learning-curve.jsonl", curves)
            print(f"seed={config.seed} arm={path.name} epoch={epoch}/{config.epochs} train={curves[-1]['metrics']['correct']}/{len(suite.cases)}", flush=True)
        manifest = _save_fit_checkpoint(path/"checkpoint", model, tokenizer, suite=suite,
            suite_sha256=digest(suite.to_dict()), arm=arm, config=config,
            initial_model_sha256=initial_hash, source_git_sha=source["git_sha"], input_serializer_id=SERIALIZER)
        _write_json(path/"training-contract.json", dict(goal_contract=CURRENT_GOAL_V1,
                    input_serializer_id=SERIALIZER, slots=slots, slot_sha256=digest(slots),
                    checkpoint_id=manifest["checkpoint_id"]))
        reloaded, rt, rm = load_v1_checkpoint(path/"checkpoint")
        cached = [r for r in prediction_rows if r["epoch"] == config.epochs]
        current = _prediction_rows(epoch=config.epochs, cases=suite.cases,
                     predictions=_predict_cases(reloaded, rt, suite.cases, serializer=compact_input))
        if current != cached:
            raise ValueError("Reload train predictions mismatch")
        summary = dict(seed=config.seed, arm=path.name, unique=len(suite.cases), slots=len(slots),
            updates=updates, epochs=config.epochs, stable_fit=_stable_fit(curves, 3),
            final=curves[-1]["metrics"], common_normal=curves[-1]["common_normal_metrics"],
            initial_weight_sha256=initial_hash, order_sha256=digest([r["slot_indices"] for r in orders]),
            checkpoint_id=rm["checkpoint_id"], tokenizer_sha256=rm["tokenizer_sha256"],
            training_goal_contract=CURRENT_GOAL_V1, reload_match=True,
            tokens=tokens, padded_tokens=padded, train_wall_seconds=update_wall,
            train_eval_wall_seconds=eval_wall, parameter_count=sum(p.numel() for p in model.parameters()),
            process_max_rss_kib=_max_rss_kib())
        _write_json(path/"train-summary.json", summary)
        return reloaded, rt, summary
    except BaseException:
        torch.save({"state_dict":model.state_dict(), "optimizer_updates":updates,
                    "resume_supported":False}, path/"interrupted-model.pt")
        raise


def measure_dev(path, model, tokenizer, cases, pairs, train_cases, slots):
    before = _state_digest(model.state_dict())
    rng = torch.get_rng_state().clone()
    lookup = defaultdict(Counter)
    by_train = {c.case_id:c.example for c in train_cases}
    for case_id in slots:
        ex = by_train[case_id]
        lookup[candidate_signature(ex)][canonical(ex.target.to_dict())] += 1
    rows, episodes, permutation_max_delta = [], [], 0.0
    for case in cases:
        result = score(model, tokenizer, case["example"])
        reverse = score(model, tokenizer, case["example"], reverse=True)
        delta = max(abs(a-b) for a,b in zip(result["scores"],reversed(reverse["scores"]),strict=True))
        permutation_max_delta = max(permutation_max_delta,delta)
        if delta > 1e-5 or result["selected"] != reverse["selected"]:
            raise ValueError("Candidate permutation mismatch")
        counter = lookup[candidate_signature(case["example"])]
        winner = min(counter,key=lambda a:(-counter[a],a)) if counter else None
        rows.append({**{k:v for k,v in case.items() if k != "example"}, **result,
                     "candidate_lookup_covered":bool(counter),
                     "candidate_lookup_correct":winner==canonical(case["example"].target.to_dict())})
        episodes.append(dict(case_id=case["case_id"], group=case["group"], variant=case["variant"],
                             **rollout(model, tokenizer, case["example"])))
    summary, pair_rows = summarize(rows, episodes, cases, pairs)
    # normal phase0 cases are exactly the 8 registered initial task variants.
    initial = [e for c,e in zip(cases, episodes, strict=True) if c["group"]=="normal" and c["factors"]["phase"]==0]
    summary["initial"] = {variant:dict(count=sum(e["variant"]==variant for e in initial),
                         success=sum(e["variant"]==variant and e["success"] is True for e in initial),
                         invalid=sum(e["variant"]==variant and bool(e["invalid"]) for e in initial))
                         for variant in ("seen","numbers","key","joint")}
    summary["candidate_lookup"] = dict(covered=sum(r["candidate_lookup_covered"] for r in rows),
                                      correct=sum(r["candidate_lookup_correct"] for r in rows))
    summary["nonoverlap_decisions"] = dict(count=sum(not r["overlap_union_train"] for r in rows),
               correct=sum(not r["overlap_union_train"] and r["correct"] for r in rows))
    summary["weights_unchanged"] = _state_digest(model.state_dict())==before
    summary["permutation_max_abs_score_delta"] = permutation_max_delta
    summary["rng_unchanged"] = torch.equal(rng,torch.get_rng_state())
    if not summary["weights_unchanged"] or not summary["rng_unchanged"]:
        raise ValueError("Dev changed weights/RNG")
    _write_jsonl(path/"dev-predictions.jsonl", rows)
    _write_jsonl(path/"prefix-episodes.jsonl", episodes)
    _write_jsonl(path/"dev-pairs.jsonl", pair_rows)
    _write_json(path/"dev-summary.json", summary)
    return summary


def run_study(root, registered_path, *, source, debug=False, deadline_seconds=1800):
    registered = json.loads(Path(registered_path).read_text())
    cases, slots = build_training()
    dev, pairs = build_cases()
    pre = audit(cases, slots, dev, pairs)
    validate_registered(pre, registered)
    epochs, seeds = (1, (42,)) if debug else (25, (42,43,44))
    tokenizer = build_byte_decision_tokenizer()
    protocol = dict(schema_version=SCHEMA, source=source, debug=debug, seeds=seeds, epochs=epochs,
                    goal_contract=CURRENT_GOAL_V1, preflight=pre, deadline_seconds=deadline_seconds,
                    lr=0.001, threads=torch.get_num_threads(), test="sealed/not_constructed/not_scored")
    run = RunLog(Path(root), "decision_mixed_v1_study", protocol, producer="trainer")
    start = time.perf_counter()
    _write_json(run.path/"protocol.json",protocol)
    _write_json(run.path/"slots.json",slots)
    _write_json(run.path/"pairs.json",pairs)
    _write_jsonl(run.path/"train-cases.jsonl",[{**{k:v for k,v in c.items() if k!='example'},"example":c["example"].to_dict()} for c in cases])
    _write_jsonl(run.path/"dev-cases.jsonl",[{**{k:v for k,v in c.items() if k!='example'},"example":c["example"].to_dict()} for c in dev])
    previous = signal.getsignal(signal.SIGALRM)
    def expire(signum, frame):
        raise TimeoutError("Arm training/evaluation deadline exceeded")
    results = []
    try:
        signal.signal(signal.SIGALRM,expire)
        for seed in seeds:
            config = DecisionFitStudyConfig(seed=seed,epochs=epochs,learning_rate=0.001)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                initial_model = _new_model(tokenizer,config)
                initial = {k:v.detach().clone() for k,v in initial_model.state_dict().items()}
            for name in ARMS:
                path = run.path/f"seed-{seed}"/name
                started = time.perf_counter()
                signal.setitimer(signal.ITIMER_REAL,deadline_seconds)
                selected = set(slots[name])
                train_cases = tuple(BenchmarkCase(c["case_id"], "train", c["group"], c["leakage_group"],c["example"])
                                    for c in cases if c["case_id"] in selected)
                suite = BenchmarkSuite(f"calculate-store-mixed-v1-{name}",train_cases)
                try:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        model, tk, fit = train_arm(path, suite, slots[name], tokenizer, initial, config, source=source)
                    measured = measure_dev(path,model,tk,dev,pairs,train_cases,slots[name])
                    # A second reload checks all fixed prefix logits after rollout.
                    re, rt, _ = load_v1_checkpoint(path/"checkpoint")
                    cached = [json.loads(line) for line in (path/"dev-predictions.jsonl").read_text().splitlines()]
                    if any(score(re,rt,c["example"])["scores"] != r["scores"] for c,r in zip(dev,cached,strict=True)):
                        raise ValueError("Dev reload mismatch")
                    result = dict(seed=seed, arm=name, status="completed",fit=fit,dev=measured,
                                  dev_reload_match=True,wall_seconds=time.perf_counter()-started)
                except (TimeoutError, ValueError, RuntimeError) as error:
                    result = dict(seed=seed,arm=name,status="failed",error=str(error),wall_seconds=time.perf_counter()-started)
                    _write_json(path/"failure.json", result)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                results.append(result)
                _write_json(run.path/"partial-results.json",results)
                run.event("arm_finished",dict(seed=seed,arm=name,status=result["status"]))
                print(f"DONE seed={seed} arm={name} status={result['status']}",flush=True)
            completed = [r for r in results if r["seed"]==seed and r["status"]=="completed"]
            if len(completed)==2 and any(completed[0]["fit"][k]!=completed[1]["fit"][k] for k in ("initial_weight_sha256","order_sha256")):
                raise ValueError("Paired initialization/order mismatch")
        common_prefix = []
        for seed in seeds:
            arms = [r for r in results if r["seed"]==seed and r["status"]=="completed"]
            if len(arms)!=2:
                continue
            episodes = [[json.loads(line) for line in (run.path/f"seed-{seed}"/n/"prefix-episodes.jsonl").read_text().splitlines()] for n in ARMS]
            valid = [i for i in range(128) if all(e[i]["invalid"] is None for e in episodes)]
            common_prefix.append(dict(seed=seed,common_valid=len(valid),all_started=128,
                           success={n:sum(e[i]["success"] is True for i in valid) for n,e in zip(ARMS,episodes,strict=True)}))
        study = dict(schema_version=SCHEMA, run_id=run.id, source=source, debug=debug,
                     registered_preflight_sha256=digest(registered), models=results,
                     common_prefix=common_prefix,wall_seconds=time.perf_counter()-start,
                     python=platform.python_version(),torch=torch.__version__,threads=torch.get_num_threads(),
                     independent_holdout=False,test_scored=0,preflight=pre)
        _write_json(run.path/"study.json",study)
        run.finish("completed" if all(r["status"]=="completed" for r in results) else "failed",study="study.json")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return run.path
