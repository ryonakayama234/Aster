"""Frozen v0 checkpoints on explicitly selected current-state v1 dev probes."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import signal
import time
from typing import Any

import torch

from aster.agent.candidates import CalculateAndStoreCandidates
from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.state_representation import canonical, compact_input, digest, executor
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.goal_contract import CURRENT_GOAL_V1, EVER_GOAL_V0
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import score_candidates
from aster.records.decision import DecisionExample
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext
from aster.training.decision_data import development_tasks
from aster.training.decision_fit import load_decision_fit_checkpoint, _state_digest
from aster.tokenizer.artifact import load_tokenizer

SCHEMA = "aster-decision-fixed-v1-diagnostics-0"
SERIALIZER = "aster-decision-compact-input-0"


class Session:
    def __init__(self, task):
        self.context = RuntimeContext(task)
        self.recorder = TrajectoryRecorder()
        self.tools = executor()
        self.available = self.tools.registry.names() + ("stop",)
        self.teacher = RuleBasedPolicy(goal_contract=CURRENT_GOAL_V1)
        self.builder = CalculateAndStoreCandidates(goal_contract=CURRENT_GOAL_V1)
        self.evaluator = TaskEvaluator(goal_contract=CURRENT_GOAL_V1)

    def execute(self, action):
        history = self.recorder.trajectory()
        before = self.context.snapshot(len(history))
        obs = (Observation(True, True, {"stopped": True, "reason": action.arguments.get("reason", "policy_stop")})
               if action.kind == "stop" else self.tools.execute(action, self.context))
        after = self.context.snapshot(len(history) + 1)
        transition = Transition(len(history), before.to_dict(), self.available, action, obs,
                               after.to_dict(), self.evaluator.evaluate(self.context.task, after, action, obs, history))
        self.recorder.record(transition)
        return transition

    def example(self):
        history = self.recorder.trajectory()
        state = self.context.snapshot(len(history))
        candidates = self.builder.build(state, history, self.available)
        target = self.teacher.decide(state, history, self.available)
        if target not in candidates:
            raise ValueError("v1 target missing from candidates")
        return DecisionExample(state, history, candidates, candidates.index(target), teacher="rule:current-v1")

    def teacher_step(self):
        return self.execute(self.example().target)


def replay(example):
    session = Session(example.state.task)
    for saved in example.trajectory.transitions:
        if session.execute(saved.action).to_dict() != saved.to_dict():
            raise ValueError("v1 prefix replay mismatch")
    actual = session.example()
    if actual != example:
        raise ValueError("v1 state/candidates/teacher replay mismatch")
    return session


def signature(example):
    return digest(sorted(compact_input(example.state, example.trajectory, a) for a in example.candidates))


def candidate_signature(example):
    return canonical(sorted(canonical(a.to_dict()) for a in example.candidates))


def build_cases():
    cases, pairs = [], []
    for operation, variant, task in development_tasks():
        task_id = f"{operation}/{variant}"
        def add(suffix, session, group, **factors):
            row = dict(case_id=f"{task_id}/{suffix}", task_id=task_id, variant=variant,
                       group=group, factors=factors, example=session.example())
            cases.append(row)
            return row["case_id"]
        normal_ids = []
        for phase in range(4):
            session = Session(task)
            for _ in range(phase):
                session.teacher_step()
            normal_ids.append(add(f"normal-{phase}", session, "normal", phase=phase))
        for phase in (0, 1):
            for value in (-999, -777):
                for writes in (1, 2):
                    session = Session(task)
                    if phase:
                        session.teacher_step()
                    for _ in range(writes):
                        session.execute(Action.tool("memory.put", key=task["store_as"], value=value))
                    case_id = add(f"wrong-p{phase}-v{value}-n{writes}", session, "factorial",
                                  phase=phase, wrong_value=value, writes=writes)
                    if phase:
                        pairs.append(dict(left=case_id, right=normal_ids[2], same_step=writes == 1))
        for mutation in ("wrong_overwrite", "same_value_write", "unrelated_write", "repair_unconfirmed"):
            session = Session(task)
            for _ in range(3):
                session.teacher_step()
            key = task["store_as"]
            if not isinstance(key, str):
                raise ValueError("Task key must be a string")
            value = session.context.memory[key]
            if mutation == "unrelated_write":
                session.execute(Action.tool("memory.put", key="fixed_v1_noise", value=-999))
            elif mutation == "same_value_write":
                session.execute(Action.tool("memory.put", key=key, value=value))
            else:
                session.execute(Action.tool("memory.put", key=key, value=-999))
                if mutation == "repair_unconfirmed":
                    session.execute(Action.tool("memory.put", key=key, value=value))
            add(mutation, session, "completion", mutation=mutation)
    if len(cases) != 128 or len(pairs) != 32:
        raise ValueError("Preregistered case count mismatch")
    return cases, pairs


def rule_rollout(example):
    session = replay(example)
    for _ in range(8):
        if session.teacher_step().evaluation.terminal:
            break
    return evaluate_episode(session.recorder.trajectory()).task_success


def train_evidence(bundle):
    """Read only the six source arms and their train snapshots, never old/reserved tests."""
    records, union, input_targets = [], set(), defaultdict(set)
    for run in sorted((Path(bundle) / "runs").iterdir()):
        protocol = json.loads((run / "protocol.json").read_text())
        seed = protocol["config"]["seed"]
        for arm in ("baseline", "state"):
            suite = json.loads((run / f"{arm}-suite.json").read_text())
            checkpoint = run / "arms" / arm / "checkpoint"
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            if digest(suite) != manifest["suite_sha256"]:
                raise ValueError("Source training suite digest mismatch")
            examples, signatures, lookup = [], set(), defaultdict(Counter)
            for case in suite["cases"]:
                if case["split"] != "train":
                    raise ValueError("Non-train source snapshot")
                example = DecisionExample.from_dict(case["example"])
                if example.trajectory.goal_contract != EVER_GOAL_V0:
                    raise ValueError("Expected old-contract source training")
                sig = signature(example)
                signatures.add(sig)
                union.add(sig)
                input_targets[sig].add(canonical(example.target.to_dict()))
                lookup[candidate_signature(example)][canonical(example.target.to_dict())] += 1
                examples.append((case["case_id"], example))
            cached = [json.loads(line) for line in (run / "arms" / arm / "epoch-predictions.jsonl").read_text().splitlines()]
            last_epoch = max(r["epoch"] for r in cached)
            cached = {r["case_id"]: r for r in cached if r["epoch"] == last_epoch}
            records.append(dict(seed=seed, arm=arm, source_run_id=run.name, checkpoint=checkpoint,
                                train_signatures=signatures, train_examples=examples, lookup=lookup, cached=cached))
    if sorted((r["seed"], r["arm"]) for r in records) != [(s,a) for s in (42,43,44) for a in ("baseline","state")]:
        raise ValueError("Expected exactly six source checkpoints")
    return records, union, input_targets


def preflight(cases, pairs, union, input_targets) -> dict[str, Any]:
    by_id, signatures = {c["case_id"]: c for c in cases}, defaultdict(set)
    for case in cases:
        example = case["example"]
        replay(example)
        if not rule_rollout(example):
            raise ValueError("Rule v1 continuation failed")
        sig = signature(example)
        target = canonical(example.target.to_dict())
        signatures[sig].add(target)
        if sig in input_targets and input_targets[sig] != {target}:
            raise ValueError("Same model input has different old/new teacher targets")
        case["signature"] = sig
        case["overlap_union_train"] = sig in union
    if any(len(v) != 1 for v in signatures.values()):
        raise ValueError("v1 compact target collision")
    for pair in pairs:
        left, right = (by_id[pair[k]]["example"] for k in ("left","right"))
        if candidate_signature(left) != candidate_signature(right) or left.target == right.target:
            raise ValueError("Main pair must have same candidates and different target")
        if (left.state.step == right.state.step) != pair["same_step"]:
            raise ValueError("Pair step relation mismatch")
        pair["nonoverlap"] = not any(by_id[pair[k]]["overlap_union_train"] for k in ("left","right"))
    groups = defaultdict(Counter)
    for case in cases:
        groups[candidate_signature(case["example"])][canonical(case["example"].target.to_dict())] += 1
    return dict(cases=len(cases), rule_success=len(cases), candidate_coverage=len(cases), input_collisions=0,
                train_overlap=sum(c["overlap_union_train"] for c in cases), primary_pairs=len(pairs),
                same_step_primary_pairs=sum(p["same_step"] for p in pairs),
                candidate_only_oracle_correct=sum(max(v.values()) for v in groups.values()),
                candidate_only_oracle_kind="in-sample information ceiling; not learned accuracy",
                independent_holdout=False, test_constructed=0, test_scored=0)


def score(model, tokenizer, example, *, reverse=False, serializer=compact_input) -> dict[str, Any]:
    candidates = tuple(reversed(example.candidates)) if reverse else example.candidates
    texts = [serializer(example.state, example.trajectory, a) for a in candidates]
    ids = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
    if any(tokenizer.decode(v) != t for v,t in zip(ids,texts,strict=True)):
        raise ValueError("Tokenizer round-trip mismatch")
    if max(map(len,ids)) > model.backbone.config.context_length:
        return dict(invalid="context_exceeded", max_tokens=max(map(len,ids)))
    with torch.no_grad():
        scores = score_candidates(model,tokenizer,example.state,example.trajectory,candidates,serializer=serializer)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite scores")
    values = scores.tolist()
    index = int(scores.argmax())
    target_index = candidates.index(example.target)
    return dict(invalid=None, scores=values, candidates=[a.to_dict() for a in candidates],
                selected=candidates[index].to_dict(), target=example.target.to_dict(),
                correct=candidates[index] == example.target,
                margin=(values[target_index]-max(v for i,v in enumerate(values) if i != target_index)
                        if len(values) > 1 else None),
                inputs=texts, token_ids=ids, input_sha256=digest(texts), max_tokens=max(map(len,ids)))


def rollout(model, tokenizer, example, *, serializer=compact_input):
    session = replay(example)
    rows, invalid, first_error = [], None, None
    for _ in range(8):
        visited = session.example()
        row = score(model, tokenizer, visited, serializer=serializer)
        if row["invalid"]:
            invalid = row["invalid"]
            break
        row["step"] = visited.state.step
        row["state"] = visited.state.to_dict()
        rows.append(row)
        if first_error is None and not row["correct"]:
            first_error = row
        last = session.execute(Action.from_dict(row["selected"]))
        if last.evaluation.terminal:
            break
    trajectory = session.recorder.trajectory()
    summary = evaluate_episode(trajectory).to_dict()
    # Verify persistence representation without rewriting old checkpoint or evidence.
    restored = type(trajectory)(tuple(Transition.from_dict(t.to_dict()) for t in trajectory.transitions))
    if restored != trajectory or evaluate_episode(restored).to_dict() != summary:
        raise ValueError("v1 trajectory persistence mismatch")
    return dict(success=None if invalid else summary["task_success"], invalid=invalid,
                goal_contract=trajectory.goal_contract, summary=summary, prefix_length=len(example.trajectory),
                first_error=first_error, decisions=rows, fallback=False,
                trajectory=[t.to_dict() for t in trajectory.transitions])


def summarize(rows, episodes, cases, pairs):
    by_id = {r["case_id"]:r for r in rows}
    pair_rows = [{**p,"both_correct":by_id[p["left"]]["correct"] and by_id[p["right"]]["correct"]} for p in pairs]
    result = {}
    for name, selected in [(g,[c["case_id"] for c in cases if c["group"]==g]) for g in ("normal","factorial","completion")]:
        chosen = [r for r in rows if r["case_id"] in selected]
        chosen_e = [e for e in episodes if e["case_id"] in selected]
        result[name] = dict(count=len(chosen),correct=sum(r["correct"] for r in chosen),
                            invalid=sum(r["invalid"] is not None for r in chosen),
                            prefix_success=sum(e["success"] is True for e in chosen_e),
                            prefix_invalid=sum(e["invalid"] is not None for e in chosen_e))
    for key in ("variant", "overlap_union_train"):
        result[key] = {str(v):dict(count=sum(r[key]==v for r in rows),correct=sum(r[key]==v and r["correct"] for r in rows))
                       for v in sorted({r[key] for r in rows},key=str)}
    result["factorial_cells"] = {
        f"phase{p}/value{v}/writes{n}":dict(count=8,correct=sum(r["correct"] for r in rows if r["group"]=="factorial" and r["factors"]==dict(phase=p,wrong_value=v,writes=n)))
        for p in (0,1) for v in (-999,-777) for n in (1,2)}
    result["normal_phases"] = {str(p):dict(count=8,correct=sum(r["correct"] for r in rows if r["group"]=="normal" and r["factors"]["phase"]==p)) for p in range(4)}
    result["completion_mutations"] = {m:dict(count=8,correct=sum(r["correct"] for r in rows if r["group"]=="completion" and r["factors"]["mutation"]==m))
                                     for m in ("wrong_overwrite","same_value_write","unrelated_write","repair_unconfirmed")}
    result["pairs"] = dict(count=len(pair_rows),both_correct=sum(p["both_correct"] for p in pair_rows),
                           nonoverlap_count=sum(p["nonoverlap"] for p in pair_rows),
                           nonoverlap_both_correct=sum(p["nonoverlap"] and p["both_correct"] for p in pair_rows))
    return result,pair_rows


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")


def write_lines(path, rows):
    path.write_text("".join(canonical(r)+"\n" for r in rows))


def run_diagnostics(root, bundle, *, source=None, deadline_seconds=1800):
    started = time.perf_counter()
    cases,pairs = build_cases()
    records,union,input_targets = train_evidence(bundle)
    audit = preflight(cases,pairs,union,input_targets)
    # All fixed inputs are length-checked for every arm before any model scoring.
    lengths = []
    for record in records:
        manifest = json.loads((record["checkpoint"] / "manifest.json").read_text())
        tokenizer = load_tokenizer(record["checkpoint"] / "tokenizer")
        maximum = 0
        for case in cases:
            example = case["example"]
            for action in example.candidates:
                text = compact_input(example.state,example.trajectory,action)
                ids = tokenizer.encode(text,add_bos=True,add_eos=True)
                if tokenizer.decode(ids) != text or len(ids) > manifest["model_config"]["context_length"]:
                    raise ValueError("Fixed input fails premeasurement length/roundtrip audit")
                maximum = max(maximum,len(ids))
        lengths.append(dict(seed=record["seed"],arm=record["arm"],max_tokens=maximum))
    audit["fixed_input_lengths"] = lengths
    protocol = dict(schema_version=SCHEMA,goal_contract=CURRENT_GOAL_V1,input_serializer_id=SERIALIZER,
                    training_goal_contract=EVER_GOAL_V0,updates=0,source=source,
                    suite_sha256=digest([{**{k:v for k,v in c.items() if k!='example'},"example":c["example"].to_dict()} for c in cases]),
                    deadline_seconds_per_model=deadline_seconds,independent_holdout=False,test="sealed/not_constructed/not_scored",
                    checkpoints=[dict(seed=r["seed"],arm=r["arm"],source_run_id=r["source_run_id"],
                                      manifest_sha256=hashlib.sha256((r["checkpoint"]/"manifest.json").read_bytes()).hexdigest()) for r in records])
    run = RunLog(Path(root),"decision_fixed_v1_diagnostics",protocol,producer="evaluator")
    previous = signal.getsignal(signal.SIGALRM)
    def expire(signum, frame):
        raise TimeoutError("Fixed v1 model diagnosis exceeded deadline")
    results=[]
    try:
        write_json(run.path/"protocol.json",protocol)
        write_json(run.path/"preflight.json",audit)
        write_json(run.path/"pairs.json",pairs)
        write_lines(run.path/"cases.jsonl",[{**{k:v for k,v in c.items() if k!='example'},"example":c["example"].to_dict()} for c in cases])
        signal.signal(signal.SIGALRM,expire)
        for record in records:
            signal.setitimer(signal.ITIMER_REAL,deadline_seconds)
            arm_started=time.perf_counter()
            checkpoint=record["checkpoint"]
            files={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.rglob('*') if p.is_file()}
            model,tokenizer,manifest=load_decision_fit_checkpoint(checkpoint,expected_input_serializer_id=SERIALIZER)
            weight_before=_state_digest(model.state_dict())
            rng=torch.get_rng_state().clone()
            # Cached old-train controls assess runtime drift without relabelling them v1.
            cache_rows=[]
            for case_id,example in record["train_examples"]:
                actual=score(model,tokenizer,example)
                saved=record["cached"][case_id]
                if actual["invalid"] or actual["selected"] != saved["predicted_action"]:
                    raise ValueError("Old cached train Action differs in current runtime")
                delta=max(abs(a-b) for a,b in zip(actual["scores"],saved["logits"],strict=True))
                if delta>1e-4:
                    raise ValueError("Old cached train logits drift exceeds 1e-4")
                cache_rows.append(dict(case_id=case_id,max_abs_logit_delta=delta))
            rows,episodes=[],[]
            for case in cases:
                example=case["example"]
                prediction=score(model,tokenizer,example)
                if prediction["invalid"]:
                    raise ValueError("Fixed prefix exceeds context before measurement")
                reversed_prediction=score(model,tokenizer,example,reverse=True)
                if prediction["selected"] != reversed_prediction["selected"]:
                    raise ValueError("Candidate permutation changed selected Action")
                if any(abs(a-b)>1e-5 for a,b in zip(prediction["scores"],reversed(reversed_prediction["scores"]),strict=True)):
                    raise ValueError("Candidate permutation changed scores")
                counter=record["lookup"].get(candidate_signature(example),Counter())
                winner=min(counter,key=lambda a:(-counter[a],a)) if counter else None
                rows.append({**{k:v for k,v in case.items() if k!='example'},**prediction,
                             "overlap_own_train":case["signature"] in record["train_signatures"],
                             "candidate_lookup_covered":bool(counter),
                             "candidate_lookup_correct":winner==canonical(example.target.to_dict())})
                episodes.append(dict(case_id=case["case_id"],group=case["group"],variant=case["variant"],**rollout(model,tokenizer,example)))
            # Separate reload compares all frozen prefix score vectors in this run.
            reloaded,rt,_=load_decision_fit_checkpoint(checkpoint,expected_input_serializer_id=SERIALIZER)
            for case,row in zip(cases,rows,strict=True):
                re=score(reloaded,rt,case["example"])
                if re["scores"]!=row["scores"]:
                    raise ValueError("Reload prefix score mismatch")
            if weight_before!=_state_digest(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
                raise ValueError("Evaluation changed weights/RNG")
            if files!={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.rglob('*') if p.is_file()}:
                raise ValueError("Source artifact files changed")
            summary,pair_rows=summarize(rows,episodes,cases,pairs)
            summary.update(seed=record["seed"],arm=record["arm"],checkpoint_id=manifest["checkpoint_id"],
                           model_state_sha256=manifest["model_state_sha256"],tokenizer_sha256=manifest["tokenizer_sha256"],
                           cached_train_max_logit_delta=max(r["max_abs_logit_delta"] for r in cache_rows),
                           candidate_lookup_correct=sum(r["candidate_lookup_correct"] for r in rows),
                           candidate_lookup_covered=sum(r["candidate_lookup_covered"] for r in rows),
                           weights_unchanged=True,rng_unchanged=True,reload_match=True,source_artifact_unchanged=True,
                           candidate_permutation_match=True,wall_seconds=time.perf_counter()-arm_started)
            output=run.path/f"seed-{record['seed']}-{record['arm']}"
            output.mkdir()
            write_lines(output/"predictions.jsonl",rows)
            write_lines(output/"episodes.jsonl",episodes)
            write_lines(output/"pairs.jsonl",pair_rows)
            write_json(output/"old-train-runtime-check.json",cache_rows)
            write_json(output/"summary.json",summary)
            results.append(summary)
            run.event("model_completed",dict(seed=record["seed"],arm=record["arm"]))
            print(f"seed={record['seed']} arm={record['arm']} normal={summary['normal']['correct']}/32 factorial={summary['factorial']['correct']}/64 completion={summary['completion']['correct']}/32",flush=True)
            signal.setitimer(signal.ITIMER_REAL,0)
        study=dict(**protocol,run_id=run.id,preflight=audit,models=results,python=platform.python_version(),
                   torch=torch.__version__,threads=torch.get_num_threads(),wall_seconds=time.perf_counter()-started)
        write_json(run.path/"study.json",study)
        run.finish("completed",study="study.json")
    except BaseException as error:
        run.finish("interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return run.path
