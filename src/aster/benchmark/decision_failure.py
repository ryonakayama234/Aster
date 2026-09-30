"""Audit cached frozen-model errors and tokenization; do not run new inference."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from aster.benchmark.state_diagnostics import _digest, _suite_digest, build_state_diagnostic_suite, action_label
from aster.corpus.pipeline import digest
from aster.records.decision import serialize_decision_input
from aster.records.runlog import RunLog
from aster.training.decision_data import anchor_cases
from aster.training.decision_fit import _serialized_texts, _write_json, _write_jsonl, load_decision_fit_checkpoint
from aster.records.transition import Action


def audit_cached_failures(root: str | Path, diagnostic_run: str | Path, checkpoint: str | Path) -> Path:
    """Validate cached provenance, then compare frozen tokenizer use with train."""
    source = Path(diagnostic_run)
    report = json.loads((source / "diagnostics.json").read_text())
    _, tokenizer, manifest = load_decision_fit_checkpoint(checkpoint)
    suite = build_state_diagnostic_suite()
    if report["checkpoint_id"] != manifest["checkpoint_id"] or report["suite_sha256"] != _suite_digest(suite):
        raise ValueError("Cached diagnostic/checkpoint/suite provenance mismatch")
    payload = (source / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {c.case_id for c in suite.cases}:
        raise ValueError("Cached prediction cases do not match suite")
    train_ids = {i for text in _serialized_texts(anchor_cases())
                 for i in tokenizer.encode(text, add_bos=True, add_eos=True)}
    results = []
    confusion = defaultdict(Counter)
    correct = Counter()
    for case in suite.cases:
        row = by_id[case.case_id]
        example = case.example
        texts = [serialize_decision_input(example.state, example.trajectory, a) for a in example.candidates]
        if row["case_sha256"] != _digest(case.to_dict()) or row["input_sha256"] != [_digest(t) for t in texts]:
            raise ValueError("Cached input digest mismatch")
        tokens = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
        if [len(ids) for ids in tokens] != row["token_lengths"]:
            raise ValueError("Cached tokenizer lengths mismatch")
        if row["target_action"] != example.target.to_dict() or row["model_action"] not in [a.to_dict() for a in example.candidates]:
            raise ValueError("Cached action/candidate mismatch")
        is_correct = row["model_action"] == row["target_action"]
        if is_correct != row["model_correct"]:
            raise ValueError("Cached correctness mismatch")
        validate_cached_scores(row, example)
        selected = Action(**row["model_action"])
        if is_correct:
            correct[case.slice_name] += 1
        else:
            confusion[case.slice_name][action_label(example.target) + " -> " + action_label(selected)] += 1
        ids = tokens[example.target_index]
        results.append({"case_id": case.case_id, "slice": case.slice_name,
            "model_correct": is_correct, "target_action": row["target_action"], "model_action": row["model_action"],
            "target_input": texts[example.target_index], "target_token_ids": ids,
            "target_token_pieces": ["<bos>" if i == tokenizer.bos_id else "<eos>" if i == tokenizer.eos_id
                                     else tokenizer.model.vocab[i].decode("utf-8", errors="backslashreplace") for i in ids],
            "target_token_length": len(ids), "unseen_in_anchor_token_occurrences": sum(i not in train_ids for i in ids)})
    summary = {"schema_version": "aster-decision-failure-audit-0", "source_run_id": source.name,
        "checkpoint_id": manifest["checkpoint_id"], "tokenizer_sha256": manifest["tokenizer_sha256"],
        "prediction_file_sha256": digest(payload), "suite_sha256": _suite_digest(suite),
        "correct_by_slice": dict(correct), "error_confusion": {k: dict(v) for k, v in confusion.items()},
        "phase_zero": [r for r in results if r["case_id"].endswith("phase-0/delay-0")],
        "token_length_ranges": {name: [min(r["target_token_length"] for r in results if r["slice"] == name),
                                       max(r["target_token_length"] for r in results if r["slice"] == name)]
                                for name in {r["slice"] for r in results}},
        "longest_vocabulary_token_bytes": max(map(len, tokenizer.model.vocab.values())),
        "interpretation": "Token changes and errors co-occur; this is not a causal tokenizer intervention.",
        "model_update": False, "new_inference": False, "test": "sealed"}
    run = RunLog(Path(root), "decision_failure_audit", {"source_run_id": source.name,
        "checkpoint_id": manifest["checkpoint_id"], "prediction_file_sha256": digest(payload)}, producer="evaluator")
    _write_json(run.path / "audit.json", summary)
    _write_jsonl(run.path / "token-cases.jsonl", results)
    run.finish("completed", audit="audit.json", token_cases="token-cases.jsonl", test_status="sealed")
    return run.path


def validate_cached_scores(row, example) -> None:
    """Recompute cached choices and numeric diagnostics without model inference."""
    candidates = [a.to_dict() for a in example.candidates]
    logits = row.get("logits")
    if row.get("candidates") != candidates or row.get("target_index") != example.target_index:
        raise ValueError("Cached candidate order/target index mismatch")
    if (not isinstance(logits, list) or len(logits) != len(candidates)
            or not all(type(v) in (int, float) and math.isfinite(v) for v in logits)):
        raise ValueError("Cached logits must be finite and match candidates")
    selected = max(range(len(logits)), key=lambda i: logits[i])
    if row.get("model_action") != candidates[selected]:
        raise ValueError("Cached model action does not match logits argmax")
    target = example.target_index
    maximum = max(logits)
    nll = maximum - logits[target] + math.log(sum(math.exp(v-maximum) for v in logits))
    margin = logits[target] - max(v for i, v in enumerate(logits) if i != target)
    for field, expected in (("nll", nll), ("target_probability", math.exp(-nll)),
                            ("target_margin", margin)):
        actual = row.get(field)
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)):
            raise ValueError(f"Cached {field} does not match logits")
