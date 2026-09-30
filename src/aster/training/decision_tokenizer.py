"""Controlled BPE-vs-byte DecisionModel comparison for Issue #28.

The comparison holds the 32 unique numbers-and-keys decisions fixed and changes
only tokenizer/input representation. Reserved test suites are constructed for split
integrity but are never tokenized or scored by this module.
"""
from __future__ import annotations

from dataclasses import asdict
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
from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from aster.benchmark.state_diagnostics import fit_step_lookup, measure_state_diagnostics
from aster.benchmark.suite import _executor
from aster.corpus.pipeline import digest, json_bytes
from aster.evaluator.verifier import TaskEvaluator
from aster.inference.decide import score_candidates
from aster.model.decision_head import DecisionModel
from aster.records.decision import serialize_decision_input
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Transition
from aster.runtime.context import RuntimeContext
from aster.tokenizer.artifact import (
    AsterTokenizer,
    TokenizerManifest,
    load_tokenizer,
    save_tokenizer,
)
from aster.tokenizer.bpe import BPEModel
from aster.training.decision_data import (
    anchor_cases,
    build_development_probes,
    build_sealed_prefix_suite,
    build_training_intervention,
    development_tasks,
    example_signature,
    validate_partition,
)
from aster.training.decision_fit import (
    DecisionFitArm,
    DecisionFitStudyConfig,
    _audit_train_cases,
    _directory_digest,
    _max_rss_kib,
    _new_model,
    _run_arm,
    _state_digest,
    _write_json,
    _write_jsonl,
    load_decision_fit_checkpoint,
)

SCHEMA = "aster-decision-tokenizer-comparison-0"
EXPECTED_BPE_SHA256 = "d68bd292d060625256cdcf16398d70db7d379542b8fe8aea45cbc7da611f9692"
MEASUREMENT_CONFIG = {
    "epochs": 50,
    "learning_rate": 1e-3,
    "seed_set": (42, 43, 44),
    "context_length": 2048,
    "width": 16,
    "heads": 2,
    "layers": 1,
    "stability_window": 3,
}


def build_byte_decision_tokenizer() -> AsterTokenizer:
    """Wrap UTF-8 byte IDs in the artifact interface used by DecisionModel."""
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    special_tokens = {"<|bos|>": 256, "<|eos|>": 257}
    return AsterTokenizer(
        model=BPEModel(merges=[], vocab=vocab),
        special_tokens=special_tokens,
        manifest=TokenizerManifest(
            name="AsterDecisionByteTokenizer",
            version="0",
            tokenizer_type="byte",
            encoding="utf-8",
            base_vocab_size=256,
            target_vocab_size=256,
            actual_bpe_vocab_size=256,
            special_token_count=2,
            min_pair_frequency=0,
        ),
    )


def build_comparison_training_suite() -> tuple[BenchmarkSuite, tuple[str, ...]]:
    """Return the exact 32 unique numbers-and-keys decisions from the #27 protocol."""
    source, arms = build_training_intervention()
    selected = next(arm for arm in arms if arm.arm_id == "numbers-and-keys")
    selected_ids = set(selected.case_ids)
    cases = tuple(case for case in source.cases if case.case_id in selected_ids)
    if len(cases) != 32 or len({example_signature(case.example) for case in cases}) != 32:
        raise RuntimeError("Tokenizer comparison requires exactly 32 unique decisions")
    return BenchmarkSuite("calculate-store-tokenizer-comparison-train-v0", cases), tuple(
        case.case_id for case in cases
    )


def _protocol_status(config: DecisionFitStudyConfig) -> str:
    exact = (
        config.epochs == MEASUREMENT_CONFIG["epochs"]
        and config.learning_rate == MEASUREMENT_CONFIG["learning_rate"]
        and config.seed in MEASUREMENT_CONFIG["seed_set"]
        and config.context_length == MEASUREMENT_CONFIG["context_length"]
        and config.width == MEASUREMENT_CONFIG["width"]
        and config.heads == MEASUREMENT_CONFIG["heads"]
        and config.layers == MEASUREMENT_CONFIG["layers"]
        and config.stability_window == MEASUREMENT_CONFIG["stability_window"]
    )
    return "preregistered-measurement" if exact else "debug/non-preregistered"


def _serialized_protocol(cases: tuple[BenchmarkCase, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        texts = [
            serialize_decision_input(case.example.state, case.example.trajectory, candidate)
            for candidate in case.example.candidates
        ]
        rows.append(
            {
                "case_id": case.case_id,
                "state_step": case.example.state.step,
                "candidate_texts": texts,
                "candidates": [candidate.to_dict() for candidate in case.example.candidates],
                "target_index": case.example.target_index,
                "target": case.example.target.to_dict(),
            }
        )
    return rows


def _protocol_digests(
    train: BenchmarkSuite,
    probes: BenchmarkSuite,
    partition: dict[str, Any],
) -> dict[str, str]:
    serialized = _serialized_protocol(train.cases)
    return {
        "training_suite_sha256": digest(json_bytes(train.to_dict())),
        "development_suite_sha256": digest(json_bytes(probes.to_dict())),
        "split_manifest_sha256": digest(json_bytes(partition)),
        "serialization_sha256": digest(
            json_bytes(
                [
                    {
                        "case_id": row["case_id"],
                        "candidate_texts": row["candidate_texts"],
                        "target_index": row["target_index"],
                    }
                    for row in serialized
                ]
            )
        ),
        "candidate_sha256": digest(
            json_bytes(
                [
                    {
                        "case_id": row["case_id"],
                        "candidates": row["candidates"],
                        "target": row["target"],
                    }
                    for row in serialized
                ]
            )
        ),
    }


def _token_pieces(tokenizer: AsterTokenizer, text: str) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    reverse_special = {token_id: token for token, token_id in tokenizer.special_tokens.items()}
    pieces: list[dict[str, Any]] = []
    cursor = 0
    for token_id in ids:
        if token_id in reverse_special:
            pieces.append(
                {
                    "token_id": token_id,
                    "special": reverse_special[token_id],
                    "byte_start": None,
                    "byte_end": None,
                    "bytes_hex": None,
                    "display": reverse_special[token_id],
                }
            )
            continue
        token_bytes = tokenizer.model.vocab[token_id]
        start = cursor
        cursor += len(token_bytes)
        pieces.append(
            {
                "token_id": token_id,
                "special": None,
                "byte_start": start,
                "byte_end": cursor,
                "bytes_hex": token_bytes.hex(),
                "display": token_bytes.decode("utf-8", errors="replace"),
            }
        )
    raw = text.encode("utf-8")
    rebuilt = b"".join(
        tokenizer.model.vocab[token_id]
        for token_id in ids
        if token_id not in reverse_special
    )
    if rebuilt != raw or cursor != len(raw):
        raise ValueError("Tokenizer byte reconstruction mismatch")
    if tokenizer.decode(ids) != text:
        raise ValueError("Tokenizer round-trip mismatch")
    return {"token_ids": ids, "token_count": len(ids), "pieces": pieces}


def _audit_tokenizer(
    tokenizer: AsterTokenizer,
    cases: tuple[BenchmarkCase, ...],
    *,
    context_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    semantic_tokens = 0
    padded_tokens = 0
    maximum = 0
    for case in cases:
        candidate_rows = []
        lengths = []
        for candidate in case.example.candidates:
            text = serialize_decision_input(case.example.state, case.example.trajectory, candidate)
            tokenized = _token_pieces(tokenizer, text)
            length = int(tokenized["token_count"])
            if length > context_length:
                raise ValueError(
                    f"{case.split} case exceeds context length for "
                    f"{tokenizer.manifest.tokenizer_type}: {case.case_id}"
                )
            lengths.append(length)
            candidate_rows.append(
                {
                    "candidate": candidate.to_dict(),
                    "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "serialized_bytes": len(text.encode("utf-8")),
                    **tokenized,
                }
            )
        case_semantic = sum(lengths)
        case_padded = len(lengths) * max(lengths)
        semantic_tokens += case_semantic
        padded_tokens += case_padded
        maximum = max(maximum, max(lengths))
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "slice": case.slice_name,
                "target_index": case.example.target_index,
                "candidate_count": len(candidate_rows),
                "semantic_tokens": case_semantic,
                "right_padded_tokens": case_padded,
                "candidates": candidate_rows,
            }
        )
    return rows, {
        "cases": len(cases),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": tokenizer.manifest.tokenizer_type,
        "max_candidate_tokens": maximum,
        "semantic_tokens_per_pass": semantic_tokens,
        "right_padded_tokens_per_pass": padded_tokens,
    }


def _shared_initial_states(
    bpe_tokenizer: AsterTokenizer,
    byte_tokenizer: AsterTokenizer,
    config: DecisionFitStudyConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Share every compatible tensor and semantically matching byte/special rows."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        bpe_model = _new_model(bpe_tokenizer, config)
        torch.manual_seed(config.seed)
        byte_model = _new_model(byte_tokenizer, config)

    bpe_state = {name: tensor.detach().cpu().clone() for name, tensor in bpe_model.state_dict().items()}
    byte_state = {name: tensor.detach().cpu().clone() for name, tensor in byte_model.state_dict().items()}

    full_shared: list[str] = []
    for name, tensor in bpe_state.items():
        other = byte_state.get(name)
        if other is not None and tensor.shape == other.shape:
            byte_state[name] = tensor.clone()
            full_shared.append(name)

    row_names = (
        "backbone.token_embedding.weight",
        "backbone.lm_head.projection.weight",
    )
    semantic_rows: dict[str, dict[str, int]] = {}
    for name in row_names:
        bpe_tensor = bpe_state[name]
        byte_tensor = byte_state[name]
        byte_tensor[:256] = bpe_tensor[:256]
        byte_tensor[byte_tokenizer.bos_id] = bpe_tensor[bpe_tokenizer.bos_id]
        byte_tensor[byte_tokenizer.eos_id] = bpe_tensor[bpe_tokenizer.eos_id]
        semantic_rows[name] = {
            "raw_byte_rows": 256,
            "bos_bpe_id": bpe_tokenizer.bos_id,
            "bos_byte_id": byte_tokenizer.bos_id,
            "eos_bpe_id": bpe_tokenizer.eos_id,
            "eos_byte_id": byte_tokenizer.eos_id,
        }

    shared_full_hash = _state_digest({name: bpe_state[name] for name in full_shared})
    bpe_semantic: dict[str, torch.Tensor] = {}
    byte_semantic: dict[str, torch.Tensor] = {}
    for name in row_names:
        bpe_semantic[name + ":raw"] = bpe_state[name][:256]
        byte_semantic[name + ":raw"] = byte_state[name][:256]
        bpe_semantic[name + ":special"] = bpe_state[name][
            [bpe_tokenizer.bos_id, bpe_tokenizer.eos_id]
        ]
        byte_semantic[name + ":special"] = byte_state[name][
            [byte_tokenizer.bos_id, byte_tokenizer.eos_id]
        ]
    bpe_semantic_hash = _state_digest(bpe_semantic)
    byte_semantic_hash = _state_digest(byte_semantic)
    if bpe_semantic_hash != byte_semantic_hash:
        raise RuntimeError("Semantic shared initialization mismatch")

    return bpe_state, byte_state, {
        "full_shared_parameter_names": sorted(full_shared),
        "full_shared_state_sha256": shared_full_hash,
        "semantic_row_mapping": semantic_rows,
        "semantic_rows_sha256": bpe_semantic_hash,
        "full_initial_weights_identical": False,
        "reason": "vocabulary-dependent tensors have different shapes",
    }


def _parameter_counts(state: dict[str, torch.Tensor]) -> dict[str, int]:
    total = sum(tensor.numel() for tensor in state.values())
    lm_head = state["backbone.lm_head.projection.weight"].numel()
    token_embedding = state["backbone.token_embedding.weight"].numel()
    return {
        "total_parameters": total,
        "decision_path_parameters": total - lm_head,
        "token_embedding_parameters": token_embedding,
        "unused_lm_head_parameters": lm_head,
    }


def _episode_details(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
) -> tuple[list[dict[str, Any]], bool]:
    """Run model-only episodes and retain the first teacher disagreement plus later path."""
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.eval()
    episodes: list[dict[str, Any]] = []
    for operation, variant, task in development_tasks():
        context = RuntimeContext(task=task)
        executor = _executor()
        evaluator = TaskEvaluator()
        recorder = TrajectoryRecorder()
        teacher = RuleBasedPolicy()
        builder = CalculateAndStoreCandidates()
        available = executor.registry.names() + ("stop",)
        decisions: list[dict[str, Any]] = []
        first_error: dict[str, Any] | None = None
        coverage: list[bool] = []

        for _ in range(8):
            history = recorder.trajectory()
            step = len(history.transitions)
            state = context.snapshot(step)
            target = teacher.decide(state, history, available)
            candidates = builder.build(state, history, available)
            coverage.append(target in candidates)
            with torch.no_grad():
                scores = score_candidates(model, tokenizer, state, history, candidates)
            logits = [float(value) for value in scores.detach().cpu().tolist()]
            selected_index = int(scores.argmax().item())
            selected = candidates[selected_index]
            decision = {
                "step": step,
                "target": target.to_dict(),
                "correct_action_set": [target.to_dict()],
                "selected": selected.to_dict(),
                "selected_index": selected_index,
                "candidates": [candidate.to_dict() for candidate in candidates],
                "scores": logits,
                "correct": selected == target,
            }
            decisions.append(decision)
            if first_error is None and selected != target:
                candidate_inputs = []
                for candidate, score in zip(candidates, logits, strict=True):
                    text = serialize_decision_input(state, history, candidate)
                    candidate_inputs.append(
                        {
                            "candidate": candidate.to_dict(),
                            "score": score,
                            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            **_token_pieces(tokenizer, text),
                        }
                    )
                first_error = {**decision, "candidate_inputs": candidate_inputs}

            observation = executor.execute(selected, context)
            after = context.snapshot(step + 1)
            evaluation = evaluator.evaluate(task, after, selected, observation, history)
            recorder.record(
                Transition(
                    step=step,
                    state_before=state.to_dict(),
                    available_actions=available,
                    action=selected,
                    observation=observation,
                    state_after=after.to_dict(),
                    evaluation=evaluation,
                )
            )
            if evaluation.terminal:
                break

        trajectory = recorder.trajectory()
        last = trajectory.last
        success = bool(last and last.evaluation.terminal and last.evaluation.goal_satisfied)
        episodes.append(
            {
                "task_id": f"{operation}/{variant}",
                "variant": variant,
                "success": success,
                "steps": len(trajectory.transitions),
                "fallback": False,
                "teacher_candidate_coverage": (
                    sum(coverage) / len(coverage) if coverage else None
                ),
                "first_error": first_error,
                "decisions": decisions,
                "trajectory": [transition.to_dict() for transition in trajectory.transitions],
            }
        )

    unchanged = all(
        torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )
    return episodes, unchanged


def run_tokenizer_comparison(
    root: str | Path,
    bpe_tokenizer_dir: str | Path,
    *,
    config: DecisionFitStudyConfig | None = None,
    source_git_sha: str | None = None,
    expected_bpe_sha256: str | None = EXPECTED_BPE_SHA256,
) -> Path:
    """Train paired BPE/byte candidates without opening any reserved test."""
    config = config if config is not None else DecisionFitStudyConfig(
        epochs=50,
        learning_rate=1e-3,
        seed=42,
        target_vocab_size=512,
        min_pair_frequency=1,
        context_length=2048,
        width=16,
        heads=2,
        layers=1,
        stability_window=3,
    )
    train, case_ids = build_comparison_training_suite()
    probes = build_development_probes()
    sealed = build_sealed_prefix_suite()
    partition = validate_partition(train, probes, sealed)
    protocol_digests = _protocol_digests(train, probes, partition)

    source_bpe_path = Path(bpe_tokenizer_dir)
    source_bpe_sha = _directory_digest(source_bpe_path)
    if expected_bpe_sha256 is not None and source_bpe_sha != expected_bpe_sha256:
        raise ValueError(
            f"BPE tokenizer digest mismatch: {source_bpe_sha} != {expected_bpe_sha256}"
        )
    bpe_tokenizer = load_tokenizer(source_bpe_path)
    if bpe_tokenizer.manifest.tokenizer_type != "byte_bpe":
        raise ValueError("BPE comparison arm must use a byte_bpe artifact")
    byte_tokenizer = build_byte_decision_tokenizer()

    # Training and development are audited. Reserved test is intentionally not tokenized.
    bpe_train_audit = _audit_train_cases(train.cases, bpe_tokenizer, config.context_length)
    byte_train_audit = _audit_train_cases(train.cases, byte_tokenizer, config.context_length)
    bpe_dev_audit, bpe_dev_summary = _audit_tokenizer(
        bpe_tokenizer, probes.cases, context_length=config.context_length
    )
    byte_dev_audit, byte_dev_summary = _audit_tokenizer(
        byte_tokenizer, probes.cases, context_length=config.context_length
    )
    _, bpe_train_summary = _audit_tokenizer(
        bpe_tokenizer, train.cases, context_length=config.context_length
    )
    _, byte_train_summary = _audit_tokenizer(
        byte_tokenizer, train.cases, context_length=config.context_length
    )

    bpe_state, byte_state, shared_init = _shared_initial_states(
        bpe_tokenizer, byte_tokenizer, config
    )
    initial_states = {"bpe": bpe_state, "byte": byte_state}
    tokenizers = {"bpe": bpe_tokenizer, "byte": byte_tokenizer}
    initial_sha = {name: _state_digest(state) for name, state in initial_states.items()}

    run = RunLog(
        Path(root),
        "decision_tokenizer_comparison",
        {
            "config": asdict(config),
            "source_git_sha": source_git_sha,
            "protocol_status": _protocol_status(config),
            "protocol_digests": protocol_digests,
            "partition": partition,
            "bpe_source_sha256": source_bpe_sha,
            "test": "sealed/not_scored",
        },
        producer="trainer",
    )
    started = time.perf_counter()
    try:
        _write_json(run.path / "training-suite.json", train.to_dict())
        _write_json(run.path / "development-probes.json", probes.to_dict())
        _write_json(run.path / "sealed-test-suite.json", sealed.to_dict())
        _write_json(run.path / "split-manifest.json", partition)
        _write_json(run.path / "protocol-digests.json", protocol_digests)
        _write_json(run.path / "shared-initialization.json", shared_init)
        _write_jsonl(run.path / "bpe-train-input-audit.jsonl", bpe_train_audit)
        _write_jsonl(run.path / "byte-train-input-audit.jsonl", byte_train_audit)
        _write_jsonl(run.path / "bpe-dev-token-audit.jsonl", bpe_dev_audit)
        _write_jsonl(run.path / "byte-dev-token-audit.jsonl", byte_dev_audit)

        for name, tokenizer in tokenizers.items():
            save_tokenizer(tokenizer, run.path / "tokenizers" / name)
        if _directory_digest(run.path / "tokenizers" / "bpe") != source_bpe_sha:
            raise RuntimeError("Saved BPE tokenizer differs from supplied artifact")
        torch.save(bpe_state, run.path / "initial-bpe-model.pt")
        torch.save(byte_state, run.path / "initial-byte-model.pt")

        results: list[dict[str, Any]] = []
        orders: dict[str, list[list[str]]] = {}
        for name in ("bpe", "byte"):
            def deadline_expired(signum, frame):
                raise TimeoutError(f"{name} exceeded 1800 seconds for training/reload/dev")
            old_handler = signal.signal(signal.SIGALRM, deadline_expired)
            signal.setitimer(signal.ITIMER_REAL, 1800)
            tokenizer = tokenizers[name]
            state = initial_states[name]
            arm = DecisionFitArm(name, case_ids, "shuffle")
            summary = _run_arm(
                run.path,
                train,
                protocol_digests["training_suite_sha256"],
                tokenizer,
                state,
                initial_sha[name],
                train.cases,
                arm,
                config,
                source_git_sha=source_git_sha,
            )
            checkpoint = run.path / "arms" / name / "checkpoint"
            frozen_model, frozen_tokenizer, manifest = load_decision_fit_checkpoint(checkpoint)
            dev_report, predictions, episodes = measure_state_diagnostics(
                frozen_model,
                frozen_tokenizer,
                suite=probes,
                task_specs=development_tasks(),
                step_lookup=fit_step_lookup(anchor_cases()),
            )
            detailed_episodes, detailed_weights_unchanged = _episode_details(
                frozen_model, frozen_tokenizer
            )
            if not detailed_weights_unchanged:
                raise RuntimeError("Detailed episode audit changed model weights")
            dev_report.update(
                checkpoint_id=manifest["checkpoint_id"],
                partition=partition,
                detailed_episode_weights_unchanged=True,
            )
            evaluation = checkpoint.parent / "dev-evaluation"
            evaluation.mkdir()
            _write_json(evaluation / "diagnostics.json", dev_report)
            _write_jsonl(evaluation / "predictions.jsonl", predictions)
            _write_jsonl(evaluation / "episodes.jsonl", episodes)
            _write_jsonl(evaluation / "episode-details.jsonl", detailed_episodes)

            order_rows = [
                json.loads(line)
                for line in (checkpoint.parent / "epoch-order.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            orders[name] = [row["case_ids"] for row in order_rows]
            token_summary = bpe_train_summary if name == "bpe" else byte_train_summary
            dev_token_summary = bpe_dev_summary if name == "bpe" else byte_dev_summary
            token_summary = {
                **token_summary,
                "development": dev_token_summary,
                "training_semantic_tokens_total": (
                    token_summary["semantic_tokens_per_pass"] * config.epochs
                ),
                "training_right_padded_tokens_total": (
                    token_summary["right_padded_tokens_per_pass"] * config.epochs
                ),
            }
            results.append(
                {
                    "arm_id": name,
                    "tokenizer_sha256": _directory_digest(run.path / "tokenizers" / name),
                    "tokenizer_manifest": asdict(tokenizer.manifest),
                    "initial_model_sha256": initial_sha[name],
                    "parameter_counts": _parameter_counts(state),
                    "token_budget": token_summary,
                    "fit": summary,
                    "dev": dev_report,
                    "episode_details": "dev-evaluation/episode-details.jsonl",
                }
            )
            run.event(
                "arm_measured",
                {
                    "arm_id": name,
                    "fit_target_met": summary["fit_target_met"],
                    "test_scored_cases": 0,
                    "weights_unchanged": dev_report["weights_unchanged"],
                },
            )
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        if orders["bpe"] != orders["byte"]:
            raise RuntimeError("Tokenizer arms did not receive identical shuffle orders")

        study = {
            "schema_version": SCHEMA,
            "run_id": run.id,
            "source_git_sha": source_git_sha,
            "protocol_status": _protocol_status(config),
            "config": asdict(config),
            "protocol_digests": protocol_digests,
            "partition": partition,
            "shared_initialization": shared_init,
            "same_epoch_orders": True,
            "bpe_source_sha256": source_bpe_sha,
            "arms": results,
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "process_max_rss_kib": _max_rss_kib(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "device": "cpu",
                "rss_scope": "process maximum across both sequential arms",
            },
            "scope": (
                "same 32 decisions and candidate ranking task; tokenizer arms differ in "
                "vocabulary size, sequence length and therefore parameter/compute cost"
            ),
            "test": {
                "old": "sealed/not_evaluated",
                "reserved_prefix": "sealed/not_scored",
                "scored_cases": 0,
            },
        }
        _write_json(run.path / "study.json", study)
        run.finish("completed", study="study.json", test_status="sealed")
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if 'old_handler' in locals():
            signal.signal(signal.SIGALRM, old_handler)
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    return run.path
