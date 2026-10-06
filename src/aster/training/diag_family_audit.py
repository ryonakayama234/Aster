"""Read-only evidence reconstruction and feature extraction for DIAG v3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import cast

import torch

from aster.benchmark.suite import build_calculate_and_store_suite
from aster.inference.decide import score_candidates
from aster.model.decision_artifact import load_decision_artifact
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionExample, serialize_decision_input
from aster.records.runlog import RunLog
from aster.records.transition import Action
from aster.tokenizer.artifact import AsterTokenizer
from aster.training.diag_family_features import (
    build_frozen_feature_schema,
    feature_schema_sha256,
    levenshtein_distance,
    longest_common_prefix_length,
    longest_common_prefix_ratio,
    numeric_summary,
)
from aster.training.diag_interference import (
    _task_from_family,
    _teacher_examples,
    _validate_parent_suite_lineage,
)
from aster.training.diag_update_depth import (
    TRIAL_CYCLE_ID as DIAG_V2_CYCLE_ID,
    TRIAL_FAMILY_IDS as DIAG_V2_FAMILY_IDS,
)
from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
    PROTOCOL_ID,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
)
from aster.training.learn_confirmatory_analysis import aggregate_confirmatory_units
from aster.training.learn_confirmatory_run import discover_completed_confirmatory_units
from aster.training.learn_probe import resolve_act_parent_artifact


EXTRACTION_SCHEMA_VERSION = "aster-diag-family-feature-extraction-0"
EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA = "c2fa2d175b4a23bbd246ed61266d0cdda2e22846"
EXPECTED_DIAG_V2_SOURCE_GIT_SHA = "e7e8342e1b844320bb610fff10a9e8cce50660c8"
EXPECTED_DIAG_V2_RUN_ID = "8100ff7fb195408995513bc0c40da901"
EXPECTED_PRIMARY_SUMMARY = {
    "wins": 6,
    "losses": 19,
    "ties": 5,
    "median_family_accuracy_delta": -0.25,
    "sign_test_p_value": 0.01463329792022705,
}
LEARN_EVIDENCE_MODES = ("original", "reproduction")
_TOKENIZER_FILES = (
    "manifest.json",
    "vocab.json",
    "merges.json",
    "special_tokens.json",
)


def reconstruct_confirmatory_evidence(
    root: str | Path,
    manifest: dict[str, object],
) -> tuple[dict[str, object], str]:
    """Reconstruct all 30 family outcomes and match the source campaign result exactly."""
    root_path = Path(root).resolve()
    units = discover_completed_confirmatory_units(root_path, manifest)
    result = aggregate_confirmatory_units(
        manifest,
        [unit for _, unit in units.values()],
    )
    if result.get("status") != "complete" or len(units) != 90:
        completed = result.get("completed_units")
        expected = result.get("expected_units")
        missing_raw = result.get("missing_units")
        missing_preview: list[str] = []
        if isinstance(missing_raw, list):
            for item in missing_raw[:12]:
                if not isinstance(item, dict):
                    continue
                family_id = item.get("family_id")
                seed = item.get("seed")
                if isinstance(family_id, str) and type(seed) is int:
                    missing_preview.append(f"{family_id}/seed={seed}")
        detail = (
            f"discovered={len(units)}, completed={completed}, expected={expected}"
        )
        if missing_preview:
            detail += ", missing=" + ", ".join(missing_preview)
            if isinstance(missing_raw, list) and len(missing_raw) > len(missing_preview):
                detail += f", ... (+{len(missing_raw) - len(missing_preview)} more)"
        raise RuntimeError(
            "DIAG v3 Gate 0 requires all 90 frozen confirmatory units; " + detail
        )
    if result.get("measurement_git_sha") != EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA:
        raise RuntimeError("Confirmatory measurement Git SHA does not match the frozen evidence")

    primary = _require_dict(result, "primary")
    for key, expected in EXPECTED_PRIMARY_SUMMARY.items():
        if primary.get(key) != expected:
            raise RuntimeError(f"Confirmatory primary identity mismatch for {key}")
    verdict = _require_dict(result, "verdict")
    if verdict.get("label") != "Not supported":
        raise RuntimeError("Confirmatory verdict identity mismatch")

    canonical_run_id, canonical = _discover_canonical_confirmatory_result(
        root_path,
        manifest,
    )
    for key in (
        "schema_version",
        "protocol_id",
        "manifest_sha256",
        "measurement_git_sha",
        "status",
        "expected_units",
        "completed_units",
        "family_results",
        "primary",
        "verdict",
    ):
        if canonical.get(key) != result.get(key):
            raise RuntimeError(
                f"Reconstructed confirmatory result differs from source campaign: {key}"
            )
    return result, canonical_run_id


def verify_learn_evidence_source(
    root: str | Path,
    current_manifest: dict[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    """Validate whether Gate 0 reads original or explicitly reproduced LEARN evidence."""
    if mode not in LEARN_EVIDENCE_MODES:
        raise ValueError(f"Unknown LEARN evidence mode: {mode}")

    root_path = Path(root).resolve()
    evidence_manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    if evidence_manifest != current_manifest:
        raise RuntimeError("LEARN evidence root frozen manifest differs from DIAG v3 manifest")

    source = _git_identity(root_path)
    if mode == "reproduction":
        if source["git_sha"] != EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA:
            raise RuntimeError(
                "LEARN reproduction evidence root must be checked out at the original "
                "measurement Git SHA"
            )
        if source["dirty"] is not False:
            raise RuntimeError("LEARN reproduction evidence root must have a clean worktree")

    return {
        "mode": mode,
        "source_git_sha": source["git_sha"],
        "source_dirty": source["dirty"],
        "measurement_git_sha_required": EXPECTED_CONFIRMATORY_MEASUREMENT_GIT_SHA,
        "original_raw_artifact_used": mode == "original",
        "claim_boundary": (
            "original_local_run_evidence"
            if mode == "original"
            else "reproduction_evidence_not_original_raw_artifact"
        ),
    }


def verify_diag_v2_run_identity(root: str | Path) -> dict[str, object]:
    """Verify the exact completed DIAG v2 campaign selected for the later overlay."""
    root_path = Path(root).resolve()
    runs_root = (root_path / "runs").resolve()
    run_dir = root_path / "runs" / EXPECTED_DIAG_V2_RUN_ID
    run_file = run_dir / "run.json"
    result_file = run_dir / "diag-update-depth-results.json"

    if (
        run_dir.is_symlink()
        or not run_dir.is_dir()
        or not run_file.is_file()
        or run_file.is_symlink()
    ):
        raise RuntimeError("Expected DIAG v2 campaign Run is missing")
    resolved_run = run_dir.resolve()
    if not resolved_run.is_relative_to(runs_root):
        raise RuntimeError("DIAG v2 campaign Run resolves outside the local Run store")

    run = _read_json(run_file)
    if run.get("schema_version") != "aster-run-0":
        raise RuntimeError("DIAG v2 campaign Run schema mismatch")
    if run.get("run_id") != EXPECTED_DIAG_V2_RUN_ID:
        raise RuntimeError("DIAG v2 campaign Run ID mismatch")
    if run.get("kind") != "diag_update_depth_trial_campaign":
        raise RuntimeError("Expected Run is not the DIAG v2 update-depth campaign")
    if run.get("status") != "completed":
        raise RuntimeError("DIAG v2 campaign Run is not completed")

    inputs = _require_dict(run, "inputs")
    if inputs.get("cycle_id") != DIAG_V2_CYCLE_ID:
        raise RuntimeError("DIAG v2 campaign cycle identity mismatch")
    if inputs.get("source_git_sha") != EXPECTED_DIAG_V2_SOURCE_GIT_SHA:
        raise RuntimeError("DIAG v2 campaign source Git SHA mismatch")
    if inputs.get("families") != list(DIAG_V2_FAMILY_IDS):
        raise RuntimeError("DIAG v2 campaign family set/order mismatch")
    if inputs.get("planned_experiment_units") != 12:
        raise RuntimeError("DIAG v2 campaign planned-unit identity mismatch")

    if not result_file.is_file() or result_file.is_symlink():
        raise RuntimeError("DIAG v2 campaign result is missing")
    result = _read_json(result_file)
    if result.get("schema_version") != "aster-diag-update-depth-result-0":
        raise RuntimeError("DIAG v2 result schema mismatch")
    if result.get("cycle_id") != DIAG_V2_CYCLE_ID or result.get("status") != "complete":
        raise RuntimeError("DIAG v2 result cycle/status mismatch")
    if result.get("source_git_sha") != EXPECTED_DIAG_V2_SOURCE_GIT_SHA:
        raise RuntimeError("DIAG v2 result source Git SHA mismatch")
    if result.get("independent_family_blocks") != len(DIAG_V2_FAMILY_IDS):
        raise RuntimeError("DIAG v2 independent-family count mismatch")
    if result.get("experiment_units") != 12:
        raise RuntimeError("DIAG v2 experiment-unit count mismatch")

    raw_summaries = result.get("family_summaries")
    if not isinstance(raw_summaries, list):
        raise RuntimeError("DIAG v2 family summaries are missing")
    family_ids = [
        summary.get("family_id")
        for summary in raw_summaries
        if isinstance(summary, dict)
    ]
    if family_ids != list(DIAG_V2_FAMILY_IDS) or len(raw_summaries) != len(family_ids):
        raise RuntimeError("DIAG v2 result family summaries mismatch")

    return {
        "run_id": EXPECTED_DIAG_V2_RUN_ID,
        "source_git_sha": EXPECTED_DIAG_V2_SOURCE_GIT_SHA,
        "cycle_id": DIAG_V2_CYCLE_ID,
        "family_ids": list(DIAG_V2_FAMILY_IDS),
        "result_file": "diag-update-depth-results.json",
        "identity_verified": True,
    }


def extract_family_feature_rows(
    manifest: dict[str, object],
    model: DecisionModel,
    tokenizer: AsterTokenizer,
) -> list[dict[str, object]]:
    """Extract the frozen outcome-independent feature table for all 30 families."""
    design = _require_dict(manifest, "family_design")
    raw_families = design.get("families")
    if not isinstance(raw_families, list) or len(raw_families) != 30:
        raise ValueError("DIAG v3 requires exactly 30 frozen families")

    rows: list[dict[str, object]] = []
    stratum_counts = {"add": 0, "subtract": 0}
    for family_index, raw_family in enumerate(raw_families, 1):
        if not isinstance(raw_family, dict):
            raise ValueError("DIAG v3 family must be an object")
        family = cast(dict[str, object], raw_family)
        family_id = _require_str(family, "family_id")
        correction_task = _task_from_family(family, "correction_task")
        sibling_task = _task_from_family(family, "sibling_task")
        operation = _require_task_operation(correction_task)
        if operation != _require_task_operation(sibling_task):
            raise ValueError("Correction/sibling operation mismatch")

        stratum_counts[operation] += 1
        correction_examples = _teacher_examples(correction_task)
        sibling_examples = _teacher_examples(sibling_task)
        if not correction_examples or len(correction_examples) != len(sibling_examples):
            raise ValueError("Aligned correction/sibling teacher prefixes are required")

        correction_measurements = [
            _measure_decision(example, model, tokenizer) for example in correction_examples
        ]
        sibling_measurements = [
            _measure_decision(example, model, tokenizer) for example in sibling_examples
        ]
        pair_measurements = [
            _measure_pair(left, right)
            for left, right in zip(
                correction_measurements,
                sibling_measurements,
                strict=True,
            )
        ]

        left = _require_task_int(correction_task, "left")
        right = _require_task_int(correction_task, "right")
        result = left + right if operation == "add" else left - right
        row: dict[str, object] = {
            "family_id": family_id,
            "family_index": family_index,
            "stratum_index": stratum_counts[operation],
            "operation": operation,
            "left_operand": left,
            "right_operand": right,
            "result": result,
            "left_digits": _digit_count(left),
            "right_digits": _digit_count(right),
            "result_digits": _digit_count(result),
            "absolute_operand_gap": abs(left - right),
        }
        _add_member_numeric(
            row,
            correction_measurements,
            sibling_measurements,
            (
                "serialized_input_char_length",
                "encoded_input_token_length",
                "target_raw_score",
                "best_non_target_raw_score",
                "target_margin",
                "target_rank",
                "parent_correct",
                "raw_target_nll",
                "raw_score_spread",
                "raw_score_entropy",
                "candidate_set_size",
                "target_candidate_token_length",
                "top_competitor_token_length",
                "target_minus_competitor_token_length_difference",
            ),
        )
        _add_member_categorical(
            row,
            correction_measurements,
            sibling_measurements,
            ("target_action_type", "top_competitor_action_type"),
        )
        _add_pair_numeric(
            row,
            pair_measurements,
            (
                "correction_sibling_input_token_length_difference",
                "token_levenshtein_distance",
                "longest_common_token_prefix_length",
                "longest_common_token_prefix_ratio",
            ),
        )
        _add_pair_categorical(
            row,
            pair_measurements,
            ("top_competitor_type_matches_pair",),
        )
        rows.append(row)

    expected_outputs = set(cast(list[str], build_frozen_feature_schema()["output_names"]))
    for row in rows:
        actual = set(row) - {"family_id"}
        if actual != expected_outputs:
            missing = sorted(expected_outputs - actual)
            extra = sorted(actual - expected_outputs)
            raise RuntimeError(
                f"DIAG v3 feature row/schema mismatch; missing={missing}, extra={extra}"
            )
    return rows


def run_family_feature_extraction(
    root: str | Path,
    *,
    learn_evidence_root: str | Path | None = None,
    learn_evidence_mode: str = "original",
) -> Path:
    """Execute DIAG v3 Gate 0 + Gate 1 without joining outcomes to features."""
    root_path = Path(root).resolve()
    evidence_root_path = (
        root_path
        if learn_evidence_root is None
        else Path(learn_evidence_root).resolve()
    )
    started = time.perf_counter()
    source = _git_identity(root_path)
    manifest = load_confirmatory_manifest(root_path / DEFAULT_MANIFEST_PATH)
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("DIAG v3 manifest identity mismatch")

    learn_evidence_source = verify_learn_evidence_source(
        evidence_root_path,
        manifest,
        mode=learn_evidence_mode,
    )
    confirmatory_result, evidence_campaign_run_id = reconstruct_confirmatory_evidence(
        evidence_root_path,
        manifest,
    )
    diag_v2_identity = verify_diag_v2_run_identity(root_path)
    lineage = _require_dict(manifest, "lineage")
    act_run_id = _require_str(lineage, "parent_act_run_id")
    artifact_id, artifact_path = resolve_act_parent_artifact(root_path, act_run_id)
    model, tokenizer, parent_manifest = load_decision_artifact(artifact_path)
    if parent_manifest.get("artifact_id") != artifact_id:
        raise RuntimeError("Resolved DIAG v3 parent artifact identity mismatch")
    _validate_parent_suite_lineage(parent_manifest, build_calculate_and_store_suite())

    schema = build_frozen_feature_schema()
    schema_sha256 = feature_schema_sha256(schema)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    model_before = _model_state_digest(model)
    tokenizer_before = _tokenizer_artifact_digest(artifact_path)
    run = RunLog(
        root_path,
        "diag_family_feature_extraction",
        {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": manifest_sha256,
            "feature_schema_sha256": schema_sha256,
            "canonical_confirmatory_run_id": (
                evidence_campaign_run_id if learn_evidence_mode == "original" else None
            ),
            "learn_evidence_campaign_run_id": evidence_campaign_run_id,
            "learn_evidence_source": learn_evidence_source,
            "diag_v2_identity": diag_v2_identity,
            "act_run_id": act_run_id,
            "parent_artifact_id": artifact_id,
            "source_git_sha": source["git_sha"],
            "source_dirty": source["dirty"],
            "causal_claim": False,
            "confirmatory_verdict": None,
            "outcome_joined": False,
        },
        producer="diagnostic_audit",
    )

    try:
        _write_json(run.path / "diag-family-feature-schema.json", schema)
        with torch.inference_mode():
            rows = extract_family_feature_rows(manifest, model, tokenizer)

        model_after = _model_state_digest(model)
        tokenizer_after = _tokenizer_artifact_digest(artifact_path)
        invariant = {
            "optimizer_constructed": False,
            "model_training": model.training,
            "all_parameters_requires_grad_false": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "model_state_sha256_before": model_before,
            "model_state_sha256_after": model_after,
            "model_state_unchanged": model_before == model_after,
            "tokenizer_artifact_sha256_before": tokenizer_before,
            "tokenizer_artifact_sha256_after": tokenizer_after,
            "tokenizer_artifact_unchanged": tokenizer_before == tokenizer_after,
            "model_artifact_promoted": False,
        }
        if (
            invariant["model_training"] is not False
            or invariant["all_parameters_requires_grad_false"] is not True
            or invariant["model_state_unchanged"] is not True
            or invariant["tokenizer_artifact_unchanged"] is not True
        ):
            raise RuntimeError("DIAG v3 no-update invariant failed")

        feature_payload = {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "feature_schema_sha256": schema_sha256,
            "manifest_sha256": manifest_sha256,
            "parent_artifact_id": artifact_id,
            "families": rows,
            "outcome_columns_present": False,
        }
        gate_payload = {
            "schema_version": "aster-diag-family-evidence-gate-0",
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": manifest_sha256,
            "canonical_confirmatory_run_id": (
                evidence_campaign_run_id if learn_evidence_mode == "original" else None
            ),
            "learn_evidence_campaign_run_id": evidence_campaign_run_id,
            "learn_evidence_source": learn_evidence_source,
            "reconstructed_confirmatory_result": confirmatory_result,
            "source_campaign_result_exact_match": True,
            "original_family_results_exactly_verified": (
                learn_evidence_mode == "original"
            ),
            "diag_v2_identity": diag_v2_identity,
        }
        summary = {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "status": "feature_extraction_complete",
            "feature_schema_sha256": schema_sha256,
            "manifest_sha256": manifest_sha256,
            "canonical_confirmatory_run_id": (
                evidence_campaign_run_id if learn_evidence_mode == "original" else None
            ),
            "learn_evidence_campaign_run_id": evidence_campaign_run_id,
            "learn_evidence_source": learn_evidence_source,
            "diag_v2_identity": diag_v2_identity,
            "act_run_id": act_run_id,
            "parent_artifact_id": artifact_id,
            "source": source,
            "families": len(rows),
            "outcome_joined": False,
            "causal_claim": False,
            "confirmatory_verdict": None,
            "no_update_invariant": invariant,
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "device": str(next(model.parameters()).device),
            },
            "next_gate": "join frozen family outcomes only after these artifacts are persisted",
        }
        _write_json(run.path / "diag-family-features.json", feature_payload)
        _write_json(run.path / "diag-family-evidence-gate.json", gate_payload)
        _write_json(run.path / "diag-family-feature-extraction.json", summary)
        run.finish(
            "completed",
            result="diag-family-feature-extraction.json",
            features="diag-family-features.json",
            evidence_gate="diag-family-evidence-gate.json",
            feature_schema="diag-family-feature-schema.json",
            families=len(rows),
            outcome_joined=False,
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _measure_decision(
    example: DecisionExample,
    model: DecisionModel,
    tokenizer: AsterTokenizer,
) -> dict[str, object]:
    if len(example.candidates) < 2:
        raise ValueError("DIAG v3 parent geometry requires at least two candidates")
    scores_tensor = score_candidates(
        model,
        tokenizer,
        example.state,
        example.trajectory,
        example.candidates,
    )
    scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
    if len(scores) != len(example.candidates) or any(
        not math.isfinite(value) for value in scores
    ):
        raise ValueError("DIAG v3 parent scores must be finite and candidate-aligned")

    target_index = example.target_index
    target_score = scores[target_index]
    non_target_indices = [
        index for index in range(len(scores)) if index != target_index
    ]
    competitor_index = max(
        non_target_indices,
        key=lambda index: (scores[index], -index),
    )
    competitor_score = scores[competitor_index]
    probabilities = torch.softmax(scores_tensor, dim=0)
    log_probabilities = torch.log_softmax(scores_tensor, dim=0)
    entropy = float(
        (-(probabilities * log_probabilities).sum()).detach().cpu().item()
    )

    target_serialized = serialize_decision_input(
        example.state,
        example.trajectory,
        example.target,
    )
    target_input_tokens = tokenizer.encode(
        target_serialized,
        add_bos=True,
        add_eos=True,
    )
    target_action_tokens = _encode_action(tokenizer, example.target)
    competitor_action = example.candidates[competitor_index]
    competitor_action_tokens = _encode_action(tokenizer, competitor_action)

    return {
        "serialized_input_char_length": len(target_serialized),
        "encoded_input_token_length": len(target_input_tokens),
        "_input_tokens": target_input_tokens,
        "target_raw_score": target_score,
        "best_non_target_raw_score": competitor_score,
        "target_margin": target_score - competitor_score,
        "target_rank": 1 + sum(value > target_score for value in scores),
        "parent_correct": int(int(torch.argmax(scores_tensor).item()) == target_index),
        "raw_target_nll": float((-log_probabilities[target_index]).detach().cpu().item()),
        "raw_score_spread": max(scores) - min(scores),
        "raw_score_entropy": entropy,
        "target_action_type": _action_type(example.target),
        "top_competitor_action_type": _action_type(competitor_action),
        "candidate_set_size": len(example.candidates),
        "target_candidate_token_length": len(target_action_tokens),
        "top_competitor_token_length": len(competitor_action_tokens),
        "target_minus_competitor_token_length_difference": (
            len(target_action_tokens) - len(competitor_action_tokens)
        ),
    }


def _measure_pair(
    correction: dict[str, object],
    sibling: dict[str, object],
) -> dict[str, object]:
    correction_tokens = cast(list[int], correction["_input_tokens"])
    sibling_tokens = cast(list[int], sibling["_input_tokens"])
    return {
        "correction_sibling_input_token_length_difference": abs(
            len(correction_tokens) - len(sibling_tokens)
        ),
        "token_levenshtein_distance": levenshtein_distance(
            correction_tokens,
            sibling_tokens,
        ),
        "longest_common_token_prefix_length": longest_common_prefix_length(
            correction_tokens,
            sibling_tokens,
        ),
        "longest_common_token_prefix_ratio": longest_common_prefix_ratio(
            correction_tokens,
            sibling_tokens,
        ),
        "top_competitor_type_matches_pair": (
            correction["top_competitor_action_type"]
            == sibling["top_competitor_action_type"]
        ),
    }


def _add_member_numeric(
    row: dict[str, object],
    correction: Sequence[dict[str, object]],
    sibling: Sequence[dict[str, object]],
    names: Sequence[str],
) -> None:
    for member, measurements in (("correction", correction), ("sibling", sibling)):
        for name in names:
            summary = numeric_summary([_numeric(item, name) for item in measurements])
            for aggregation, value in summary.items():
                row[f"{member}_{name}_{aggregation}"] = value


def _add_member_categorical(
    row: dict[str, object],
    correction: Sequence[dict[str, object]],
    sibling: Sequence[dict[str, object]],
    names: Sequence[str],
) -> None:
    for member, measurements in (("correction", correction), ("sibling", sibling)):
        for name in names:
            values = [_categorical(item, name) for item in measurements]
            counts, mode = _categorical_summary(values)
            row[f"{member}_{name}_counts"] = counts
            row[f"{member}_{name}_mode"] = mode


def _add_pair_numeric(
    row: dict[str, object],
    pairs: Sequence[dict[str, object]],
    names: Sequence[str],
) -> None:
    for name in names:
        summary = numeric_summary([_numeric(item, name) for item in pairs])
        for aggregation, value in summary.items():
            row[f"pair_{name}_{aggregation}"] = value


def _add_pair_categorical(
    row: dict[str, object],
    pairs: Sequence[dict[str, object]],
    names: Sequence[str],
) -> None:
    for name in names:
        values = [_categorical(item, name) for item in pairs]
        counts, mode = _categorical_summary(values)
        row[f"pair_{name}_counts"] = counts
        row[f"pair_{name}_mode"] = mode


def _categorical_summary(
    values: Sequence[str | bool],
) -> tuple[dict[str, int], str | bool | list[str | bool]]:
    if not values:
        raise ValueError("At least one categorical feature value is required")
    counter = Counter(values)
    counts = {str(key): counter[key] for key in sorted(counter, key=str)}
    maximum = max(counter.values())
    modes = sorted(
        (value for value, count in counter.items() if count == maximum),
        key=str,
    )
    mode: str | bool | list[str | bool] = modes[0] if len(modes) == 1 else modes
    return counts, mode


def _discover_canonical_confirmatory_result(
    root: Path,
    manifest: dict[str, object],
) -> tuple[str, dict[str, object]]:
    runs_root = root / "runs"
    if not runs_root.exists():
        raise RuntimeError("Canonical confirmatory campaign Run store is missing")
    manifest_sha256 = confirmatory_manifest_sha256(manifest)
    matches: list[tuple[str, dict[str, object]]] = []
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        result_file = run_dir / "confirmatory-results.json"
        if not run_file.is_file() or run_file.is_symlink():
            continue
        run = _read_json(run_file)
        if run.get("kind") != "learn_confirmatory_campaign" or run.get("status") != "completed":
            continue
        inputs = run.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if inputs.get("protocol_id") != PROTOCOL_ID:
            continue
        if inputs.get("manifest_sha256") != manifest_sha256:
            continue
        if not result_file.is_file() or result_file.is_symlink():
            raise RuntimeError("Completed confirmatory campaign is missing its canonical result")
        matches.append((run_dir.name, _read_json(result_file)))
    if not matches:
        raise RuntimeError("No completed canonical confirmatory campaign result was found")
    reference = matches[0][1]
    if any(result != reference for _, result in matches[1:]):
        raise RuntimeError("Canonical confirmatory campaign results are ambiguous")
    return matches[0][0], reference


def _git_identity(root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("DIAG v3 requires a readable Git worktree") from error
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("DIAG v3 source Git SHA is invalid")
    dirty_entries = [line for line in status.splitlines() if line.strip()]
    return {
        "git_sha": revision,
        "dirty": bool(dirty_entries),
        "dirty_entry_count": len(dirty_entries),
    }


def _model_state_digest(model: DecisionModel) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        frozen = tensor.detach().cpu().contiguous()
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(str(frozen.dtype).encode("ascii"))
        value.update(b"\0")
        value.update(json.dumps(list(frozen.shape)).encode("ascii"))
        value.update(b"\0")
        value.update(bytes(frozen.view(torch.uint8).flatten().tolist()))
        value.update(b"\0")
    return value.hexdigest()


def _tokenizer_artifact_digest(artifact_path: Path) -> str:
    tokenizer_dir = artifact_path / "tokenizer"
    value = hashlib.sha256()
    for name in _TOKENIZER_FILES:
        item = tokenizer_dir / name
        if not item.is_file() or item.is_symlink():
            raise ValueError(f"Decision artifact tokenizer file is missing: {name}")
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def _encode_action(tokenizer: AsterTokenizer, action: Action) -> list[int]:
    serialized = json.dumps(
        action.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return tokenizer.encode(serialized, add_bos=True, add_eos=True)


def _action_type(action: Action) -> str:
    if action.kind == "stop":
        return "stop"
    return f"tool:{action.name}"


def _digit_count(value: int) -> int:
    return len(str(abs(value)))


def _numeric(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"DIAG v3 feature {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"DIAG v3 feature {key!r} must be finite")
    return number


def _categorical(data: dict[str, object], key: str) -> str | bool:
    value = data.get(key)
    if not isinstance(value, (str, bool)):
        raise ValueError(f"DIAG v3 feature {key!r} must be categorical")
    return value


def _require_task_operation(task: Mapping[str, object]) -> str:
    value = task.get("operation")
    if value not in {"add", "subtract"}:
        raise ValueError("DIAG v3 task operation must be add/subtract")
    return cast(str, value)


def _require_task_int(task: Mapping[str, object], key: str) -> int:
    value = task.get(key)
    if type(value) is not int:
        raise ValueError(f"DIAG v3 task field {key!r} must be an integer")
    return cast(int, value)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"DIAG v3 field {key!r} must be an object")
    return cast(dict[str, object], value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"DIAG v3 field {key!r} must be a non-empty string")
    return value
