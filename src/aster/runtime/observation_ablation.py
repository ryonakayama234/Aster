"""OBS v0b: offline intervention probes on frozen DecisionModel inputs.

Only factual prefixes come from ToolExecutor. Every modified observation is a
serialized-input probe: it is NEVER executed or fed to TaskEvaluator as a real
trajectory. v0a runtime, old builder, and weights are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from json import dumps, loads

import torch

from aster.records.decision import serialize_decision_input
from aster.runtime.observation_recovery import (
    BRANCHES, FIXED_CANDIDATES, RETRY, STOP,
    assert_observation_only_pair, create_episode,
)

PROTOCOL_ID = "obs-observation-ablation-v0b"
PINNED_ARTIFACT_ID = (
    "decision_model:0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca"
)
CONDITIONS = (
    "factual", "masked", "status_only", "message_redacted",
    "neutral_char_length", "swapped_offline",
)


def _encode(payload: dict) -> str:
    return dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_probes() -> dict[str, dict[str, tuple[str, str]]]:
    """Return exact two-candidate serialized inputs in [RETRY, STOP] order."""
    episodes = {branch: create_episode(branch) for branch in BRANCHES}
    assert_observation_only_pair(episodes["success"], episodes["transient_failure"])
    factual: dict[str, tuple[str, str]] = {}
    for branch, episode in episodes.items():
        state = episode.context.snapshot(1)
        pair = [
            serialize_decision_input(state, episode.trajectory, candidate)
            for candidate in FIXED_CANDIDATES
        ]
        factual[branch] = (pair[0], pair[1])

    probes: dict[str, dict[str, tuple[str, str]]] = {
        name: {} for name in CONDITIONS
    }
    for branch in BRANCHES:
        for condition in CONDITIONS:
            texts = []
            for index, original in enumerate(factual[branch]):
                payload = loads(original)
                observation = payload["history"][0]["observation"]
                if condition == "factual":
                    pass
                elif condition == "masked":
                    # Uniform null mask; no synthetic executable Observation.
                    payload["history"][0]["observation"] = None
                elif condition == "status_only":
                    payload["history"][0]["observation"] = {
                        "accepted": observation["accepted"],
                        "ok": observation["ok"], "output": None, "error": None,
                    }
                elif condition == "message_redacted":
                    if observation["error"] is not None:
                        observation["error"]["message"] = "[REDACTED]"
                elif condition == "neutral_char_length":
                    # A character-length *proxy*, not a tokenizer-length match.
                    payload["history"][0]["observation"] = {"neutral": ""}
                    base = _encode(payload)
                    n = len(original) - len(base)
                    if n < 0:
                        raise AssertionError("Neutral payload exceeds factual char length")
                    payload["history"][0]["observation"]["neutral"] = "x" * n
                elif condition == "swapped_offline":
                    other = "transient_failure" if branch == "success" else "success"
                    payload["history"][0]["observation"] = deepcopy(
                        loads(factual[other][index])["history"][0]["observation"]
                    )
                else:
                    raise AssertionError("Unknown intervention")
                rendered = _encode(payload)
                if condition == "neutral_char_length" and len(rendered) != len(original):
                    raise AssertionError("Char length control failed")
                if condition != "factual":
                    original_payload = loads(original)
                    altered_payload = loads(rendered)
                    original_payload["history"][0]["observation"] = None
                    altered_payload["history"][0]["observation"] = None
                    if original_payload != altered_payload:
                        raise AssertionError("Intervention changed a non-observation field")
                texts.append(rendered)
            if len(texts) != 2:
                raise AssertionError("Probe must contain exactly two candidates")
            probes[condition][branch] = (texts[0], texts[1])

    for index in range(2):
        if probes["masked"]["success"][index] != probes["masked"]["transient_failure"][index]:
            raise AssertionError("Mask must erase all branch information")
        if probes["swapped_offline"]["success"][index] != factual["transient_failure"][index]:
            raise AssertionError("Swap does not match the other factual input")
        if probes["swapped_offline"]["transient_failure"][index] != factual["success"][index]:
            raise AssertionError("Swap does not match the other factual input")
    return probes


def preflight(model, tokenizer, probes=None) -> dict:
    """Measure every exact modified input, with no model call or truncation."""
    if probes is None:
        probes = build_probes()
    metrics = {}
    max_length = 0
    for condition in CONDITIONS:
        metrics[condition] = {}
        for branch in BRANCHES:
            pair = probes[condition][branch]
            metrics[condition][branch] = []
            for value in pair:
                length = len(tokenizer.encode(value, add_bos=True, add_eos=True))
                metrics[condition][branch].append({
                    "sha256": sha256(value.encode("utf-8")).hexdigest(),
                    "token_length": length,
                    "char_length": len(value),
                    "utf8_bytes": len(value.encode("utf-8")),
                })
                max_length = max(max_length, length)
    limit = model.backbone.config.context_length
    return {
        "status": "PASS" if max_length <= limit else "BLOCKED_CONTEXT",
        "context_length": limit, "max_input_length": max_length,
        "inputs": metrics,
    }


def _score_pair(model, tokenizer, pair: tuple[str, str]) -> dict:
    """Faithful batched Aster DecisionModel scoring, not a model policy episode."""
    encoded = [tokenizer.encode(s, add_bos=True, add_eos=True) for s in pair]
    sizes = [len(ids) for ids in encoded]
    device = next(model.parameters()).device
    token_ids = torch.full(
        (2, max(sizes)), tokenizer.eos_id, dtype=torch.long, device=device,
    )
    for row, ids in enumerate(encoded):
        token_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(token_ids, torch.tensor(sizes, dtype=torch.long, device=device))
        scores = scores.detach().cpu()
        if scores.shape != (2,) or not torch.isfinite(scores).all():
            raise ValueError("Invalid candidate scores")
        probabilities = torch.softmax(scores, dim=0)
        selected = int(torch.argmax(probabilities).item())
    return {
        "scores": [float(x) for x in scores.tolist()],
        "probabilities": [float(x) for x in probabilities.tolist()],
        "selected_index": selected,
        "selected_action": FIXED_CANDIDATES[selected].to_dict(),
        "stop_minus_retry_score": float(scores[1] - scores[0]),
    }


def run_probe(model, tokenizer) -> dict:
    """Offline diagnostics only. No tool continuation, retries or task success claims."""
    probes = build_probes()
    audit = preflight(model, tokenizer, probes)
    if audit["status"] != "PASS":
        return {"protocol_id": PROTOCOL_ID, "status": "BLOCKED_CONTEXT", "preflight": audit}

    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    was_training = model.training
    model.eval()
    try:
        choices = {
            condition: {
                branch: _score_pair(model, tokenizer, probes[condition][branch])
                for branch in BRANCHES
            }
            for condition in CONDITIONS
        }
    finally:
        model.train(was_training)
    unchanged = all(
        torch.equal(before[name], current.detach().cpu())
        for name, current in model.state_dict().items()
    )
    if not unchanged:
        raise AssertionError("Frozen model weights changed")

    # Controls that must hold mathematically, independent of learned weights.
    mask_a = choices["masked"]["success"]
    mask_b = choices["masked"]["transient_failure"]
    if mask_a["scores"] != mask_b["scores"] or mask_a["selected_index"] != mask_b["selected_index"]:
        raise AssertionError("Identical masked inputs scored differently")
    for branch, opposite in (("success", "transient_failure"), ("transient_failure", "success")):
        if choices["swapped_offline"][branch]["scores"] != choices["factual"][opposite]["scores"]:
            raise AssertionError("Identical swapped input scored differently")

    factual_correct = (
        choices["factual"]["success"]["selected_index"] == FIXED_CANDIDATES.index(STOP)
        and choices["factual"]["transient_failure"]["selected_index"] == FIXED_CANDIDATES.index(RETRY)
    )
    return {
        "protocol_id": PROTOCOL_ID, "status": "DIAGNOSTIC_ONLY",
        "factual_first_decision": "REPLICATED" if factual_correct else "NOT_REPLICATED",
        "preflight": audit, "choices": choices,
        "control_assertions": {
            "same_masked_inputs_same_scores": True,
            "swapped_offline_equals_opposite_factual_scores": True,
            "neutral_matches_factual_character_length": True,
            "neutral_matches_token_length": all(
                audit["inputs"]["neutral_char_length"][b][i]["token_length"]
                == audit["inputs"]["factual"][b][i]["token_length"]
                for b in BRANCHES for i in range(2)
            ),
        },
        "model_weights_unchanged": unchanged, "training_updates": 0,
        "scope": "offline input interventions on one observed development pair; "
                 "no counterfactual environment executions, no generalization inference",
    }
