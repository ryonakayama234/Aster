# OBS v0b — Offline Observation Ablation

2026-10-12 JST. Parent: OBS Issue #103. Prerequisite: OBS v0a PR #104, whose frozen-model continuation passed exactly one scripted-prefix development pair. This is an **offline input-score audit**, not an independent Agent capability test.

## Pinned model and experiment

Use only the already preselected artifact ID:
decision_model:0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca

The original candidate builder remains calculate-and-store-v0. The OBS experimental candidate set is fixed in both branches and always ordered [memory.get(key=answer), stop(reason=policy_stop)].

The task is add(23,19) and verify prepopulated memory answer=42. A first memory.get action is a **scripted ToolExecutor prefix**, never counted as a model choice. A truthful success or a deterministic nonmutating one-shot transient_read_error generates the branch-specific observation. At the first model-controlled decision, task, memory, step and available candidates are identical; only serialized observation differs.

## Frozen six-condition intervention matrix

Each condition is applied to both branches and both candidate Actions. Six conditions × two branches × two candidates = **24 exact serialized inputs**.

| Condition | Intervention on prior observation | Supported inference |
|---|---|---|
| factual | None; serialize the real ToolExecutor prefix | Recheck first choice; no fresh episode |
| masked | Replace complete observation with JSON null in both branches | Inputs for same candidate must be literally identical and score identically |
| status_only | Keep accepted/ok, null output/error | Structural status signal probe; invalid as an executable failed Observation |
| message_redacted | Failure arm replaces only error.message with [REDACTED], keeps code; success unchanged | Specific English error-text sensitivity |
| neutral_char_length | Neutral object with x-filled text to reproduce factual JSON character count | **Only a character-length proxy. Equal characters do not imply equal tokenizer length or equal token distribution** |
| swapped_offline | Swap only the observations in the serialized first-decision inputs | Duplicates opposite factual inputs/scores; not a new real trajectory |

These modified observations are *offline payloads only*. Never execute a Tool action based on them, record a synthetic Transition or use an Evaluator as if they were truthful environment history. Candidate order, candidate action arguments, task/memory/step, previous Action and all fields except the single Observation are invariant. Existing canonical sorted compact serializer is reused; never write a new production serializer.

## Gates and measurements

G0 structural: create the genuine two prefixes using the unchanged v0a executor; assert observed pair isolation, mask equality, swap equality and field-level isolation. No model is needed.

Preflight: SHA-256 of each **exact input string**; measure UTF-8 byte length, character length, and tokenizer length including BOS/EOS for all 24 inputs. Compare against actual frozen model context. If over context, output BLOCKED_CONTEXT, **without** inference or truncation.

Frozen G2 **offline score-only**: apply the ModelPolicy equivalent right-padded two-candidate batch scorer to all inputs with frozen weights; record per-condition and branch raw [retry, stop] scores, softmax probabilities, selected action/index and signed margin stop_score - retry_score. Assert identical masked inputs score identically, swapped inputs score exactly like opposite factual inputs, and weights are unchanged. No rule fallback, updates, calibration, held-out test or checkpoint search.

Diagnostic status must be DIAGNOSTIC_ONLY, or BLOCKED_CONTEXT. factual_first_decision=REPLICATED confirms only the first choices seen in OBS v0a, not any new sequential success. A different masked choice is *not* proof of semantics; flags, code, length, or token patterns are alternative explanations. Do not derive a p-value or generalization claim from this single already observed pair.

Record a new append-only Run containing obs-v0b-summary.json, frozen artifact/Tokenizer/model digests, source Git SHA, individual input hashes/lengths, raw scores and summary assertions. Do not modify v0a Runs, artifacts, Corpus, Tokenizer, sealed tests, ACT v0 implementation or model weights.

## WSL2

From a reviewed code checkout:

    cd ~/src/Aster
    .venv/bin/python scripts/run_obs_v0b.py --root .
    .venv/bin/python scripts/run_obs_v0b.py --root . --artifact artifacts/decision_models/0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca

The first command is structural-only; the second loads the exactly pinned model and writes a local score audit Run. GitHub CI tests use synthetic model inputs; **real frozen-model evidence requires the private local WSL2 artifact**. Independent fault-family generalization will be a separately preregistered OBS v0c with real tool and evaluator outcomes.
