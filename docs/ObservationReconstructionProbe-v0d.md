# OBS v0d — Step-2 Observation Sensitivity (design review only)

**Status: PROPOSAL; no code implementation, score probing, or new learned-model execution is authorized by this file.** Separate research issue [#108](https://github.com/ryonakayama234/Aster/issues/108), parent [#106](https://github.com/ryonakayama234/Aster/issues/106). The source OBS v0c code was merged via PR #107 at commit `9414ea0906959d8e69c07e45bde3a5af0f715280`.

## Research question / competing explanations

A previous **single development Trial** terminated early after a real successful calculation. The user reported the original step=2 frozen DecisionTrace:

| Candidate index | Action | Score |
| --- | --- | ---: |
| 0 | calculator(23,19) | 3.2742202281951904 |
| 1 | memory.put(answer,42) | 3.2743303775787354 |
| 2 | memory.get(answer) | 3.2764711380004883 |
| 3 | stop(policy_stop) — selected | 3.27929425239563 |

Primary reference margin **stop minus put** = **+0.004963874816894531**, verified with Wolfram; the approximately **0.12418777 percentage-point** softmax gap is neither a success probability nor calibrated certainty.

Question: **Does masking the model-visible successful calculator output change its ranking of early stop relative to storing that observed value?**

- **H1 (input sensitivity):** score margin differs when calculator Observation output is masked.
- **H0/alternatives:** margin does not detectably differ, or observed difference stems from token count, token identity, formatting/length, or pre-existing candidate-specific priors instead of any semantic interpretation.
- **Selection bias:** this step was selected *after* seeing v0c failure. Always record `post_hoc_selected=true`. This is not an independent replication, model capability verdict, or a prospective statistical test.

## Exact source and input boundary

The **only source** is the existing original local WSL2 run `runs/c404f3a17eeb4e07a7e0edb29cc3376c/`, specifically `trajectory.jsonl`, `decision-traces.jsonl`, `obs-v0c-summary.json`, `run.json`, `evaluation.json`. Keep every original file immutable.

```text
Saved real trajectory:  step 0 scripted failed memory.get
                        step 1 frozen model calculator -> real output=42
                        step 2 frozen model stop (failure)
                               |
           reconstruct original, exact model-visible step-2 inputs
                               |
                [candidate #0, #1, #2, #3] SAME IN EVERY ARM
                               |
              input-only predeclared Observation modifications
                               |
            frozen DecisionModel score (NO action/ToolExecutor)
                               |
          candidate scalar scores / stop-minus-put margin
```

For every candidate, the input is produced using the unchanged `serialize_decision_input`, `RuntimeState` from **step=2 transition.state_before**, and only **steps 0 and 1** as trajectory history. Step=2 actual stop or EvaluationResult MUST NOT enter the input. Reconstructing from the recorded step-2 state prevents an oracle/reference success path from quietly replacing the real failed trajectory. The underlying task, memory state, step number, prior Action objects, four Candidate objects, candidate argument values/order, and unchanged serializer are invariant across all arms. **Never recompute or change the put candidate value** from an altered offline Observation.

## Frozen four-arm matrix

| Arm (order fixed) | Exact single-field input intervention | What it diagnoses |
| --- | --- | --- |
| `factual` | No changes | Exact original score/input reproduction |
| `calc_output_masked` | `history[1].observation.output = null` | Sensitivity to successful calculation output presence |
| `prefix_message_redacted` | `history[0].observation.error.message = "[REDACTED]"` | Placebo sensitivity to earlier missing-key error phrasing |
| `calc_value_43` | `history[1].observation.output = 43` instead of 42 | Same-character-length **numeric** perturbation probe; measure (do not assume) identical token lengths |

Four candidates per arm = **16 exact serialized inputs**, no extra arm, perturbation-value search, or score-adaptive retry. `calc_output_masked` and `calc_value_43` are **counterfactual offline strings**, intentionally inconsistent with the real observed tool output and the fixed `memory.put(answer,42)` candidate. They MUST NEVER become fake Transition/Observation records, executable tool histories, TaskEvaluator input, or evidence of successful recovery.

For `calc_value_43`, if any tokenizer length differs from the corresponding factual candidate, report `length_control_matched=false`; never automatically search for another numeric replacement. Even if the lengths match, the tokens may differ, so numerical sensitivity is not proof of semantic understanding.

## Falsifiable gates

**G0 provenance / immutable replay:** Require that:

- `run.json` identifies the original OBS v0c protocol, model artifact `decision_model:0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca`, and source Git SHA `9414ea0906959d8e69c07e45bde3a5af0f715280`. The local run ID must be exactly `c404f3a17eeb4e07a7e0edb29cc3376c`.
- `trajectory.jsonl` contains exactly three contiguous transitions, truthful scripted missing-key read at 0; selected calculator with successful actual output 42 at 1; stop selected at 2; memory empty at model step=2. No verified goal or terminal task success.
- `decision-traces.jsonl` has step 1 and step 2; step=2 trace contains **exactly** the four predeclared candidates and `selected_index=3`, with the four original scalar scores above. Trace.selected must equal transition.action.
- Reconstructed **four** factual candidate-input SHA-256 values exactly equal the previously written `obs-v0c-summary.json.decision_input_digests[1].candidate_input_sha256`. This comparison uses no new model call.
- Model/tokenizer come only from the same pinned artifact (verified by existing artifact loader). Do not infer missing historical source training provenance.

**G1 structural interventions (no torch necessary):** Use deep-copied serialized JSON documents, changing precisely the one allowed field per arm. Assert all remaining fields (including the whole candidate object) exactly equal factual. Serialize with the existing deterministic JSON contract (sorted keys, compact separators, UTF-8). Save the exact input string SHA-256, UTF-8 byte length, and character length for all 16 strings. **Token lengths require the pinned artifact tokenizer in G2**, not an optional dependency in the pure-structure path.

**G2 frozen offline score audit:** Before scoring, verify *every* tokenizer input length fits the actual model context. On overflow report `BLOCKED_CONTEXT` without truncation. **Score factual first** with the existing frozen DecisionModel scorer and require original raw scores for each candidate reproduced within a fixed absolute tolerance **1e-5**. Otherwise `BLOCKED_REPLAY_MISMATCH` and do not score altered arms. Also verify unchanged model `state_dict` after all scoring. Structural invalidity is `BLOCKED_SOURCE`. Missing file/artifact is `BLOCKED_INPUT`, not a behavioral failure.

No `ModelPolicy.decide`, argmax-based tool continuation, re-executed `ToolExecutor`, `TaskEvaluator` on altered data, SFT/RL/KLPO, sealed tests, alternate checkpoints, sampling, changes to weights/tokenizer/service, fallback, or threshold tuning.

## Endpoints and interpretation (all descriptive)

For each arm `c` and candidate `a` record scalar `s[c,a]`, relative softmax (un-calibrated), and rank/index.

Primary: `m(c) = s[c,stop] - s[c,put]`; `Δcalc = m(calc_output_masked) - m(factual)`. Secondary: `Δplacebo` for `prefix_message_redacted`, `Δ43` for `calc_value_43`, length equality flags, and all 16 hash/length entries. A negative `Δcalc` large enough to make `m(calc_output_masked) < 0` would reverse the stop-versus-put ranking **under the synthetic input only**. A small or zero `Δcalc` is a lack of detected sensitivity under this probe, **not** proof of no observation use. The placebo and numeric arms are alternative-mechanism probes, not independent samples. Do not compute an inferential p-value from a single post-hoc step.

Status in a completed score run is **`DIAGNOSTIC_ONLY`** regardless of any rank flips. Do **not** upgrade OBS v0c `NOT_SUPPORTED_ON_DEV_CASE` or claim a real successful episode.

## Evidence / implementation separation

**Stage 1 (this PR):** Review protocol only: fix its four arms, evidence boundary, replay tolerance, gates, and endpoints.

**Stage 2 (separate reviewed PR):** An isolated loader/probe, synthetic unit tests (field-only invariants, hash check, replay block, context block), and local WSL2 scoring CLI with two modes: source/structural-only without `torch`; optional *explicit* pinned model scoring. No new expensive CI or training. Only one fixed original run and 16 candidate strings. An audit writes a **new append-only Run** with source run ID, source trajectory and original trace hashes, the exact old source Git SHA, current analysis code SHA, artifact model/tokenizer hashes, all input hashes and lengths, baseline replay residuals, all raw scores, margin deltas, and `training_updates=0`. Do not modify or upload private source run/model artifact.

**Decision after scoring:** If output sensitivity appears, the next narrowly scoped trial tests whether that is attributable to observation meaning versus numeric/token representation. If insensitive, look at candidate-specific/stop training priors. Both are only diagnostic leads. Freeze a prospective different task/fault-family Batch before generalization claims. Follow `docs/ExperimentCycle-v0.md` Trial/Batch boundaries.
