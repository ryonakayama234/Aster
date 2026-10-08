# DIAG v5 — Frozen-parent KL preservation objective Trial closeout

Issue: #87. Implementation: merged PR #88 at `2227729b2c1589ebc2027b3b42d03993665eaa68`. Original Trial campaign: `runs/bdd77d5a2dfc455790d0cca0dba879c9`.

## Evidence status

**`user_attested_blob_match` — user-reported WSL Git blob SHA matches the GitHub JSON blob SHA byte-for-byte.**

This report transcribes and interprets the complete aggregate pasted from the user's WSL2 run on 2026-10-08. The accompanying `reports/diag-objective-kl-v5.json` was reconstructed from that user-visible aggregate, **not** fetched directly from WSL. The user subsequently supplied a terminal capture of `git hash-object runs/bdd77d5a2dfc455790d0cca0dba879c9/diag-objective-kl-results.json` returning **`08d72a79e0b30a1366b9e27e3c92969c9441d5fb`**. An audit detected a single transcription error in the reconstructed family 15 parent-Q correction digest (`...d487b2ab4b57...` instead of `...d487b2aeb4b57...`); after restoring the omitted `e`, the GitHub JSON blob SHA became **`08d72a79e0b30a1366b9e27e3c92969c9441d5fb`**, exactly matching the user-reported local SHA. Wolfram independently computed the same Git blob SHA from the corrected report bytes. The local file itself, eight per-unit artifacts and raw parent weights were not independently inspected through GitHub; the runner checks parent and tokenizer/artifact lineage locally.

User-reported completion: `status=complete`, `stop_rule_satisfied=true`, `source_git_sha=2227729b2c1589ebc2027b3b42d03993665eaa68`; all four fixed families × two objectives = **8 experiment units**, Replay and Correction = **16 training trajectories**, checkpoints = **40 unit / 80 arm observations**. The JSON's `diagnostic_classification` stays `null` by measurement contract; the interpretation is in this separate report. The NumPy warning did not prevent Trial completion.

## Pre-registered diagnostic quantities

`D = Δ(Correction sibling margin) - Δ(Replay sibling margin)`.

`Repair_CR = Δ(Correction repair accuracy) - Δ(Replay repair accuracy)`.

For each, `Δ` is step100 minus step0. Differences below compare `CE+KL` minus `CE`. A larger `D` is safer, but negative `D` still indicates residual Correction-specific loss of sibling margin.

| Family | Stratum | D CE | D CE+KL | stability gain KL | Repair_CR CE | Repair_CR CE+KL | Repair gain KL |
|---|---|---:|---:|---:|---:|---:|---:|
| learn-confirm-keyshift-03 (add) | loss | -0.0029068440198898315 | -0.0026132166385650635 | +0.00029362738132476807 | 0.25 | 0.375 | +0.125 |
| learn-confirm-keyshift-15 (add) | control | -0.006513774394989014 | -0.005339369177818298 | +0.0011744052171707153 | 1.0 | 1.0 | 0 |
| learn-confirm-keyshift-02 (subtract) | loss | -0.010984674096107483 | -0.009453758597373962 | +0.0015309154987335205 | 0 | 0 | 0 |
| learn-confirm-keyshift-12 (subtract) | control | -0.00906236469745636 | -0.008331239223480225 | +0.0007311254739761353 | 0 | 0 | 0 |

Descriptive median gains:

| Stratum | Stability gain KL | Repair gain KL |
|---|---:|---:|
| loss (03, 02) | +0.0009122714400291443 | +0.0625 |
| control (15, 12) | +0.0009527653455734253 | 0 |

Wolfram independently recalculated contrasts, medians, all signs, and the exact pre-registered decision predicate from the user-reported numbers.

## Classification and limitations

**Diagnostic classification following successful user-attested original-blob match: `preservation_supported_diagnostic`.**

This is the first pre-registered descriptive branch of `docs/DiagObjectiveKL-v5.md`:

1. Loss-family (03 and 02) stability gains strictly positive: **yes**.
2. Loss-family Repair gains nonnegative: **yes**.
3. Controls (15 and 12) have neither negative stability nor negative Repair gains: **yes**.

**Crucial negative evidence:** all four `D_(CE+KL)` remain **negative**. KL reduces the magnitude of the observed interference but does **not** eliminate it, and family 02 has `Repair_CR=0` under both objectives. Improvements are small in raw margin units; no minimum practically important effect or ability to generalize has been established. The four families are *post-hoc diagnostic selections* reused from prior investigations, not a new confirmatory sample. No p-values, seed/checkpoint pseudo-replication, model promotion, population generalization or confirmatory verdict. `causal_claim=false` and `confirmatory_verdict=null` remain unchanged.

The treatment preserved model architecture and switched only the training objective (CE vs CE+frozen-parent forward KL with `lambda=1.0`, temperature `1.0`). The completed run emitted parent-Q and matched training example SHA-256 fingerprints for Replay/Correction arms. Those match within each paired objective by the experiment's gate; independent audit of the eight local unit files was not performed through this report. A numerical model-only margin effect is not proof of Aster's end-to-end capability.

## Exactly one next causal question

> On a **prospectively frozen, genuinely disjoint family-level evaluation block**, does the same fixed CE+KL objective (lambda=1, temperature=1) deliver a positive Correction-specific *uncorrected sibling* effect relative to CE alone, **without worsening Correction Repair**, using identical parent/tokenizer/serializer/candidate builder and a sealed family-level primary endpoint?

This requires a separate research plan and new independent family manifest **frozen before any new endpoint measurement**. No family reuse from the diagnostic/previous confirmatory suites, hyperparameter tuning, seed cherry-picking or inspection of the project's sealed test. Do not start the next trial before the source evidence gate and new study pre-registration.

## Artifact identity and closing gate

1. **WSL user-terminal attestation:** `git hash-object runs/bdd77d5a2dfc455790d0cca0dba879c9/diag-objective-kl-results.json` returned `08d72a79e0b30a1366b9e27e3c92969c9441d5fb` on 2026-10-08.
2. **GitHub content API verification:** `reports/diag-objective-kl-v5.json` blob SHA, after correcting a single-character transcription omission in family 15's `parent_q_sha256.correction`, equals `08d72a79e0b30a1366b9e27e3c92969c9441d5fb`.
3. Wolfram's Git `blob <length>\\0<bytes>` SHA-1 calculation independently matched that exact SHA on corrected JSON bytes. This verifies content identity conditional on the user's genuine local command output, not independent access to their WSL filesystem.
4. Confirm only the two report files changed; CI and PR review before merging report-only PR #89. No independent Codex reviewer approval is claimed.
5. Close Issue #87 only after report merge; retain `causal_claim=false`, `confirmatory_verdict=null` and no AsterDecision-v1 promotion. Issue #90 is a **separate study design gate**, not automatically a confirmatory trial.
