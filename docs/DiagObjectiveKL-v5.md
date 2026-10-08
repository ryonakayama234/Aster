# DIAG v5 — Frozen-parent KL preservation objective Trial

Issue: #87. Implementation base: `a9a979588bc501575de24be89d5423ebe6fe7f3d` (DIAG v4 closed in #86).

## Single diagnostic question

> On the same frozen ACT backbone and same linear Decision Head, does adding a fixed forward KL penalty against the parent candidate-action distribution improve Correction-specific uncorrected sibling stability without reducing Repair, compared with CE alone?

This is a **post-hoc four-family diagnostic Trial**, not confirmatory evidence or permission to promote AsterDecision-v1.

## One experimental factor

- **Control `ce`**: `L = CE(z_student, target)`.
- **Treatment `ce-kl`**: `L = CE(z_student, target) + 1.0 * KL(q_parent || p_student)`.

The fixed teacher distribution `q_parent` is `softmax(z_parent / 1)` on the **identical ordered candidate set for the same training example**. The student distribution is `softmax(z_student / 1)`. The teacher is the **unchanged ACT parent DecisionModel**, evaluated under `no_grad`, with all parameters frozen. `KL` uses PyTorch `kl_div(log_softmax(student_scores), softmax(parent_scores).detach(), reduction="sum")`; the only reduction is over the candidates of **one** example. Do not add sibling examples as regularization inputs. `ce` and `ce-kl` use the same parent-scoring operation for component logging; only the scalar loss changes.

`lambda=1.0` and `temperature=1.0` are fixed before any endpoint observation. No tuning or extra conditions after the first run.

## Same conditions and lineage

- Verified ACT parent: `resolve_act_parent_artifact(root, DEFAULT_ACT_RUN_ID)` with existing artifact ID/suite and tokenizer identity checks. Actual artifact bytes and parent-tokenizer digest must be confirmed on the WSL machine at runtime; do not invent a remote artifact digest.
- Both objectives: **identical linear head**, identical parent backbone and parameter count, frozen backbone, trainable `head.*` only, existing AdamW.
- Same tokenizer `aster-decision-input-0`, candidate builder `calculate-and-store-v0`, candidate order and Replay/Correction example digests.
- Exactly four post-hoc families: `03` add/loss, `15` add/control, `02` subtract/loss, `12` subtract/control, with complete IDs from Issue #87.
- `seed=42`, `learning_rate=0.003`, `steps=100`, checkpoints 0/10/25/50/100, CPU 2 threads, model-only decision rollout.
- Trial source Git SHA fixed on run start, checked on resumption; a new revision cannot reuse completed evidence for the same cycle. Existing `runs/` is not checked into Git.
- Train examples and parent Q probability sequences each get a digest per Replay/Correction arm; between the two objectives both must be byte-identical under project serialization.
- Test/sealed evaluation stays unopened; no new corpus, tokenizer, candidate builder, RL, scaling, adapter, architecture intervention, family or seed.

## Pre-update checks

1. Candidate scores at step0 are **exactly equal** for both objective copies across Replay, Correction and sibling diagnostic examples, without mutating the parent.
2. On an actual first Correction training example, KL is approximately zero and the head gradients of CE and CE+KL agree to `1e-6`; independent unit tests check the analytic gradient and teacher detachment.
3. `parent_q_sha256` is saved and paired across objectives, and the frozen backbone and tokenizer are checked unchanged at every requested checkpoint.
4. Training logs save stepwise `losses`, `ce_losses` and `kl_losses`, rather than silently treating total loss as CE.
5. If a gate fails, mark the unit failed and do not fill the missing slot with another family, revision or seed.

## Outcome quantities

For each family and objective, `D = Δ sibling_margin_Correction - Δ sibling_margin_Replay`. `Repair_CR = Δ repair_accuracy_Correction - Δ repair_accuracy_Replay`. `Δ` is step100 minus step0. Matched objective contrasts:

```
stability_gain_kl = D_(ce-kl) - D_ce
repair_gain_kl    = Repair_CR_(ce-kl) - Repair_CR_ce
```

Report individual outcomes for all four families and descriptive medians separately for loss and control strata. Interpret family as the unit of independence; replay arms, checkpoints and candidates are **not** independent tests.

**Precommitted descriptive decision tree**, applied after the entire fixed campaign:
- `preservation_supported_diagnostic`: stability gain strictly positive for both loss families, Repair gain nonnegative for both loss families, and neither stability nor Repair gain negative in either control family.
- `repair_only`: both loss families have Repair gain positive, but the conjunction above fails.
- `stability_only`: both loss families have stability gain positive but the full conjunction above fails, without both Repair gains positive.
- `mixed_or_no_useful_effect`: all remaining cases.

Zero is an exact neutral boundary for this **descriptive** decision tree, not a statistically established materiality threshold. Do not claim significance or a causal population effect from four post-hoc units. Record `causal_claim=false`, `confirmatory_verdict=null`, and do not auto-promote the model.

## Stop rule and outputs

```
4 families × 2 objectives = 8 experiment units
8 units × 2 matched arms = 16 optimizer trajectories
8 units × 5 checkpoints = 40 unit-checkpoint records
16 arms × 5 checkpoints = 80 arm-checkpoint observations
```

Stop exactly at eight completed units, without adding seeds, families or hyperparameters.

Run locally after implementation PR passes review and is merged:

```bash
.venv/bin/python scripts/run_diag_objective_kl.py --root . --threads 2
```

The CLI creates `runs/<campaign-id>/diag-objective-kl-results.json` with machine-readable outcomes and `runs/<unit-id>/diag-objective-kl-unit.json` and per-arm training traces. Rerunning resumes only same-SHA completed unit evidence. The aggregate leaves `diagnostic_classification=null` until a *separate report-only closeout*, at which point the above decision tree is applied once with exactly one next causal question.

**Execution boundary:** GitHub CI verifies code but does not have access to the user's local ACT parent artifact. Only WSL can perform the registered 8-unit measurement. No paid GitHub runners.
