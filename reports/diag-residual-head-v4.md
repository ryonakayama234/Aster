# DIAG v4 — Zero-init Residual Decision Head Trial closeout

## Diagnostic conclusion

- **Chosen branch:** `mixed_no_useful_effect` (protocol: **No useful effect / mixed**).
- **Observed phenotype:** Repair gains in both preselected loss exemplars, without a consistent sibling-stability gain; both preselected controls show more-negative stability contrasts.
- **Promotion:** none. Do not promote the residual head to `AsterDecision-v1`.
- `causal_claim=false`; `confirmatory_verdict=null`.
- No p-values, independent-family generalization claim, or new experiments in this closeout.

This is a post-hoc four-family diagnostic Trial; the family is the unit of interpretation, not a checkpoint, arm, or candidate. The treatment controls only the Decision Head architecture in the fixed runner.

## WSL execution and bookkeeping

- Reported campaign Run ID: `7fd70402448d463fac4472931c623b84`
- WSL Run path: `runs/7fd70402448d463fac4472931c623b84`
- Source Git SHA: `91ed7432caffc32c40bc0923f4c2c3b9ea595439` (post-fix PR #85)
- Campaign status: `complete`
- Stop rule: `stop_rule_satisfied=true`
- Four family blocks × two head architectures = **8 experiment units**.
- Matched Replay/Correction = **16 training trajectories**.
- **40 unit-checkpoint / 80 arm-checkpoint** records at steps 0, 10, 25, 50, 100.
- `architecture_conditions_are_independent_samples=false`
- `checkpoint_observations_are_independent=false`
- NumPy initialization warning was nonfatal; the experiment produced its completed aggregate.

## Four-family architecture contrasts

`D` = (Correction sibling-margin step100-step0) − (Replay sibling-margin step100-step0).  
`Repair_CR` = (Correction repair-accuracy step100-step0) − (Replay repair-accuracy step100-step0).

`stability_gain_residual=D_residual−D_linear`; more positive is better.  
`repair_gain_residual=Repair_CR_residual−Repair_CR_linear`; more positive is better.

| family | stratum | D linear | D residual | stability gain | Repair_CR linear | Repair_CR residual | repair gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 03 / add | loss | -0.002906844 | -0.001650020 | **+0.001256824** | 0.25 | 0.50 | **+0.25** |
| 15 / add | control | -0.006513774 | -0.010413304 | **-0.003899530** | 1.00 | 1.00 | 0 |
| 02 / subtract | loss | -0.010984674 | -0.012958005 | **-0.001973331** | 0.00 | 0.333333 | **+0.333333** |
| 12 / subtract | control | -0.009062365 | -0.013955817 | **-0.004893452** | 0.00 | 0.00 | 0 |

Descriptive medians, directly from the completed aggregate:

| stratum | median stability gain | median repair gain |
| --- | ---: | ---: |
| loss (03/02) | **-0.00035825371742248535** | **+0.29166666666666663** |
| control (15/12) | **-0.004396490752696991** | **0** |

Observations:
- Repair improves in **2/2** loss examples, but stability improves in only **1/2**.
- Both residual-head sibling `D` values in loss examples remain negative.
- Both control examples show a negative residual stability contrast; neither gains Repair.
- The numerical size of margin differences is small on the raw score scale. No preregistered materiality threshold or uncertainty interval was specified, so no claim of a statistically reliable degradation follows.

## Why `mixed_no_useful_effect`, rather than promoting capacity

The preregistered **Residual capacity supported** branch requires the two loss examples to improve sibling stability while preserving/improving Repair, without harmful control degradation. That conjunction is not observed. The **Repair-only** phenotype is visible, but even it is not a clean beneficial tradeoff across the four fixed families: sibling stability is mixed among the loss examples and worsens in both controls. The single closeout decision is therefore **No useful effect / mixed**.

This does not establish that residual capacity is useless in general. It rejects **promotion on the evidence of this fixed, post-hoc four-family Trial**.

## Exactly one next causal question

**Objective / regularization factor**:

> On the same frozen ACT backbone, same *linear* Decision Head, same tokenizer/serializer/candidate sets, same Replay/Correction examples, and same training budget, does adding one pre-registered parent-score-preservation KL term to the existing candidate cross-entropy objective improve Correction-vs-Replay sibling stability without sacrificing Correction Repair, compared with cross-entropy alone?

Only the training **objective** would differ. Use the same fixed training examples and candidate scores to form the added penalty; do not add a sibling training dataset, modify the candidate builder, or alter model architecture. Fix the regularization coefficient and evaluation rules *before* measuring outcomes. This is a proposed next question only, **not implemented or tested** in DIAG v4.

## Provenance and verification boundary

- Source: the user's full WSL terminal output of `diag-residual-head-results.json` for Run `7fd70402448d463fac4472931c623b84`.
- Companion `reports/diag-residual-head-v4.json` is a reserialization of that printed aggregate, with `diagnostic_classification=null` preserved exactly as the runner emitted it. The interpretive classification exists **only in this report and Issue #82**, not as a retroactive edit of measurement output.
- **Pending checksum gate:** compare `git hash-object runs/7fd70402448d463fac4472931c623b84/diag-residual-head-results.json` on the WSL machine with the GitHub blob SHA for `reports/diag-residual-head-v4.json`. Until the hashes match, the repo copy is **transcript-derived, not verified byte-identical to the source**.
- This report does not include or independently read the eight `diag-residual-head-unit.json` files. Their per-checkpoint frozen-backbone digests, step-0 score identity, tokenizer identity, and architecture parameter counts are asserted by the successful runner gates but not independently audited from the pasted aggregate.
- The full generated local Run tree is retained under ignored `runs/`; no model promotion, sealed-test access, or rerun was performed for this report.

## Research action

Close Issue #82 only after this report and the transcript-derived JSON are merged and the byte-identity check is documented. Start any objective Trial under a new, separately pre-registered issue rather than extending the eight-unit DIAG v4 budget.
