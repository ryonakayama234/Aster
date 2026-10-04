# DIAG v0 — Correction Interference Curve Closeout

Date: 2026-10-04

Issue: #54

## Trial lineage

- active cycle: `diag-correction-interference-trial-1`
- supersedes: `diag-correction-interference-trial-0`
- source Git SHA: `20da70b859e02c16d2a017239c039b7f2a2e6133`
- local campaign Run ID: `0b55d23115f54fb797ca1898ca07751a`
- canonical local result: `runs/0b55d23115f54fb797ca1898ca07751a/diag-interference-results.json`
- status: complete
- independent training trajectories: 6
- repeated checkpoint observations: 30
- confirmatory verdict: none
- evidence scope: diagnostic / post-hoc selected

Trial 0 remains partial/superseded because singleton-candidate margin instrumentation changed after real WSL measurement began. Trial 0 units are preserved but are not mixed into Trial 1.

## Question

LEARN v0 showed strong corrected-state Repair but negative uncorrected-sibling transfer relative to matched Replay.

DIAG v0 asked whether the failure looked primarily like:

- A — local interference / correction overfit
- B — operation-specific interference
- C — broad forgetting

Trial 1 was diagnostic only. It was not designed to estimate population prevalence or produce a confirmatory p-value.

## Result summary

### A — local interference: **observed, heterogeneous**

Two post-hoc selected LEARN loss exemplars show the clearest signature.

| family | operation | Correction Repair Δ | Correction sibling accuracy Δ | Correction sibling margin Δ | Replay sibling accuracy Δ | Replay sibling margin Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 03 | add | +0.75 | -0.25 | -1.5885 | +0.25 | +0.0293 |
| 02 | subtract | +0.3333 | 0.00 | -0.7990 | +0.25 | +0.0060 |

Relative to Replay, the sibling-margin deltas are approximately:

- family 03: -1.6178
- family 02: -0.8051

In both cases, Correction learns the corrected state while moving the uncorrected sibling in a substantially worse direction than Replay.

The other selected win/tie exemplars do not show the same strong pattern. Therefore the mechanism is **not an inevitable consequence of Correction training**; it is family/state dependent in this panel.

### B — operation-specific interference: **not established**

Correction sibling-margin median delta:

- add: +0.0138
- subtract: -0.00433

Correction-minus-Replay sibling-margin median delta:

- add: +0.0689
- subtract: -0.0228

The severe interference exemplars occur once in add and once in subtract. With only three post-hoc selected families per operation, this Trial does not support treating subtraction as the primary mechanism.

A secondary parameter-drift pattern exists: Correction backbone drift exceeds Replay for all three selected add families and is below Replay for all three selected subtract families. This is descriptive and does not align cleanly with the loss exemplars, so it is not treated as the causal explanation.

### C — broad forgetting: **not observed**

Absolute Correction base retention from step 0 to step 100:

- base accuracy improves in 5/6 families
- base NLL improves in 5/6 families
- only family 15 shows both worse base accuracy and worse base NLL

So the observed sibling failures are not well described as global catastrophic forgetting.

However, relative to matched Replay, Correction's base-NLL improvement is weaker in **6/6 families**. The median

`(Correction base NLL Δ) - (Replay base NLL Δ)`

is approximately **+0.3144**.

This suggests a more local trade-off: targeted correction can buy Repair while giving up some of the broader optimization benefit obtained by Replay, without globally destroying the base behavior.

## Diagnostic decision

DIAG v0 result:

- **A: observed, heterogeneous**
- **B: not observed as a primary explanation**
- **C: not observed**
- D / optimizer instability: not evaluated because the current training path is deterministic

The most useful next causal question is therefore:

> Is the strong loss-exemplar interference caused by updating the shared backbone, or does it persist when only the Decision Head is trained?

## Measurement limitation

`first_negative_sibling_margin_step` is 0 for all six selected families because the raw sibling margin is already negative at the parent checkpoint.

Therefore this field does not localize the onset of degradation in this Trial.

A follow-up should instead record the first checkpoint where either:

- sibling margin change from step 0 becomes negative beyond a fixed threshold, or
- Correction-minus-Replay sibling-margin change becomes negative beyond a fixed threshold.

## Next decision

Do not escalate to a confirmatory Batch yet.

Run one more small causal Trial with a `train_backbone=false` control on the strongest loss exemplars plus matched non-loss controls.

If freezing the backbone removes the negative sibling movement while preserving Repair, shared-representation updates are implicated.

If interference persists with the backbone frozen, the next target should be the Decision Head / candidate geometry / objective rather than global representation forgetting.
