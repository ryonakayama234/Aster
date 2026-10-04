# LEARN v0 — Correction Transfer Closeout

Date: 2026-10-04

Issue: #20

## Question

learner自身が訪れた失敗stateへのTeacher correctionは、同一parent・matched Replay updateより、未訂正sibling stateへ転移するか。

## Frozen protocol

- protocol: `learn-correction-transfer-confirmatory-v0`
- manifest SHA-256: `72fa77acf48403d5a45f927f1122d6fb5753b1118d9f25322e1d6700f2cb1563`
- measurement Git SHA: `c2fa2d175b4a23bbd246ed61266d0cdda2e22846`
- families: 30
- seeds: 42 / 43 / 44
- total units: 90
- primary: uncorrected sibling teacher-prefix accuracy
- unit of evidence: task/generator family
- test: exact two-sided sign test over non-tied family outcomes
- sealed project test: unopened

Canonical measured result remains in the local Run artifact produced by the WSL confirmatory campaign. This repository report records the result summary and lineage; it does not pretend the local Run artifact was committed to Git.

## Primary result

- wins: 6
- losses: 19
- ties: 5
- non-tied families: 25
- median family accuracy delta (Correction - Replay): -0.25
- exact sign-test p-value: 0.01463329792022705
- verdict: **Not supported**

Interpretation:

The frozen v0 hypothesis was not merely undetected. Within the frozen key-shift benchmark distribution, Correction was more often worse than the matched Replay control on the primary uncorrected-sibling endpoint.

This does **not** establish that correction learning is generally harmful, nor that all operations or task distributions behave identically.

## Repair / transfer boundary

Strong corrected-state Repair was frequently observed while Local Transfer was negative or tied.

Therefore:

- learning the corrected state,
- transferring to an uncorrected sibling state,
- succeeding in a model-only sequential episode

must remain separate evidence levels.

A high Repair score is not evidence of transfer.

## Post-hoc diagnostic lead

After the confirmatory verdict was fixed, operation was inspected as a diagnostic lead.

Among non-tied families:
- add: 5 wins / 7 losses
- subtract: 1 win / 12 losses

For subtract alone, a two-sided sign-test calculation gives p ≈ 0.00341797.
A two-sided Fisher-style exact comparison of add vs subtract win/loss allocation gives p ≈ 0.0730435.

These are **post-hoc diagnostics**, not new confirmatory findings. They motivate the next experiment but do not change the LEARN v0 verdict.

## Gate 5 decision

- LEARN v0 verdict: **Not supported**
- no automatic candidate promotion
- do not proceed directly to RL / KLPO
- next question: why can Correction improve Repair while degrading sibling transfer?
- next milestone: **DIAG v0 — Correction Interference Curve**

The next milestone should use Trial mode first and only escalate to Batch if a discriminating mechanism signature emerges.
