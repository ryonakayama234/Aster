# DIAG v2 — Last-Block Unfreeze Trial

Issue: #61

## Question

DIAG v1 found a strong backbone-mediated local-interference signal: freezing the shared
backbone nearly removed Correction-specific sibling degradation in the two loss exemplars,
but also reduced Repair. DIAG v2 asks the next single causal question:

> Can updating only the final Transformer block recover Repair while preserving most of
> the sibling protection seen under full backbone freeze?

This remains **Trial / diagnostic / post-hoc** evidence. It does not produce a
confirmatory capability verdict.

## Fixed family blocks

- `learn-confirm-keyshift-03` — add / loss exemplar
- `learn-confirm-keyshift-15` — add / non-loss control
- `learn-confirm-keyshift-02` — subtract / loss exemplar
- `learn-confirm-keyshift-12` — subtract / non-loss control

## Causal factor

Every family is rerun from one clean source Git SHA under three update-depth conditions.

### `head-only`

Train only `DecisionHead`.

### `last-block`

Train:

- Transformer block 1
- final LayerNorm
- DecisionHead

Freeze token/position embeddings and Transformer block 0.

### `full`

Train the complete Decision encoder path:

- token/position embeddings
- Transformer block 0
- Transformer block 1
- final LayerNorm
- DecisionHead

The TinyLM LM head is frozen in all three conditions because `DecisionModel.forward()`
uses `backbone.encode()` and does not call `backbone.lm_head`.

The exact trainable parameter names and their SHA-256 digest are stored in evidence.

## Fixed training conditions

The only manipulated factor is the explicit trainable-parameter mask.

Held fixed:

- parent ACT artifact lineage
- candidate set and Decision objective
- Replay/Correction construction
- seed: 42
- optimizer steps: 100
- learning rate: 0.003
- checkpoints: 0 / 10 / 25 / 50 / 100
- model-only rollout
- tokenizer / serializer
- model scale
- replay ratio

The historical `DecisionTrainConfig.train_backbone` API remains unchanged. DIAG v2
opts into an explicit parameter-name mask; legacy callers with no mask keep the old
boolean semantics.

## Bookkeeping

- independent family blocks: 4
- update-depth conditions per family: 3
- experiment units: 12
- matched arms per unit: Replay / Correction
- arm training trajectories: 24
- unit-checkpoint records: 60
- arm-checkpoint records: 120

Conditions, arms, and checkpoints are repeated measurements inside a family block and
are not independent samples.

## Primary quantities

For each family and update depth:

```
D =
  (Correction sibling margin step100 - step0)
  -
  (Replay sibling margin step100 - step0)
```

and

```
Repair_CR =
  (Correction repair accuracy step100 - step0)
  -
  (Replay repair accuracy step100 - step0)
```

More-negative `D` means worse Correction-specific sibling degradation.
Larger `Repair_CR` means more Correction-specific Repair.

## Paired contrasts

For each family:

```
stability_gain_last_vs_full = D_last_block - D_full
stability_cost_last_vs_head = D_last_block - D_head_only

repair_gain_last_vs_head = Repair_CR_last_block - Repair_CR_head_only
repair_gap_last_to_full = Repair_CR_last_block - Repair_CR_full
```

These contrasts are stored separately. DIAG v2 does not collapse stability and
plasticity into one scalar score and does not emit p-values.

## Parameter drift

Every checkpoint stores L2 drift for:

- embeddings
- block 0
- block 1
- final norm
- LM head
- DecisionHead

Legacy aggregate `head_l2` and `backbone_l2` fields are also retained so the existing
trajectory summarizer can be reused.

Every group declared frozen for a condition must have exact zero drift. A completed unit
is rejected otherwise.

## Training-data identity

Replay and Correction example digests are stored per unit. Within a family, all three
update-depth conditions must have identical Replay digests and identical Correction
digests. Aggregation fails if the causal factor accidentally changes training data.

## Resume / source revision

A campaign requires a clean worktree and records one source Git SHA. Completed units are
reused only when they belong to the same DIAG v2 cycle and exact source SHA. Mixing code
revisions is rejected.

## Stop rule

Stop after exactly:

```
4 families x 3 update-depth conditions = 12 experiment units
```

Do not add adapters, LoRA, learning-rate variants, replay-ratio variants, extra layers,
or families after inspecting partial results.

## Diagnostic branches

### Last-block sweet spot

`D_last_block` is materially safer than `D_full`, Repair recovers substantially above
`head-only`, and base retention does not show broad degradation.

Next: treat the shallow update policy as the first candidate for
**AsterDecision-v1 Stable Plasticity**.

### Last-block behaves like full

Repair returns but sibling interference largely returns too.

Next: test constrained added capacity such as an Adapter.

### Last-block behaves like head-only

Sibling protection remains but Repair does not return.

Next: test added trainable capacity or a separately staged deeper-unfreeze Trial.

### Mixed / family-dependent

Loss exemplars respond differently or controls show comparable effects.

Next: diagnose family/state features before selecting an update policy.
