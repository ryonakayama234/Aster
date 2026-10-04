# DIAG v1 — Backbone Freeze Causal Trial

Issue: #59

## Question

DIAG v0 observed heterogeneous local interference: some Correction updates repaired the visited state while moving an uncorrected sibling substantially worse than matched Replay.

DIAG v1 asks one causal question:

> Does that interference depend on updating the shared backbone?

This remains **Trial / diagnostic / post-hoc** evidence. It does not produce a confirmatory capability verdict.

## Fixed family blocks

- `learn-confirm-keyshift-03` — add / loss exemplar
- `learn-confirm-keyshift-15` — add / non-loss control
- `learn-confirm-keyshift-02` — subtract / loss exemplar
- `learn-confirm-keyshift-12` — subtract / non-loss control

The selection uses DIAG v0 / LEARN results and is therefore `post_hoc_selected=true`.

## Causal factor

Each family is run under both:

- `train_backbone=true` (`full`)
- `train_backbone=false` (`freeze`)

Each family-condition unit contains two matched training arms:

- Replay
- Correction

All other training settings remain fixed:

- seed field: 42
- optimizer steps: 100
- learning rate: 0.003
- checkpoints: 0 / 10 / 25 / 50 / 100
- model-only rollout
- same ACT parent artifact lineage
- same baseline supervision construction
- same correction/replay example construction

## Bookkeeping

- independent family blocks: 4
- paired backbone conditions per family: 2
- experiment units: 8
- arms per unit: 2
- arm training trajectories: 16
- unit-checkpoint records: 40
- arm-checkpoint records: 80

Backbone conditions, arms, and checkpoints are repeated measurements inside a family block. They are not independent samples.

## Training-data identity

The Correction and Replay training-example digests are stored per unit.

For the same family, the `full` and `freeze` units must have identical training-example digests. Aggregation fails if the causal factor accidentally changes training data.

## Primary diagnostic estimand

For family (f) and backbone condition (b):

```
D[f,b] =
  (Correction sibling margin step100 - step0)
  -
  (Replay sibling margin step100 - step0)
```

Family-level freeze effect:

```
freeze_effect[f] = D[f,freeze] - D[f,full]
```

Positive freeze effect means freezing the backbone moved Correction-specific sibling interference in the safer direction.

No p-value is produced.

## Repair safeguard

For each family-condition:

```
Repair_CR =
  (Correction repair accuracy step100 - step0)
  -
  (Replay repair accuracy step100 - step0)
```

A safer sibling result is not treated as a useful mechanism fix if Repair collapses at the same time.

## Baseline-relative onset metrics

The raw sign of sibling margin is not an onset metric because DIAG v0 found negative parent margins at step 0.

Use epsilon (10^{-6}) and record:

- first checkpoint where Correction sibling-margin change from step 0 is < -epsilon
- first checkpoint where Correction-minus-Replay sibling-margin divergence from step 0 is < -epsilon

The repeated checkpoints remain descriptive observations from one trajectory.

## Other metrics

At every checkpoint, retain DIAG v0 metrics:

- Repair accuracy / NLL / defined margin
- sibling accuracy / NLL / defined margin
- base-retention accuracy / NLL / defined margin
- Decision Head L2 drift
- backbone L2 drift
- cumulative optimizer steps / token presentation / wall time

For `train_backbone=false`, backbone L2 drift must remain exactly 0 within numerical storage semantics.

## Diagnostic classification

### Backbone-mediated

The two loss exemplars (03 and 02) show materially positive freeze effects while Repair_CR remains substantially present.

### Head/objective-mediated

Loss exemplars remain strongly negative in D after freezing the backbone.

### Mixed / family-dependent

The two loss exemplars respond differently, or comparable freeze effects appear in non-loss controls.

No fixed statistical significance threshold is defined for this Trial. Classification is a mechanism diagnosis, not a population claim.

## Stop rule

Stop after exactly 8 experiment units.

Do not add families, seeds, learning rates, replay ratios, or other conditions to this cycle after inspecting results.

## Non-goals

- confirmatory claim
- sealed test
- RL / KLPO
- optimizer stochasticity study
- tokenizer / serializer change
- model scaling
- LR tuning
- replay-ratio tuning

## Next branch

- backbone-mediated → layer-selective/backbone-update control Trial
- head/objective-mediated → Decision Head / candidate geometry / objective Trial
- mixed → family/state feature diagnosis
- no useful signal → instrumentation redesign

Batch promotion remains deferred until a mechanism produces a preregisterable endpoint on independent unseen families.
