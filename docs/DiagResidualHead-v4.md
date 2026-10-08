# DIAG v4 — Zero-init Residual Decision Head capacity Trial

Issue: #82

## Question

DIAG v3 closed with the descriptive classification `parent_geometry_linked`.
DIAG v4 therefore asks exactly one causal question:

> On the frozen ACT backbone, does adding a zero-init nonlinear residual scorer to the
> existing linear Decision Head reduce uncorrected sibling interference while preserving
> Correction Repair?

This is a **Trial / diagnostic / post-hoc** mechanism experiment. It is not an
AsterDecision-v1 promotion and does not produce a confirmatory capability verdict.

## Causal factor

Only the Decision Head architecture changes.

### `linear-head` control

```text
h ∈ R^16
score = linear(h)
```

The frozen ACT parent is copied exactly. Only the existing Decision Head is trainable.

### `residual-head` treatment

```text
h ∈ R^16

score =
  linear(h)
  +
  residual(h)

residual(h):
  Linear(16 -> 16)
  GELU
  Linear(16 -> 1)
```

The parent linear projection is copied exactly. Only the final residual
`Linear(16 -> 1)` weight and bias are initialized to zero. The first residual linear
layer keeps a normal nonzero initialization. Therefore the residual branch contributes
exactly zero at step 0 without making the whole branch permanently gradient-dead.

The residual branch adds 289 parameters:

```text
16*16 + 16 + 16*1 + 1 = 289
```

The frozen design records the absolute Decision-path count from the exact loaded ACT
parent artifact, then checks the architecture relation:

- linear control Decision path = loaded ACT parent Decision path;
- residual treatment Decision path = loaded ACT parent Decision path + 289 parameters;
- added fraction = `289 / loaded_parent_decision_path_parameter_count`.

The original preregistration arithmetic (44,289 -> 44,578, about 0.652532%) assumed
that the tokenizer's `target_vocab_size=512` was also the realized vocabulary size.
Aster BPE may stop before the target when no eligible pair remains, so the realized
vocabulary is an artifact property rather than a protocol constant. This bookkeeping
correction does not change the parent artifact, tokenizer, residual width, training
objective, family block, or any measured outcome.

The Decision path count excludes the unused TinyLM LM head because
`DecisionModel.forward()` calls `backbone.encode()` and the Decision Head. Machine-readable
evidence records both `backbone_vocab_size` and
`parent_decision_path_parameter_count`.

## Mandatory step-0 equivalence gate

Before any optimizer update, each unit constructs both architectures from the same
verified ACT parent and scores every candidate in the Replay training, Correction
training, and sibling diagnostic examples.

The unit fails before training unless:

- every corresponding candidate score is bitwise equal;
- every argmax Action is equal;
- maximum absolute score difference is exactly 0.

This makes architecture capacity, rather than a changed initial policy, the manipulated
factor.

## Fixed family block

The Trial reuses the DIAG v1/v2 post-hoc four-family block:

- `learn-confirm-keyshift-03` — add / loss exemplar
- `learn-confirm-keyshift-15` — add / control
- `learn-confirm-keyshift-02` — subtract / loss exemplar
- `learn-confirm-keyshift-12` — subtract / control

Each family is run under both architectures. Each family/architecture unit contains
matched Replay and Correction arms.

## Fixed training conditions

Held fixed across architecture conditions:

- exact ACT parent lineage
- tokenizer and verified tokenizer artifact digest
- serializer and candidate builder
- correction rollout construction
- Replay/Correction training-example digests
- seed: 42
- optimizer steps: 100
- learning rate: 0.003
- checkpoints: 0 / 10 / 25 / 50 / 100
- objective: existing candidate cross-entropy
- model-only rollout
- frozen backbone
- family selection
- evaluation metrics

The explicit trainable-parameter mask contains only `head.*` parameters.

## Frozen-state invariants

Every checkpoint records component L2 drift and the exact backbone state digest.

A unit is rejected if:

- backbone L2 drift is nonzero in either arm at any checkpoint;
- backbone digest differs from the architecture baseline;
- tokenizer artifact digest changes;
- the parent linear-head parameter count changes;
- the linear control Decision path differs from the exact loaded ACT parent;
- the residual branch is not exactly 289 parameters;
- the residual Decision path is not exactly loaded ACT parent + 289 parameters;
- the architecture changes the training-example digests.

## Primary quantities

For each family and architecture:

```text
D =
  (Correction sibling margin step100 - step0)
  -
  (Replay sibling margin step100 - step0)
```

```text
Repair_CR =
  (Correction repair accuracy step100 - step0)
  -
  (Replay repair accuracy step100 - step0)
```

Architecture contrasts:

```text
stability_gain_residual = D_residual - D_linear
repair_gain_residual    = Repair_CR_residual - Repair_CR_linear
```

More-positive `stability_gain_residual` means the residual architecture moved
Correction-specific sibling interference in the safer direction. Positive
`repair_gain_residual` means it also improved Correction-specific Repair relative to
the linear control.

No p-value is emitted.

## Bookkeeping and stop rule

```text
4 families x 2 architecture conditions = 8 experiment units
8 units x 2 matched arms = 16 training trajectories
8 units x 5 checkpoints = 40 unit-checkpoint records
16 arms x 5 checkpoints = 80 arm-checkpoint records
```

Stop after exactly eight experiment units. Do not add families, seeds, learning rates,
hidden widths, optimizer steps, adapters, or other architecture variants after inspecting
partial results.

## Machine-readable evidence

Each unit writes:

- `diag-residual-head-unit.json`
- `training-replay.json`
- `training-correction.json`

The campaign writes:

- `progress.json`
- `diag-residual-head-results.json`

The evidence includes architecture parameter counts, step-0 equivalence, parent/tokenizer
identity, training-example digests, checkpoint metrics, parameter drift, backbone state
digests, resource measurements, and paired family contrasts.

The campaign result intentionally leaves `diagnostic_classification=null`. The final
diagnostic branch is recorded only after the WSL CPU Trial is complete.

## Run after implementation merge

From the WSL repository root:

```bash
.venv/bin/python scripts/run_diag_residual_head.py --root . --threads 2
```

The implementation PR does not contain WSL endpoint results. Real-machine measurement and
the final report remain a later step on Issue #82.

## Non-goals

This Trial does not introduce:

- Adapter / LoRA
- backbone unfreeze
- new tokenizer or serializer
- candidate-builder change
- objective or regularization change
- RL / KLPO
- language TinyLM scaling
- sealed project test
- automatic model promotion
- independent unseen-family Batch
