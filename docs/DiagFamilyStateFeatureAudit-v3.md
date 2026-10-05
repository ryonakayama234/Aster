# DIAG v3 — Family/State Feature Audit

Issue: #64

## Question

DIAG v2 ended **Mixed / family-dependent**. The next question is not which new training
intervention performs best. It is:

> Which already-observable family/state properties track the heterogeneity in
> Correction transfer/interference strongly enough to justify one next causal Trial?

DIAG v3 is therefore a **read-only / post-hoc diagnostic audit**. It does not update
model weights, fit a predictor, open the sealed project test, or emit a confirmatory
capability verdict.

## Parent evidence

Use only already-fixed evidence:

- LEARN v0 confirmatory protocol:
  `learn-correction-transfer-confirmatory-v0`
- frozen manifest SHA-256:
  `72fa77acf48403d5a45f927f1122d6fb5753b1118d9f25322e1d6700f2cb1563`
- canonical LEARN evidence:
  30 independent families x 3 fixed seeds = 90 completed units
- ACT parent Run:
  `5a8076571dca446fa07190cf4fc62509`
- parent suite:
  `calculate-and-store-v0`
- DIAG v2 source SHA:
  `e7e8342e1b844320bb610fff10a9e8cce50660c8`
- DIAG v2 local Run:
  `8100ff7fb195408995513bc0c40da901`

The four DIAG v2 selected families remain post-hoc overlays:

- 03 add / loss
- 15 add / control
- 02 subtract / loss
- 12 subtract / control

No LEARN or DIAG v2 campaign is rerun or modified.

## Evidence unit

The independent evidence unit is the **family**.

Seed, prefix, candidate, checkpoint, correction member, and sibling member are repeated
measurements or nested observations. They are never counted as independent samples.

## Gate 0 — Evidence identity before feature work

Before extracting new features:

1. discover the canonical 90 LEARN units read-only;
2. reproduce the canonical 30 family-level outcomes exactly;
3. verify the frozen manifest digest;
4. resolve the exact ACT parent artifact through the recorded Run lineage;
5. verify the DIAG v2 Run identity for the four-family overlay.

If the reconstructed 30-family outcomes differ from the frozen LEARN result, stop. Do
not continue with a partly reconstructed table.

Expected frozen LEARN primary summary:

- wins / losses / ties = 6 / 19 / 5
- median family accuracy delta = -0.25
- exact sign-test p = 0.01463329792022705
- confirmatory verdict = Not supported

These values are identity checks only. DIAG v3 does not retest that hypothesis.

## Gate 1 — Freeze features before outcome join

The feature extractor must be able to run **without family outcome columns**.

Persist a feature-schema object containing:

- schema version;
- exact feature names;
- definitions and units;
- aggregation rule;
- missing/degenerate handling;
- feature group;
- whether a feature is eligible for outcome screening.

Hash the canonical serialized schema with SHA-256 before joining outcomes.

Changing the feature schema after outcome association has been inspected ends the
current DIAG v3 cycle. A changed schema requires a new cycle/version.

## Fixed feature groups

### A. Design/task features

Per family:

- `family_index`: 1..30 from the frozen manifest
- `stratum_index`: 1..15 separately within add/subtract
- `operation`: add / subtract
- `left_operand`
- `right_operand`
- `result`
- `left_digits`
- `right_digits`
- `result_digits`
- `absolute_operand_gap`

`family_index` and `stratum_index` are **design controls**, not candidate causal
mechanisms.

### B. Serializer/tokenizer geometry

For correction and sibling teacher-prefix decisions, using the unchanged serializer and
Tokenizer:

- serialized input character length;
- encoded input token length;
- candidate count;
- target-candidate encoded token length;
- absolute correction-vs-sibling input token-length difference;
- token-level Levenshtein distance between aligned correction/sibling serialized inputs;
- longest common token-prefix length;
- longest-common-prefix ratio:
  `lcp / max(correction_tokens, sibling_tokens)`.

Do not retrain the Tokenizer and do not change serialization.

### C. Parent Decision geometry

Use the unchanged saved ACT parent, under `torch.inference_mode()`, with no optimizer.

Per decision:

- target raw score;
- best non-target raw score;
- target margin = target score - best non-target score;
- target rank = `1 + count(score > target_score)`;
- parent selected-action correctness;
- raw target NLL = `-log softmax(raw_scores)[target]`;
- raw score spread = max score - min score;
- raw-score entropy using softmax at temperature 1.

Calibration temperature is not folded into these raw geometry features.

### D. Candidate/action geometry

Per decision:

- target Action type;
- highest-scoring non-target Action type;
- candidate-set size;
- target-candidate token length;
- top-competitor token length;
- target-minus-competitor token-length difference.

Per aligned correction/sibling pair also record whether the top competitor Action type is
the same.

## Family aggregation

For every continuous decision-level feature, store separately for correction and sibling:

- min;
- mean;
- max.

For aligned correction/sibling pair features, store:

- min;
- mean;
- max.

Categorical features store counts and the deterministic mode; ties in the mode are stored
as an explicit sorted list rather than silently broken.

No endpoint-dependent aggregation rule may be chosen after outcome inspection.

## Pre-outcome confounding audit

Before joining outcomes, compute pairwise Spearman rank correlations among continuous
features and between each feature and `stratum_index`.

Use average ranks for ties.

Mark a feature `design_order_confounded=true` when, within either operation stratum,

`abs(Spearman rho(feature, stratum_index)) >= 0.95`.

Build feature-only collinearity clusters using edges

`abs(Spearman rho(feature_i, feature_j)) >= 0.95`

within the relevant analysis population. Connected components form one descriptive
cluster. Outcome interpretation is at the cluster level, not by counting every correlated
column as separate evidence.

### Why this is mandatory for the frozen manifest

A pre-outcome design audit of the frozen family definitions gives:

- add stratum: rho(stratum_index, left) = 1
- add stratum: rho(stratum_index, right) = 1
- add stratum: rho(stratum_index, result) = 1
- subtract stratum: rho(stratum_index, left) = 1
- subtract stratum: rho(stratum_index, right) = 1
- subtract stratum: rho(stratum_index, result) = 1
- subtract stratum: rho(stratum_index, absolute gap) = 1

Therefore operand/result magnitude cannot be interpreted as an independent static
explanation in this dataset merely because it correlates with the endpoint.

## Gate 2 — Outcome join

Only after Gate 1 artifacts exist, join the frozen family outcomes:

- primary accuracy delta:
  Correction - Replay uncorrected-sibling teacher-prefix accuracy;
- win / loss / tie;
- secondary NLL effect;
- Repair accuracy for Parent / Replay / Correction;
- Repair_CR = Correction - Replay.

The four DIAG v2 families receive their overlay only after the 30-family feature table and
descriptive screening are materialized.

## Descriptive analysis

No p-values are emitted in DIAG v3.

For each eligible continuous feature/cluster report:

- n families;
- number of distinct feature values;
- Spearman rho with primary accuracy delta for all 30 families;
- Spearman rho separately for add (n=15) and subtract (n=15);
- whether the sign is the same across strata;
- design-order-confounded flag;
- all 30 family value pairs in machine-readable output.

Spearman is used as a monotonic descriptive measure, not as evidence of causation.

For categorical features report:

- family counts;
- median primary delta by fixed category;
- win / loss / tie counts by fixed category.

Do not fit multivariate regression, a classifier, a feature selector, or an embedding in
this cycle.

## Operational lead rule

These thresholds are **diagnostic heuristics**, not significance thresholds.

An unconfounded continuous feature cluster becomes a `global_descriptive_lead` only if:

- `abs(rho_all30) >= 0.40`;
- add and subtract rho have the same sign;
- `min(abs(rho_add), abs(rho_sub)) >= 0.25`.

A cluster becomes a `stratum_specific_lead` when:

- one stratum has `abs(rho) >= 0.50`; and
- the other stratum does not satisfy the global-consistency rule.

Design-order-confounded clusters may be reported as leads but cannot by themselves
support `task_state_linked`.

## Four-family pair inspection

After the 30-family screening is fixed, show side-by-side feature differences for:

- 03 vs 15 within add;
- 02 vs 12 within subtract.

Then overlay the DIAG v2 update-depth outcomes.

This is case-comparison evidence only. It must not override the 30-family classification.

## Deterministic diagnostic classification

Classify exactly once at cycle closeout.

### `parent_geometry_linked`

Exactly one qualifying feature group remains after collinearity/confounding handling and
it is Parent Decision geometry.

Next: one Decision-head/candidate-geometry/objective causal Trial.

### `serializer_tokenization_linked`

Exactly one qualifying group remains and it is serializer/tokenizer geometry.

Next: one representation-controlled causal Trial, changing one serialization/tokenization
factor only.

### `task_state_linked`

Exactly one qualifying group remains and it is task/state, and at least one qualifying
cluster is **not** design-order-confounded.

Next: construct a matched family/state causal Trial that varies the implicated feature
without preserving the old ordinal coupling.

### `design_order_confounded`

The strongest task/state leads are all design-order-confounded and no other unconfounded
group qualifies.

Next: do not call magnitude causal. Build a new matched family design that breaks
family-order/operand-magnitude coupling before any model intervention.

### `mixed`

Two or more feature groups qualify, or strong stratum-specific leads point to different
groups.

Next: choose exactly one discriminating causal factor for a new Trial. Do not combine
interventions.

### `no_clear_static_explanation`

No unconfounded group qualifies and no design-order-confounded task lead dominates.

Next: return to causal update decomposition. First candidate is a 2x2
`embedding update x block update` Trial that adds the missing `embeddings + head`
condition while keeping the existing parent/data/objective fixed.

## No-update invariant

DIAG v3 must prove that it is read-only:

- no optimizer is constructed;
- model stays in evaluation/inference mode;
- no parameter has `requires_grad` enabled for training;
- model state digest before/after is identical;
- Tokenizer artifact digest before/after is identical;
- no new model artifact is promoted.

Any mutation invalidates the audit Run.

## Outputs

Local Run evidence should contain at minimum:

- `diag-family-feature-schema.json`
- `diag-family-features.json`
- `diag-family-feature-audit.json`

Repository closeout should add:

- `reports/diag-family-feature-audit-v3.json`
- `reports/diag-family-feature-audit-v3.md`

The machine-readable audit records:

- source Git SHA / dirty state;
- frozen feature-schema SHA-256;
- LEARN manifest SHA-256;
- parent artifact identity;
- exact 30 family rows;
- collinearity clusters;
- design-order-confounding flags;
- descriptive associations;
- four-family DIAG v2 overlay;
- one diagnostic classification;
- `causal_claim=false`;
- `confirmatory_verdict=null`;
- runtime/resource measurements.

## Implementation shape

Preferred small implementation:

- `src/aster/training/diag_family_features.py`
- `scripts/run_diag_family_feature_audit.py`
- `tests/model/test_diag_family_features.py`

Reuse existing LEARN evidence discovery/extraction and Decision scoring code. Do not copy
the 90-unit campaign parser into a second incompatible implementation.

## Stop rule

DIAG v3 stops when all of the following are true:

1. canonical 30 LEARN family outcomes reproduce exactly;
2. feature schema is frozen and hashed before outcome join;
3. all 30 families have deterministic feature rows;
4. no-update invariants pass;
5. pre-outcome confounding/collinearity audit is saved;
6. descriptive screening is complete;
7. four-family DIAG v2 overlay is complete;
8. exactly one diagnostic classification is recorded;
9. one next causal question is written;
10. no new training arm was introduced.

There is no Batch escalation inside DIAG v3. The audit exists to decide what the next
Trial should manipulate.

## Boundary

Do not introduce in DIAG v3:

- Adapter / LoRA;
- embedding-only or deeper-unfreeze training;
- replay-ratio changes;
- regularization;
- Decision objective changes;
- Tokenizer retraining;
- serializer redesign;
- model scaling;
- RL / KLPO;
- sealed-test access.

DIAG v3 explains the existing heterogeneity as far as the frozen evidence permits. It
does not solve Stable Plasticity itself.
