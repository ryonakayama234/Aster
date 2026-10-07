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

DIAG v2 is never rerun or modified.

### Evidence recovery addendum — 2026-10-06

After this protocol was frozen, the original local machine-readable LEARN v0 Run store
(90 confirmatory units plus the completed campaign `confirmatory-results.json`) was
found unavailable on the WSL checkout. The published closeout summary and frozen
protocol remain in Git, but they do not contain the 30-family machine-readable table.

The ACT lineage is not lost: the original ACT Run
`5a8076571dca446fa07190cf4fc62509` still exists under the Service workbench Run
store, the exact content-addressed parent DecisionModel artifact remains registered, and
all 12 preserved DIAG v2 units independently reference that same ACT Run and parent
artifact.

Do not synthesize or infer the missing LEARN family rows from the published summary.
Gate 0 may instead consume one explicitly labeled **LEARN reproduction** only if all of
the following hold:

- it is run in an isolated clean worktree at the original measurement Git SHA
  `c2fa2d175b4a23bbd246ed61266d0cdda2e22846`;
- it uses the unchanged frozen manifest, exact preserved ACT Run, exact parent artifact,
  fixed seeds, and original runner;
- all 90 units complete under the original retry/failure rules;
- `runs/learn-v0-reproduction.json` explicitly marks the Run store as reproduction
  evidence and binds the original measurement SHA, frozen manifest, ACT Run, and exact
  parent artifact;
- the reconstructed result exactly matches that reproduction campaign's saved result;
- every preserved LEARN identity invariant matches exactly, including:
  - global wins / losses / ties = 6 / 19 / 5;
  - add wins / losses / ties = 5 / 7 / 3;
  - subtract wins / losses / ties = 1 / 12 / 2;
  - selected-family outcomes 03=loss, 15=win, 21=tie, 02=loss, 12=win, 22=tie;
  - median family accuracy delta = -0.25;
  - exact sign-test p = 0.01463329792022705;
  - verdict = Not supported;
- DIAG v3 records `learn_evidence_mode=reproduction` and never labels those rows as the
  original raw artifact.

Because the original 30-family machine-readable table is unavailable, a reproduction
cannot prove row-for-row equality to the lost original table. In reproduction mode the
audit must therefore record `original_family_results_exactly_verified=false`. The
reproduced family rows may be used for this post-hoc diagnostic only with that provenance
boundary explicit. No tuning, feature-schema change, or endpoint-dependent choice is
allowed in response to the reproduction.

## Evidence unit

The independent evidence unit is the **family**.

Seed, prefix, candidate, checkpoint, correction member, and sibling member are repeated
measurements or nested observations. They are never counted as independent samples.

## Gate 0 — Evidence identity before feature work

Before extracting new features:

1. select either original LEARN local evidence or the explicit reproduction source;
2. require original mode to read only the active Aster root; a separate evidence root must
   be explicitly declared as reproduction;
3. discover all 90 LEARN units read-only from that source;
4. verify every unit's Run and experiment provenance against the exact ACT Run, exact
   parent artifact, frozen family/seed, correction task, and sibling task;
5. reconstruct the source campaign's 30 family-level outcomes exactly;
6. verify the frozen manifest digest and original measurement Git SHA;
7. resolve the exact ACT parent artifact through the recorded Run lineage;
8. verify the DIAG v2 Run identity for the four-family overlay.

In original-evidence mode, the reconstructed 30-family outcomes must match the canonical
saved LEARN result exactly. In reproduction mode, the reconstructed result must match the
reproduction campaign exactly and all preserved original summary invariants must match;
the audit must not claim exact equality to the unavailable original family table. Stop on
any mismatch or partial reconstruction.

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

Each primitive measurement has exactly one owning feature group. Derived duplicates are
not registered under a second group. In particular, candidate-set size and candidate
token-length measurements belong only to Candidate/action geometry, not to
Serializer/tokenizer geometry.

Categorical features store counts and the deterministic mode; ties in the mode are stored
as an explicit sorted list rather than silently broken.

No endpoint-dependent aggregation rule may be chosen after outcome inspection.

## Pre-outcome confounding audit

Before joining outcomes, compute pairwise Spearman rank correlations among continuous
features and between each feature and `stratum_index`.

Use average ranks for ties. A population with fewer than two distinct feature values has
undefined Spearman rho; record `rho = null`, `degenerate = true`, and never promote
that population to a descriptive lead.

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

## Gate 1.5 — Persist pre-outcome confounding/collinearity audit

After raw feature extraction completes, but **before any outcome join**, materialize a
separate read-only audit Run from the completed feature Run. It must:

- verify the source feature Run is completed and still reports `outcome_joined=false`;
- verify the frozen feature-schema SHA-256 and exact 30-row shape;
- compute tie-aware Spearman summaries for all 30 families and separately for add/subtract;
- record degenerate populations as `rho=null, degenerate=true`;
- record within-operation feature-vs-`stratum_index` correlations;
- mark `design_order_confounded=true` at `abs(rho) >= 0.95` in either operation;
- persist pairwise continuous-feature Spearman values and deterministic connected
  components at `abs(rho) >= 0.95`;
- contain no outcome column, p-value, causal claim, or model update.

This intermediate artifact is `diag-family-preoutcome-audit.json`. The completed raw
feature Run remains immutable; Gate 1.5 creates a separate Run that points back to it.

## Gate 2A — Frozen outcome join and 30-family descriptive screening

Only after the Gate 1.5 pre-outcome artifact exists, join the frozen family outcomes:

- primary accuracy delta:
  Correction - Replay uncorrected-sibling teacher-prefix accuracy;
- win / loss / tie;
- secondary NLL effect;
- Repair accuracy for Parent / Replay / Correction;
- Repair_CR = Correction - Replay.

Persist this as a separate completed Run before reading the DIAG v2 overlay. The Gate 2A
artifact is `diag-family-outcome-screening.json` and must record
`diag_v2_overlay_joined=false` and `classification=null`.

For continuous columns, apply the preregistered lead rule to each column while retaining
its pre-outcome collinearity-component IDs for all30/add/subtract. Do not choose a
post-outcome "representative" column from a correlated component. For each pre-outcome
component, persist the set of qualifying member columns and the union of their qualifying
feature groups. A cross-group component remains cross-group; do not collapse it to a
single mechanism label.

Collinearity components are evidence-counting/interpretation units, not a feature-selection
step. Multiple qualifying columns in the same component are not counted as independent
support. The later classification consumes sets of qualifying feature groups rather than
the number of correlated columns.

For categorical features, use the frozen deterministic mode outputs (and the direct
family-level category for `operation`) to report family count, median primary delta, and
win/loss/tie counts per fixed category.

No p-values are emitted.

## Gate 2B — DIAG v2 overlay and exactly-one classification

Only after Gate 2A has been persisted may the four DIAG v2 families receive their overlay.
Then perform the fixed pair inspection and emit exactly one diagnostic classification.
The overlay is case-comparison evidence and must not alter or rerun the 30-family screen.

### Gate 2B implementation

Gate 2B consumes exactly one completed Gate 2A Run and the already-frozen DIAG v2
campaign. It does not recompute the 30-family screen. The deterministic classification
is computed from the persisted Gate 2A lead metadata before the four-family overlay is
attached; DIAG v2 case-comparison values cannot override that classification.

Run:

```bash
.venv/bin/python scripts/run_diag_family_closeout.py \
  --root . \
  --outcome-screening-run-id <completed-gate-2a-run-id>
```

The closeout writes:

- `runs/<run-id>/diag-family-feature-audit.json`;
- `reports/diag-family-feature-audit-v3.json`;
- `reports/diag-family-feature-audit-v3.md`.

The repository reports are generated from the same in-memory closeout object as the
Run artifact. They remain descriptive/post-hoc evidence and do not add a new training
arm or a confirmatory claim.

## Descriptive analysis

No p-values are emitted in DIAG v3.

For each eligible continuous feature report:

- n families;
- number of distinct feature values;
- Spearman rho with primary accuracy delta for all 30 families;
- Spearman rho separately for add (n=15) and subtract (n=15);
- whether the sign is the same across strata;
- design-order-confounded flag;
- pre-outcome collinearity-component membership for all30/add/subtract;
- all 30 family value pairs in machine-readable output.

For each pre-outcome collinearity component, also report its full member list, qualifying
member list, qualifying feature-group union, and whether the component crosses feature
groups. This cluster summary is deterministic metadata over the column-level screen; it
does not refit, select, or substitute a representative column.

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

A cluster becomes a `stratum_specific_lead` when either add or subtract has
`abs(rho) >= 0.50`, and at least one of the following holds for the opposite stratum:

- its rho is undefined/degenerate;
- `abs(rho_other) < 0.25`;
- its nonzero rho has the opposite sign.

If both strata have `abs(rho) >= 0.50` with opposite signs, store
`bidirectional_stratum_heterogeneity = true`; this is still one stratum-specific lead,
not two independent leads.

A same-sign pattern with the other stratum in `0.25 <= abs(rho_other) < 0.50` is neither
a global lead nor a stratum-specific lead unless the global rule itself is satisfied.

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

### `candidate_action_linked`

Exactly one qualifying group remains and it is Candidate/action geometry.

Next: one candidate/action-geometry causal Trial. Keep the parent model, serializer,
Tokenizer, training examples, and optimizer budget fixed; manipulate exactly one candidate
construction or action-representation factor.

### `task_state_linked`

Exactly one qualifying group remains and it is task/state, and at least one qualifying
cluster is **not** design-order-confounded.

Next: construct a matched family/state causal Trial that varies the implicated feature
without preserving the old ordinal coupling.

### `design_order_confounded`

No unconfounded feature group qualifies, at least one task/state cluster is a descriptive
lead, and every task/state lead is `design_order_confounded=true`.

Next: do not call magnitude causal. Build a new matched family design that breaks
family-order/operand-magnitude coupling before any model intervention.

### `mixed`

Two or more unconfounded feature groups qualify, or stratum-specific leads qualify
different groups in add versus subtract.

Next: choose exactly one discriminating causal factor for a new Trial. Do not combine
interventions.

### `no_clear_static_explanation`

No unconfounded group qualifies and the `design_order_confounded` rule is not satisfied.

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

- Gate 1 feature Run:
  - `diag-family-feature-schema.json`
  - `diag-family-features.json`
- Gate 1.5 pre-outcome Run:
  - `diag-family-preoutcome-audit.json`
- Gate 2A outcome-screening Run:
  - `diag-family-outcome-screening.json`
- final Gate 2B audit Run:
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
- `src/aster/training/diag_family_preoutcome.py`
- `src/aster/training/diag_family_outcome_screening.py`
- `scripts/run_diag_family_feature_audit.py`
- `scripts/run_diag_family_preoutcome_audit.py`
- `scripts/run_diag_family_outcome_screening.py`
- `tests/model/test_diag_family_features.py`
- `tests/model/test_diag_family_preoutcome.py`
- `tests/model/test_diag_family_outcome_screening.py`

Reuse existing LEARN evidence discovery/extraction and Decision scoring code. Do not copy
the 90-unit campaign parser into a second incompatible implementation.

## Stop rule

DIAG v3 stops when all of the following are true:

1. LEARN Gate 0 passes under either exact original evidence or explicitly labeled
   reproduction evidence with all preserved identity checks;
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
