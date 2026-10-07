# DIAG v3 — Family/State Feature Audit closeout

## 結論

- diagnostic classification: `parent_geometry_linked`
- `causal_claim=false`
- `confirmatory_verdict=null`
- p-value / confirmatory retest: なし
- new training arm: なし

この分類は保存済みGate 2Aの30-family descriptive screeningだけから決定する。
固定済み4-family DIAG v2 overlayは別のcase-comparison evidenceであり、分類を変更しない。
いずれもpost-hoc diagnostic evidenceであり、因果効果の証明ではない。

## 30-family operational leads

| feature | group | rho all30 | rho add | rho subtract | lead | design-order confounded |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `absolute_operand_gap` | `task_state` | -0.581175 | -0.805129 | 0.00382192 | `stratum_specific_lead` | true |
| `correction_raw_target_nll_min` | `parent_geometry` | 0.539375 | 0.711802 | 0.105103 | `stratum_specific_lead` | false |
| `sibling_best_non_target_raw_score_max` | `parent_geometry` | -0.535426 | -0.507368 | -0.17963 | `stratum_specific_lead` | false |
| `sibling_raw_score_entropy_mean` | `parent_geometry` | 0.557029 | 0.46834 | 0.292377 | `global_descriptive_lead` | false |
| `sibling_raw_score_entropy_min` | `parent_geometry` | 0.529969 | 0.58275 | 0.31913 | `global_descriptive_lead` | false |
| `sibling_raw_score_spread_max` | `parent_geometry` | -0.559816 | -0.572416 | -0.202562 | `stratum_specific_lead` | false |
| `sibling_target_margin_min` | `parent_geometry` | 0.562836 | 0.392142 | 0.39748 | `global_descriptive_lead` | false |

## Four-family pair inspection

Machine-readable JSONには全screening featureのside-by-side比較を保存する。
ここではoperational leadのみを要約し、その後にDIAG v2 overlayを示す。

### learn-confirm-keyshift-03 vs learn-confirm-keyshift-15 (add)

| lead feature | loss | control | control-loss |
| --- | ---: | ---: | ---: |
| `absolute_operand_gap` | 13 | 1 | -12 |
| `correction_raw_target_nll_min` | 1.09164 | 1.10482 | 0.0131828 |
| `sibling_best_non_target_raw_score_max` | -1.02757 | -1.05611 | -0.0285333 |
| `sibling_raw_score_entropy_mean` | 1.37009 | 1.37014 | 5.07832e-05 |
| `sibling_raw_score_entropy_min` | 1.09858 | 1.09861 | 2.7895e-05 |
| `sibling_raw_score_spread_max` | 0.0441955 | 0.0215937 | -0.0226018 |
| `sibling_target_margin_min` | -0.0361993 | -0.0196086 | 0.0165907 |

DIAG v2 overlay:

| family | condition | D sibling margin | Repair_CR |
| --- | --- | ---: | ---: |
| `learn-confirm-keyshift-03` | `head-only` | -0.00290684 | 0.25 |
| `learn-confirm-keyshift-03` | `last-block` | -0.107558 | 0.25 |
| `learn-confirm-keyshift-03` | `full` | -1.61782 | 0.5 |
| `learn-confirm-keyshift-15` | `head-only` | -0.00651377 | 1 |
| `learn-confirm-keyshift-15` | `last-block` | -0.504249 | 1 |
| `learn-confirm-keyshift-15` | `full` | 0.0689182 | 1 |

### learn-confirm-keyshift-02 vs learn-confirm-keyshift-12 (subtract)

| lead feature | loss | control | control-loss |
| --- | ---: | ---: | ---: |
| `absolute_operand_gap` | 181 | 211 | 30 |
| `correction_raw_target_nll_min` | 1.08313 | 1.09595 | 0.0128195 |
| `sibling_best_non_target_raw_score_max` | -0.997327 | -0.977006 | 0.0203208 |
| `sibling_raw_score_entropy_mean` | 1.37001 | 1.36997 | -4.37796e-05 |
| `sibling_raw_score_entropy_min` | 1.09855 | 1.09857 | 1.84774e-05 |
| `sibling_raw_score_spread_max` | 0.0815531 | 0.100473 | 0.0189195 |
| `sibling_target_margin_min` | -0.0713227 | -0.0659577 | 0.00536495 |

DIAG v2 overlay:

| family | condition | D sibling margin | Repair_CR |
| --- | --- | ---: | ---: |
| `learn-confirm-keyshift-02` | `head-only` | -0.0109847 | 0 |
| `learn-confirm-keyshift-02` | `last-block` | 0.548828 | -0.333333 |
| `learn-confirm-keyshift-02` | `full` | -0.805075 | 0.333333 |
| `learn-confirm-keyshift-12` | `head-only` | -0.00906236 | 0 |
| `learn-confirm-keyshift-12` | `last-block` | -0.000826299 | 0 |
| `learn-confirm-keyshift-12` | `full` | -0.00353563 | 0.5 |

## Next causal question

Does one controlled change to parent Decision geometry causally change Correction-vs-Replay sibling transfer while data, serializer, Tokenizer, and optimizer budget stay fixed?

## Provenance boundary

- Gate 2A Run: `e89f1370cfae4aac8e151cc87fef20c4`
- DIAG v2 Run: `8100ff7fb195408995513bc0c40da901`
- feature schema SHA-256: `435814be88512282a8263a9b9fbf96a48f666f7280e2d479e0f0ad2b4d053855`
- parent artifact: `decision_model:e28fb30d64f9b9f00c71c232e5200648eed2d208922909e065dbd65d7c5d4bf0`
