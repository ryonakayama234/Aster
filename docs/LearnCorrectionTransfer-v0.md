# LEARN v0 — Correction Transfer

2026-10-03。Issue #20 の研究仕様。ACT v0の次のGateとして、Asterが自分で訪れた失敗状態への訂正から何を学ぶかを、Replay controlと分離して測る。

## Research question

```text
student rollout
  ↓
student-visited state
  ↓
teacher correction
  ↓
candidate update
  ↓
uncorrected sibling state / model-only episode
```

中心質問は「訂正したstateを次回正解できるか」ではない。

> **Correctionを学習したことで、学習に含めていない近縁stateの判断・episode成功が、同じ追加学習量のReplayより改善するか。**

## Falsifiable hypotheses

### H1 — Correction Transfer

同一parent・同一追加update budgetで、`Delta_transfer(C1) > Delta_transfer(R1)` となる。

### H0 — No correction-specific transfer

Correction childの改善はReplay controlを上回らない。訂正caseだけ改善する場合もH1の支持にはしない。

## Arms

| arm | update | 役割 |
|---|---|---|
| P0 Parent | none | 基準 |
| R1 Replay | base supervisionの再提示 | 追加update自体の対照 |
| C1 Correction | base + student-visited correction | 訂正効果の候補 |

R1はC1のtrainable correction件数と同数のbase exampleをdeterministicにreplayし、training example列長を合わせる。R1/C1は同じparent、Tokenizer、serializer、candidate builder、DecisionTrainConfig、seed、optimizer step数を使う。

equal stepはequal FLOPsではない。token数、wall time、最大RSSも保存する。

## Evidence hierarchy

### L0 Repair

訂正state自身の target accuracy / NLL / margin。L0だけ改善してもmemorizationと区別できない。

### L1 Local Transfer

correctionへ入れていないsibling stateを事前固定して評価する。既存の数値変更・key変更・到達可能state診断は設計材料として使えるが、既存devを独立holdoutへ名称変更しない。新しいconfirmatory familyは元family/group単位でleakageを隔離する。

### L2 Sequential Transfer

model-only episode success / goal_verified / first error / failure propagationを比較する。teacher-prefixのdecision改善と、自分のActionで次stateを変えながらgoalへ到達する能力を分離する。

## Unit of evidence

seedや1 trajectory内のdecision stepを独立標本として数えない。基本単位はtask/generator family。

最初の1 family × 1 seedはdevelopment wiring probeで、能力結論には含めない。confirmatory実行前にfamily manifestとseed集合を固定する。seedは `42, 43, 44` を既定候補とし、結果を見て成功seedだけ追加しない。

統計検定は副次的。まずfamily単位のpaired effect、wins/ties/losses、全case evidenceを保存する。

## Existing components

- `run_logged_intervention_agent`
- `InterventionTrace`
- `examples_from_interventions`
- `train_intervention_candidate`
- DecisionModel Artifact
- ACT v0 model-only runtime
- TaskEvaluator / canonical trajectory/evaluation
- Research Observatory

ACTで固定したserializer/candidate/tool semanticsをLEARNの同じ実験内で変更しない。

## Gates

### LEARN-Wiring

Parent → student rollout → teacher labels visited state → Replay candidate / Correction candidate → save/reload → same evaluation suite → comparison evidence が一周する。

### LEARN-Repair

Correction childがcorrected stateを改善したか。Transferとは別判定。

### LEARN-Transfer

未訂正siblingでR1/C1を比較できる。結果は Supported / Not supported / Inconclusive のいずれでもよい。

### LEARN-Sequential

同じmodel-only episode suiteでR1/C1を比較し、最初の誤りと以後の状態変化を保存できる。

## Failure interpretations

- corrected ↑ / sibling ↔ : repairまたはmemorization。
- Replay ≈ Correction : 追加updateの効果で説明可能。
- teacher-prefix ↑ / model-only ↔ : sequential distribution shiftが残る。
- correctionがbase/seenを壊す : catastrophic regression。
- teacher Actionがcandidateに無い : ranking failureではなくcandidate coverage failure。
- train fit不成立 : transfer結論より先にoptimization/representation問題。

## Non-goals

RL、reward optimization、KLPO、new tokenizer、new serializer、model scaling、sealed test、automatic promotion、automatic curriculumは含めない。

## Artifacts

parent artifact / rollout + interventions / replay candidate artifact / correction candidate artifact / training metadata / corrected-state evaluation / sibling evaluation / model-only episode evaluation / comparison summary をlineage付きで残す。

Webはcanonical evidenceを表示するだけで再計算しない。
