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


## Development probe v0 — predeclared

Confirmatory familyを設計する前に、既存の実ACT parentを使って1 family × 1 seedの配線確認を行う。この結果は能力結論・family勝率・confirmatory統計へ算入しない。

### Lineage

- ACT Run ID: `5a8076571dca446fa07190cf4fc62509`
- `runs/<ACT Run ID>/run.json` またはService実行時の `runs/workbench/<JOB_ID>/runs/<ACT Run ID>/run.json` を解決し、その `inputs.decision_model_artifact_id` を唯一のparent Artifact参照とする。
- Run IDからArtifact IDを推測・再生成しない。
- completed / `agent-decision-model-v0` / model-only / `act-calculate-store-dev-v0` を満たさないRunは拒否する。

### Fixed family

`learn-dev-key-shift-family-v0`

Correction member:

```json
{"operation":"add","left":2,"right":3,"store_as":"diagnostic_total"}
```

Uncorrected sibling:

```json
{"operation":"add","left":2,"right":3,"store_as":"sibling_total"}
```

数値・operation・tool semanticsを固定し、保存keyだけを変える。Correction rolloutへsiblingは入れない。

### Fixed training budget

- seed: 42
- optimizer steps: 100
- learning rate: 0.003
- train_backbone: true
- max episode steps: 8
- Replay/Correction added-example countを一致
- model-only; fallbackなし

### Metrics

- Repair: student-visited exact correction statesをP0/R1/C1で直接再評価。
- Local Transfer: uncorrected sibling teacher-prefix decisionを同一suiteで比較。
- Sequential Transfer: sibling model-only episodeをP0/R1/C1別に保存し、strict terminal successとnonterminal goal-reachingを分離して読む。

benchmark runner内部の `test` splitはこのdevelopment-only sibling測定の保持場所であり、project sealed testを意味しない。旧sealed testは開かない。

### WSL entry point

```bash
.venv/bin/python scripts/run_learn_correction_transfer_dev.py \
  --root . \
  --act-run-id 5a8076571dca446fa07190cf4fc62509 \
  --threads 2
```

training条件はCLI引数で変更できない。


### Model-only calibration semantics

ACT v0は`ModelPolicy`を直接使い、Actionはraw score argmaxで選択する。LEARN development probeはtrace収集のため`SelectivePolicy`を使うが、fallbackなし・threshold 0で常にmodel routeとする。

このためrouting temperatureはidentity `1.0` に固定し、parent Artifactのcalibration temperatureをAction選択やprobe実行可否へ使わない。Artifact calibrationはprovenance diagnosticとしてのみ保存する。これによりACT parentと同じAction-selection semanticsを保つ。


## Sequential Transfer v0 — predeclared development metrics

teacher-prefix Local Transferを見た後もtraining条件は変更しない。Sequential development probeは同一P0/R1/C1を、未訂正sibling taskへmodel-onlyで実行する。

保存する主項目:
- task_success / goal_verified
- episode steps / stop_reason
- shadow teacher agreement / disagreement
- first_teacher_divergence_step
- steps_after_first_teacher_divergence
- first_tool_failure_step
- final memory
- DecisionTrace / Transition alignment
- evaluation前後のweights unchanged

Teacherはshadow label専用で、Action executionへ介入しない。

R1/C1は非promotion candidateとしてDecisionModel Artifactを保存し、ArtifactCatalogへimmutable登録した後にreloadする。reload weight一致を確認し、そのreload後model/tokenizerでSequential episodeを実行する。P0も既存registered parent Artifactを再resolve/reloadして同じ検証を行う。

この追加観測はdevelopment probeのinstrumentation拡張であり、既に見たLocal Transfer結果に合わせたlearning hyperparameter tuningではない。


## Development Gate 2 closeout — 2026-10-04

Fixed development Run `720c9e43a8a248a48829fc6b789ba75f` で、実ACT parent lineageからP0/R1/C1 Artifactのsave/register/reload、Repair、teacher-prefix sibling、model-only sibling episode、trajectory diagnosisまで一周した。

### Observed / not demonstrated

- L0 Repair: observed。
- L1 Local Transfer: not demonstrated。
- L2 strict terminal task success within the fixed 8-step horizon: not demonstrated。
- L2 sequential goal-reaching: C1だけでobserved。
- capability claim: none。1 family × 1 seedのdevelopment evidenceでありconfirmatory結果へ算入しない。

C1はstep 7で初めて `memory.get("sibling_total")` に到達してgoal_verified=trueとなった。step 7は固定8-step horizonの最終許容stepなので、その後のstop decisionは観測されていない。したがって「goal到達後にterminationを誤った」とは結論しない。

8-step horizonはこの結果を見て延長しない。追加horizon診断を行う場合はpost-hoc development diagnosticとして元の8-step結果と分離する。

既存development specは `task_success` / `goal_verified` の両軸を保存対象にしていたが、primary/secondary hierarchyは明示していなかった。Confirmatory manifestではepisode horizonとendpoint hierarchyを実測前に固定する。

### Gate transition

Gate 2はwiring/instrumentationの成立を目的として完了とする。次のGate 3では、結果を見る前に以下をmanifestへ固定する:

- independent task/generator familyとleakage group
- correction member / uncorrected sibling生成規則
- candidate coverage
- seed集合
- episode horizon
- endpoint hierarchy
- family-level win / tie / loss規則
- failed/missing runの扱い
- sealed-test boundary

Matched-controlのtoken count / wall time / RSSは、PR #51のreviewを受けてGate 3着手前にinstrumentationを追加した。以後のR1/C1 training artifactとcorrection-transfer summaryへ、encoded token presentations / padded token positions / candidate sequence数 / wall time / process max RSS before/after/increaseを保存する。optimizer step一致をequal FLOPsとは解釈しない。既存development Runはinstrumentation追加前の実測なので、これらのresource値を後から推測して補わない。


## Confirmatory Gate 3 — frozen protocol v0（2026-10-04）

confirmatory測定前の正本を
`docs/experiments/learn-correction-transfer-confirmatory-v0.json`
として固定する。canonical JSON SHA-256は
`72fa77acf48403d5a45f927f1122d6fb5753b1118d9f25322e1d6700f2cb1563`。
`src/aster/training/learn_confirmatory.py` と自動テストがschema、主要条件、family構造、hashを検査する。

### Frozen family design

- 30 independent experimental familyを明示列挙し、測定後に生成・追加・差し替えしない。
- 15 add / 15 subtract。
- 各familyで operation / left / right をcorrectionとsiblingで同一にし、`store_as` keyだけを変える。
- development family `learn-dev-key-shift-family-v0` はconfirmatory結果へ含めない。
- sibling memberはtrainingへ入れない。
- seedsは `42, 43, 44`。seedやtrajectory内stepを独立familyとして数えない。
- optimizer steps=100、LR=0.003、train_backbone=true、model-only、fallbackなし、8-step horizonを固定する。
- serializerは `aster-decision-input-0`、candidate builderは `calculate-and-store-v0` を維持する。
- parentはACT Run `5a8076571dca446fa07190cf4fc62509` の記録したlogical DecisionModel Artifactから解決し、current `calculate-and-store-v0` suiteのdigestとartifact manifestの `suite_sha256` を一致要求する。

### Candidate coverage Gate

全30 familyのcorrection/sibling、計60 taskについて、RuleBased teacher trajectoryから作る全teacher Actionがcandidate setに存在することを測定前に検査する。現在の固定manifestは60 task / 240 decisionでcoverageが成立することを自動テスト対象とする。

candidate coverage failureはranking/transfer failureとして数えず、protocol-invalid familyとして扱う。

### Endpoint hierarchy

confirmatory primaryは **L1 Local Transfer only**。

`uncorrected sibling teacher-prefix accuracy` の
`C1 Correction - R1 Replay` を各seedで測り、固定3 seedのfamily内平均差をfamily outcomeへ変換する。

- positive: win
- negative: loss
- zero: tie

L0 Repairはmanipulation check。L1 raw NLL、L2 strict terminal `task_success`、L2 `goal_verified` はsecondary/descriptiveとし、primary hypothesis testを増やさない。

### Confirmatory decision rule

primary testはnon-tied familyだけに対するexact two-sided sign test、alpha=0.05。

- Supported: p < 0.05、wins > losses、かつmedian family accuracy delta > 0。
- Not supported: p < 0.05、losses > wins。
- Inconclusive: 上記以外、またはnon-tied family < 20。

tieは隠さず件数を報告し、test分母からのみ除く。fixed familyを別familyで補充しない。

Wolframでの事前planningでは、独立family・tieなしという単純化の下、n=30でtrue win probability 0.75ならpower≈0.8034、0.80なら≈0.9389。これはsample-size planningだけで、development evidenceへの推論ではない。

### Failure / retry boundary

- modelの誤Action、policy_stop、tool/task failureはbehavioral outcomeとして残す。
- candidate coverage failureはprotocol invalidでありwin/lossにしない。
- infrastructure failureはendpoint metric出力前、同一code/manifestの場合だけ同じfamily/seedをretryできる。全attemptを記録する。
- missing familyを新生成familyで置換しない。
- 最初のconfirmatory endpointを測定した後にcodeまたはmanifestを変更した場合、v0のconfirmatory claimを継続せずprotocol versionを上げて再開する。
- project sealed testは開かない。

このGate 3は「結果を得るGate」ではなく、**結果を見てから研究質問を動かせなくするGate**である。Gate 4で初めてこのmanifestを入力としてconfirmatory measurementを行う。
