# Intervention Learning / DAgger v0

PR #8でDecisionのgeneralization / calibrationを測り、PR #9でcalibrated confidenceをruntime controlへ接続した。
この段階では、Agentが実際に遭遇したstateは記録できても、その経験を次のDecision policyへ戻す学習loopはまだ閉じていなかった。

Intervention Learning v0は、次の最小loopを追加する。

```text
Student v0 rollout
    ↓
student-induced states
    ↓
shadow Teacher label
    ↓
InterventionTrace
    ↓
DecisionExample
    ↓
D0 + intervention examples
    ↓
supervised update
    ↓
Student v+1 candidate
```

## 重要な境界

実行の真実と学習の教師情報を混ぜない。

- `Transition.action`: runtimeが実際に実行したAction。
- `DecisionTrace`: student modelのcandidate scores / probabilities / selected Action。
- `RoutingTrace`: confidence gateがmodel / fallback / abstainのどこへ送ったか。
- `InterventionTrace`: 同じpre-action state/historyについてshadow Teacherが選んだAction。

一つのstepでは`student_action`、`executed_action`、`teacher_action`が異なり得る。

fallbackとTeacherも別の役割として扱う。fallbackはruntimeで今何を実行するかを決める。Teacherはlearningで何をtargetにするかを決める。v0では両方に`RuleBasedPolicy`を使えても、interface上は分離する。

## shadow Teacher

TeacherはStudentが訪れた全stateで問い合わせる。fallbackが発火した場所だけに限定しない。
これによりhigh/low confidenceとTeacher agreement/disagreementを区別できる。特にhigh-confidence disagreementは、calibration / selective execution / learningが接続する重要な観測である。

TeacherはActionを返すだけで、環境を実行しない。過去のtrajectoryも書き換えない。Studentが間違ったActionを実行して別stateへ進んだ場合、次のTeacher queryはその実際のstudent-induced stateに対して行う。

## InterventionTrace

`InterventionTrace`はstate/historyとcandidate集合に加えて、student scores、raw/calibrated probabilities、student Action、route、executed Action、Teacher Actionを保存する。

Teacher Actionがstudent candidate集合に存在すれば`teacher_target_index`を持ち、既存`DecisionExample`へ変換できる。
存在しない場合もrolloutを失敗させず、`trainable=false`の観測として保存する。candidate-generation側の表現力不足とpolicyの選択ミスを混同しないためである。

## Dataset aggregation

デフォルトではTeacher disagreementだけでなく、trainableな全visited stateを追加する。

```text
D1 = D0 + I0
D2 = D1 + I1
...
```

v0では既存例を静かに置換・deduplicateしない。重複除去、reweighting、replay sampling、hard-example miningは別の実験として扱う。

## v+1 candidate

`train_intervention_candidate(...)`はparent `DecisionModel`をdeep copyし、既存`train_decision(...)`のcross-entropy学習を再利用する。parent modelは変更しない。返るのはcandidate modelと、aggregate / intervention例に対するbefore/after metricsである。

このPRでは新しいRL objectiveを導入しない。正しいTeacher Actionを直接得られるため、まず教師あり更新で閉じる。Rewardだけからpolicyを更新するKL-regularized RLは次段階とする。

## Sites-readable rollout artifacts

`run_logged_intervention_agent(...)`は次を保存する。

```text
runs/<run-id>/
├── run.json
├── events.jsonl
├── learning.json
├── trajectory.jsonl
├── routing.jsonl
└── interventions.jsonl
```

`learning.json`には、task success、route counts、Teacher agreement/disagreement、trainable example数、Teacher Actionがcandidateに無かった件数、high-confidence disagreement件数を保存する。

## 評価

v+1を作っただけでは採用しない。PR #8の固定benchmarkへcandidateを再投入し、calibration splitだけでtemperatureを再fitして、held-out accuracy / NLL / Brier / ECE / risk-coverage / distribution-shift slicesをv0と比較する。

intervention例への適合が改善してもheld-out generalizationが悪化する可能性がある。その場合も、局所適応とgeneralizationを分離して観測できたことがこのloopの目的に含まれる。

## v0で意図的に含めないもの

- reward / RL / KL optimization
- Teacher自身の学習
- counterfactual trajectory生成
- intervention datasetの自動reweighting / curriculum
- candidate modelの自動production昇格
- Sites UI変更
- DecisionModel checkpoint永続化の新schema
