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
    ↓
re-calibration + same held-out benchmark
    ↓
v0 / v+1 comparison artifact
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

JSONLは`read_intervention_traces_jsonl(...)`で再読込できる。runtime runとtrainer runを別runとして保持し、後続学習は保存済みintervention evidenceを入力にできる。

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

`run_logged_intervention_agent(...)`はruntime evidenceを独立runとして保存する。

```text
runs/<rollout-run-id>/
├── run.json
├── events.jsonl
├── learning.json
├── trajectory.jsonl
├── routing.jsonl
└── interventions.jsonl
```

`learning.json`には、task success、route counts、Teacher agreement/disagreement、trainable example数、Teacher Actionがcandidateに無かった件数、high-confidence disagreement件数を保存する。

## v0 → v+1比較experiment

`run_logged_intervention_learning_experiment(...)`は一つの学習experimentとして、上のruntime rolloutを子runに持ち、その`rollout_run_id`をlineageとして保存する。runtimeの実測ログをtrainer側へ複製・改変しない。

その後、同じ親modelから次を実行する。

1. parent v0を固定benchmarkで評価する。
2. interventionを既存datasetへaggregateしてdeep-copied v+1 candidateを学習する。
3. candidateを同じbenchmarkへ投入する。
4. v0とv+1について、それぞれcalibration splitだけでtemperatureを独立にfitする。
5. 同じtest cases / slicesについてraw・calibrated metricsを比較する。

```text
runs/<experiment-run-id>/
├── run.json
├── events.jsonl
├── experiment.json
├── training.json
├── training-examples.jsonl
├── benchmark-parent/
│   ├── benchmark.json
│   ├── predictions.jsonl
│   └── calibration.json
└── benchmark-candidate/
    ├── benchmark.json
    ├── predictions.jsonl
    └── calibration.json
```

`experiment.json`のdeltaはすべて`candidate - parent`として保存する。ただしdelta自体を「改善」「採用可否」という判定へ変換しない。accuracy、NLL、Brier、ECEは方向の意味が異なり、sliceごとのtrade-offもあるためである。

candidateは`promotion = candidate_only`のまま。自動production昇格は行わない。

## 評価

v+1を作っただけでは採用しない。PR #8の固定benchmarkへcandidateを再投入し、calibration splitだけでtemperatureを再fitして、held-out accuracy / NLL / Brier / ECE / risk-coverage / distribution-shift slicesをv0と比較する。

intervention例への適合が改善してもheld-out generalizationが悪化する可能性がある。その場合も、局所適応とgeneralizationを分離して観測できたことがこのloopの目的に含まれる。

v0とv+1のbenchmark comparisonは、同一suite・同一example count・同一slice集合でなければ失敗させる。比較条件が変わった結果をmodel更新の効果として表示しないためである。

## v0で意図的に含めないもの

- reward / RL / KL optimization
- Teacher自身の学習
- counterfactual trajectory生成
- intervention datasetの自動reweighting / curriculum
- candidate modelの自動production昇格
- Sites UI変更
- DecisionModel checkpoint永続化の新schema
