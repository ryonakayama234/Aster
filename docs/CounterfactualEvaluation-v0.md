# Counterfactual Evaluation v0

Credit Assignment v0は、step-localなrewardだけを直接観測から配賦し、task success/failureを`unassigned`に残す。
Counterfactual Evaluation v0は、その未解決部分をいきなりcriticやRLへ渡さず、保存済みdecision stateから別candidateを隔離branchで実行して比較する実験境界を追加する。

```text
saved Trajectory + RoutingTrace
          │
          ├── candidate Aを強制
          ├── candidate Bを強制
          └── candidate Cを強制
                    │
          fixed continuation policy
                    │
                    ▼
        branch Evaluation + Reward
                    │
                    ▼
        candidate reward comparison
                    │
                    ▼
        policy-weighted baseline
        + advantage estimate
```

## 何をcounterfactualと呼ぶか

source step `t`の直前stateをforkし、RoutingTraceに保存された各candidate Actionを一度だけ強制する。
その後は全branchで同じcontinuation policyを使う。

したがって比較対象は、

```text
forced candidate action
+ fixed continuation policy
+ fixed RewardSpec
```

である。

v0ではこれを一般的な最適`Q*(s,a)`とは呼ばない。
保存する値は「強制actionと固定continuation policyの下で得たfull-episode reward」であり、continuation policyとhorizonに依存する。

## baseline / advantage estimate

あるsource stepで全candidateを安全に実行できた場合、保存済みmodel probability `pi(a|s)`を使って、

```text
baseline = Σ_a pi(a|s) * branch_reward(a)
advantage_estimate(a) = branch_reward(a) - baseline
```

を計算する。

branch rewardにはsource step以前のepisode reward contributionも含まれるが、同じsource stepのcandidate間ではprefixが共通なので、baselineとの差分では共通項が相殺される。
このためv0はabsolute branch rewardとadvantage estimateを分けて保存する。

一つでもcandidateが安全に実行できなければ、そのstepではbaseline / advantageを作らない。
未評価candidateを除外して確率を再正規化すると、元model policyとは別の分布になるためである。

## runtime refactor

通常Agent loopから、policy選択を含まない一step実行primitiveを切り出す。

```text
execute_step(
    action already selected,
    history,
    context,
    executor,
    evaluator,
) -> Transition
```

通常runもcounterfactual branchも同じprimitiveを使い、Action execution / Observation / Evaluationの意味論を複製しない。
通常`run_loop`の外部contractは変えない。

## state fork

現在のRuntimeState / RuntimeContextはtaskとmemoryをdeep-copyするため、calculate-and-store環境はローカルにforkできる。

```text
source RuntimeState
      │
      ├── RuntimeContext A
      ├── RuntimeContext B
      └── RuntimeContext C
```

元runのRuntimeContextやTrajectoryは変更しない。

## counterfactual-safe tools

stateをcopyできても、外部副作用はcopyできない。
そのためToolSpecへfail-closedなmetadataを追加する。

```text
counterfactual_safe = False   # default
```

v0で明示的にsafeとするのは、現在のローカル環境だけである。

- calculator
- memory.put
- memory.get

将来のfilesystem / shell / browser / email / GitHubなどは、個別に隔離・rollback semanticsを設計するまで自動的にはcounterfactual実行しない。
unsafe candidateやunsafe continuation actionは実行せず、`unsupported_counterfactual_tool`として証拠を残す。

## 明示実行だけにする理由

Counterfactual Evaluationはcandidate数だけ追加実行を発生させる。
通常logged rolloutへ暗黙接続すると、run costと副作用面積が急増する。

そのためv0は通常runとは別の`counterfactual_analysis` runとして明示的に起動する。

## artifact

```text
runs/<analysis-run-id>/
├── run.json
├── events.jsonl
├── counterfactual.json
├── counterfactuals.jsonl
└── branches/
    └── step-000-candidate-000/
        ├── trajectory.jsonl
        ├── evaluation.json
        └── reward.json
```

`counterfactuals.jsonl`ではcandidateごとに次を対応づける。

- source step
- model ID / source route
- candidate Action
- calibrated model probability
- model-selected candidateか
- source runで実際に実行されたActionか
- continuation policy
- completed / unsupported status
- branch reward
- task success / terminal
- policy-weighted baseline
- advantage estimate

## validation scope

実装テストでは、calculate-and-storeの決定論的環境でcandidateごとのbranch reward、model probabilityによるbaseline、advantage estimateを数値まで固定して確認する。
またcounterfactual-safeでないtest toolのhandlerが一度も呼ばれないことを確認する。

これらのテストはbranch実行・serialization・安全境界の整合性を保証するもので、counterfactual advantageが一般環境で因果効果や最適Q値を表すことまでは保証しない。

## v0で意図的に含めないもの

- unsafe external toolsのsandbox実行
- stochastic環境のMonte Carlo反復
- learned critic / value model
- `Q*`推定という主張
- counterfactual rewardからの自動学習
- KL / PPO / KLPO update
- model自動昇格
- Sites UI変更

## 次の境界

このartifactが得られれば、次に初めて

```text
model pi(a|s)
+ counterfactual advantage estimate
        ↓
KL-regularized target policy
        ↓
future Policy Update v0
```

を、観測可能な入力を持った状態で設計できる。
