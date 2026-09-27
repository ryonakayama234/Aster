# Credit Assignment v0

Reward Contract v0はEpisodeEvaluationをscalar rewardへ写像するが、そのrewardをどのstep/actionへ帰属させるかは決めない。
Credit Assignment v0は、その次の境界として「直接観測から責任stepを特定できるrewardだけ」を配賦する。

```text
Trajectory
    ↓
EpisodeEvaluation
    ↓
RewardSpec
    ↓
RewardResult
    ↓
CreditAssigner: direct-evidence-v0
    ↓
CreditResult
    ↓
future: Counterfactual Evaluation / Policy Update
```

## 原則

v0は因果責任を推測しない。

Reward Contract v0のcomponentのうち、次はTransitionから対応stepを直接特定できる。

- `step_cost`: すべての実行step。
- `rejected_action`: `observation.accepted == false` のstep。
- `failed_execution`: `observation.accepted == true && observation.ok == false` のstep。

一方、次はepisode-level outcomeなのでv0ではstepへ配賦しない。

- `task_success`
- `task_failure`

これらは`unassigned`として明示的に残す。

## Reward conservation

`CreditResult`は次を必ず満たす。

```text
assigned_total + unassigned_total == reward_total
```

浮動小数点比較は小さな許容誤差を使うが、rewardを消したり新しく作ったりしない。

またRewardResultのcomponent quantityをTrajectoryから再計算し、一致しなければfailする。
これにより別run・別trajectoryから得たRewardResultを誤って接続した場合に黙ってcreditへ変換しない。

## Sites-readable artifact

Rewardを明示したlogged runでは、Credit Assignment v0も次を保存する。

```text
runs/<run-id>/
├── trajectory.jsonl
├── evaluation.json
├── reward.json
└── credit.json
```

`credit.json`はTrajectoryやReward本体を複製せず、入力artifactを参照する。

```json
{
  "schema_version": "aster-credit-0",
  "trajectory_file": "trajectory.jsonl",
  "reward_file": "reward.json",
  "result": {
    "method_id": "direct-evidence-v0",
    "reward_spec_id": "balanced-v0",
    "reward_total": 0.76,
    "steps": [],
    "unassigned": [],
    "assigned_total": -0.24,
    "unassigned_total": 1.0
  }
}
```

## 解釈

`StepCredit`は「このactionが最終成功を因果的に生んだ」という主張ではない。
保存しているのは、RewardSpecが明示的にstep-localな出来事へ課したreward contributionである。

特に`task_success`を最後のactionや全stepへ均等配分しない。
成功へのaction-level contributionは、後続のCounterfactual Evaluationで別candidateをfork実行し、continuation policyを固定した比較として扱う。

## v0で意図的に含めないもの

- task success/failureのheuristicなstep配賦
- discounted return-to-goを因果creditとして扱うこと
- counterfactual rollout
- learned value / critic
- advantage推定
- policy gradient / PPO / KLPO
- model weight更新
- Sites UI変更

## テストで保証すること

- direct componentが対応するobserved stepだけへ配賦される。
- episode-level outcomeが`unassigned`に残る。
- assigned + unassigned が元rewardを保存する。
- Reward component quantityとTrajectory由来quantityの不一致を拒否する。
- `credit.json`が入力artifactを参照し、入力内容を複製しない。
- rewardを要求しないrunではcreditも暗黙生成しない。

テスト成功はcreditが因果的に正しいことを意味しない。
v0が保証するのは、直接観測できる局所rewardと未解決のepisode rewardを混同しないことである。
