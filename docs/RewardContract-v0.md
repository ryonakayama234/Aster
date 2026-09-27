# Reward Contract v0

保存済みTrajectoryから決定論的に`EpisodeEvaluation`を作る評価境界の後ろへ、明示的な価値判断を適用して説明可能な学習信号へ写像する最小境界を追加する。

```text
Trajectory
    ↓
EpisodeEvaluation
    ↓
RewardSpec
    ↓
RewardResult
    ↓
future: Credit Assignment / Policy Update
```

## 境界

Asterでは次を混ぜない。

- `Trajectory`: 実際に起きたstate / action / observation / step evaluation。
- `EpisodeEvaluation`: trajectoryから導出したrun-levelの判定・集計。
- `RewardSpec`: その判定の何をどれだけ価値とみなすかという明示的な設計。
- `RewardResult`: 一つのRewardSpecを一つのEpisodeEvaluationへ適用した結果。

`EvaluationResult`、`Transition`、`EpisodeEvaluation`へreward fieldは追加しない。
同じ保存済みEvaluationへ複数のRewardSpecを適用できることを優先する。

## RewardSpec v0

v0は小さな線形specだけを持つ。

```text
task_success
task_failure
rejected_action
failed_execution
step_cost
```

例えば、

```text
task_success      +1.00 × 1
failed_execution  -0.20 × 1
step_cost         -0.01 × 4
--------------------------------
total              0.76
```

これは唯一の正しいrewardではない。同じEvaluationに別specを適用すれば別のtotalになり得る。
その差はAgent capabilityの変化ではなく、reward designの差として扱う。

## componentを保存する理由

scalarだけを保存すると、後から「なぜ0.76だったのか」を復元しにくい。
そのため`RewardResult`は各componentについて次を保持する。

- `name`
- `weight`
- `quantity`
- `amount = weight * quantity`
- component合計としての`total`

Sitesや後続の学習器はtotalだけでなく、その内訳を監査できる。

## Pure mapping

`compute_reward(evaluation, spec)`はmodel、Torch、runtime state、optimizerを知らないpure mappingとする。
同じEvaluationと同じSpecからは同じRewardResultを返す。

Reward Contract v0はcredit assignmentを行わない。
Episode rewardが得られても、そのrewardをどのstep/actionへ帰属させるかは別問題として残す。

```text
RewardResult
    ↓
future CreditAssigner
    ↓
future Policy Update
```

## Sites-readable artifact

Rewardを要求したlogged runでは次を保存する。

```text
runs/<run-id>/
├── trajectory.jsonl
├── evaluation.json
└── reward.json
```

`reward.json`は概念的に次の形を持つ。

```json
{
  "schema_version": "aster-episode-reward-0",
  "trajectory_file": "trajectory.jsonl",
  "evaluation_file": "evaluation.json",
  "spec": {
    "spec_id": "balanced-v0",
    "task_success": 1.0,
    "task_failure": 0.0,
    "rejected_action": -0.1,
    "failed_execution": -0.2,
    "step_cost": -0.01
  },
  "result": {
    "spec_id": "balanced-v0",
    "components": [],
    "total": 0.76
  }
}
```

Evaluation本体は複製せず`evaluation_file`で参照する。
`trajectory.jsonl`は引き続きstep-level evidenceのsource of truthである。

## runtimeへの接続

RewardSpecは暗黙には選ばない。
logged runtimeへ`reward_spec=None`なら従来どおりreward artifactを作らない。
明示的にRewardSpecを渡したrunだけ、episode evaluationの後に`reward.json`を生成する。

これにより、runtimeの実行意味論とreward designを分離したままSitesへ観測可能にする。

## v0で意図的に含めないもの

- step/actionごとのcredit assignment
- policy gradient / PPO / KL optimization / KLPO
- value model / critic
- reward shapingの自動探索
- learned reward model
- evaluator学習
- curriculum / hard-example mining
- rewardに基づくcandidate modelの自動昇格
- Sites UI変更

## テストで保証すること

- componentの合計とscalar totalが一致する。
- 同じEvaluationへ異なるSpecを適用できる。
- Reward計算が元のEvaluationを変更しない。
- NaN / infinityのweightを拒否する。
- artifactがtrajectory / evaluationを参照し、Evaluation本体を重複保存しない。
- rewardを要求したselective / intervention runで同じReward Contractを使える。

テスト成功はRewardSpecの価値判断が正しいことや、Agent能力が向上することを意味しない。
保証するのは境界、計算、serializationの一貫性である。
