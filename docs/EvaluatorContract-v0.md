# Evaluator Contract v0

PR #10で、Student rolloutからshadow Teacherの介入記録、教師あり更新、再calibration、同一held-out benchmark比較までを閉じた。
次の段階ではrewardやRLへ進む前に、Agentが実際に経験した結果を、後から人間・Sites・学習器が再解釈できる評価層として固定する。

## 目的

Asterでは次の情報を混ぜない。

```text
model intent
    ↓
runtime Action
    ↓
Environment Observation
    ↓
Evaluation
    ↓
Teacher / Reward / Learning
```

`Observation`は実際に起きたこと、`Evaluation`はその証拠をどう判定したか、`Reward`は評価を学習信号へどう写像するかである。
このPRではRewardを導入しない。

## Step evaluation

既存の`EvaluationResult`をstep-levelの評価として維持する。

- `action_valid`: Actionがruntimeに受理されたか。
- `execution_success`: 受理されたActionの実行が成功したか。
- `goal_satisfied`: このstepまでにtask goalが検証済みか。
- `terminal`: このstepがterminal actionか。
- `notes`: 判定理由を補助する構造化可能な短い記録。

`Transition`は引き続き、`state_before -> action -> observation -> state_after -> evaluation`を一つのimmutableな経験として保存する。
`trajectory.jsonl`をstep-level evaluationのsource of truthとし、同じ内容を別JSONLへ複製しない。

runtime factとEvaluator判断の境界を壊さないため、`Transition`は次を不変条件として検査する。

```text
evaluation.action_valid      == observation.accepted
evaluation.execution_success == observation.ok
evaluation.terminal          == (action.kind == "stop")
```

また`goal_satisfied`はepisode内で累積的である。一度goalがverifiedになった後のTransitionが`goal_satisfied=false`へ戻るTrajectoryは不正とする。

## StepEvaluator protocol

runtimeは具体的な`TaskEvaluator`ではなく、次の契約だけに依存する。

```text
StepEvaluator.evaluate(
    task,
    state_after,
    action,
    observation,
    history,
) -> EvaluationResult
```

現在の`TaskEvaluator`はこのprotocolを構造的に満たす。
将来、code・math・file・webなど別のtask evaluatorへ差し替えてもAgent loop自体を変更しなくてよい。

Protocolは静的なinterface境界を表し、上記の値レベルのsemantic invariantは`Transition` / `Trajectory`がruntimeで強制する。

## Episode evaluation

stepごとのTransitionから、run全体を説明する`EpisodeEvaluation`を決定論的に導出する。

```text
trajectory.jsonl
    ↓ read back
Trajectory
    ↓
evaluate_episode(...)
    ↓
EpisodeEvaluation
```

v0の項目は次の通り。

- `task_success`
- `terminal_reached`
- `steps`
- `accepted_actions`
- `rejected_actions`
- `successful_executions`
- `failed_executions`
- `goal_verified`
- `first_goal_verified_step`
- `stop_reason`
- `notes`

`failed_executions`は、runtimeに受理されたが実行に失敗したActionだけを数える。拒否されたActionは`rejected_actions`へ分離する。

`task_success`は既存`Trajectory.successful`の意味を維持する。goalを途中で検証できてもterminal successに到達していなければtask successとはしない。

## Sites-readable artifact

logged runtimeは次を追加する。

```text
runs/<run-id>/
├── run.json
├── events.jsonl
├── trajectory.jsonl
├── routing.jsonl
├── interventions.jsonl   # intervention runのみ
├── evaluation.json       # NEW
└── selective.json / learning.json
```

`evaluation.json`は保存済み`trajectory.jsonl`を読み戻した結果から生成し、参照先とSHA-256を保持する。

```json
{
  "schema_version": "aster-episode-evaluation-0",
  "trajectory_file": "trajectory.jsonl",
  "trajectory_sha256": "...",
  "summary": {
    "task_success": true,
    "terminal_reached": true,
    "steps": 4,
    "accepted_actions": 4,
    "rejected_actions": 0,
    "successful_executions": 4,
    "failed_executions": 0,
    "goal_verified": true,
    "first_goal_verified_step": 2,
    "stop_reason": "goal_complete",
    "notes": ["goal_verified"]
  }
}
```

Sitesは`evaluation.json`をrun-level summaryとして読み、必要なら`trajectory.jsonl`を展開して各stepの根拠を確認できる。
`events.jsonl`の`episode_evaluated` eventはsummary自体を複製せず、`evaluation.json`への参照だけを記録する。

## Source-of-truth rule

persisted artifactの依存関係は次に固定する。

```text
trajectory.jsonl      # canonical step-level evidence
    ↓ deterministic materialization
evaluation.json       # run-level view
```

runtimeが保持しているin-memory `Trajectory`と同じ内容であることに暗黙依存せず、episode artifact生成時には必ず保存済みJSONLを読み戻す。
`trajectory_sha256`により、後からevaluationとtrajectory bytesの対応を監査できる。

## 再評価可能性

Episode evaluationはruntime内部の別状態を持たず、保存済みTrajectoryだけから導出する。
そのため将来、過去の`trajectory.jsonl`を別versionのEvaluatorで再評価する経路を作れる。

```text
saved trajectory
    ↓
Evaluator v0 / v1 / ...
    ↓
comparison
```

v0では再評価CLIやmigration機構までは実装しない。

## Rewardとの境界

このPRはscalar rewardを作らない。

```text
Evidence
    ↓
Evaluation vector
    ↓
Reward specification   # future
    ↓
scalar / component rewards
    ↓
policy update          # future
```

同じEpisodeEvaluationに異なるRewardSpecを適用できるようにし、Agent capabilityの変化とreward designの変化を区別する。
`evaluator/scoring.py`やRL objectiveはこのPRでは変更しない。

## Static typing boundary

このPRからPyrightをCIへ追加する。

- repo全体は`standard` modeで検査する。
- `StepEvaluator` protocolと`run_loop`は最初から`strict` modeで検査する。
- 値レベルのsemantic invariantはPyrightではなくpytest/runtime validationで保証する。

型checkerは`Policy`や`StepEvaluator`などのinterface contractを、pytestはserialization・値・behavioral semanticsを担当する。
strict範囲は今後、境界を型付けできたmoduleから段階的に広げる。

## v0で意図的に含めないもの

- scalar reward / reward shaping
- KL optimization / RL
- evaluator学習
- curriculum / hard-example mining
- counterfactual evaluation
- automatic promotion
- `evaluations.jsonl`のようなstep評価の重複保存
- Sites UI変更

## テストで保証すること

- Action rejectionとaccepted execution failureを別に数える。
- runtime factと`EvaluationResult`の不整合をTransitionが拒否する。
- goal verificationが後続stepで未検証へ戻らない。
- goal verificationの最初のstepを保持する。
- terminalとtask successを区別する。
- `evaluation.json`を保存済み`trajectory.jsonl`から導出する。
- `evaluation.json`がtrajectoryのSHA-256を保持する。
- EpisodeEvaluationのschemaをallowlistで固定し、rewardや学習判断のフィールド追加を明示的変更なしに許さない。
- selective / intervention logged runの双方が同じepisode evaluation artifactを出す。
- Pyrightでruntime-facing Protocolの型境界を検査する。

テスト成功はEvaluatorの知能やtask定義の妥当性を意味しない。保証するのはinterface、serialization、集計規則、provenance、semantic invariantの一貫性である。
