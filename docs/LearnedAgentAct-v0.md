# Learned Agent ACT v0

2026-10-02。Issue #48 の実装仕様。Research Observatory（SEE）の次のGateとして、保存済みDecisionModelをAster Agent Runtimeの実Tool実行へ接続する。

## 目的

Asterの既存部品を次の一本の経路へ接続する。

```text
DecisionModel Artifact
  ↓ verified load
ModelPolicy
  ↓
RuntimeState + Trajectory
  ↓
CalculateAndStoreCandidates
  ↓
aster-decision-input-0
  ↓
candidate scores / probabilities
  ↓
selected Action
  ↓
ToolExecutor
  ↓
Observation
  ↓
RuntimeState
  ↓
TaskEvaluator
  ↓
DecisionTrace + Transition + episode evaluation
  ↓
canonical Run
```

ACT v0はモデル能力の改善実験ではない。保存済みモデルが実際の環境へActionを出し、その結果がcanonical evidenceへ戻る実行経路を成立させる。

## 既存実装を正本として再利用する

- model load: `src/aster/model/decision_artifact.py::load_decision_artifact`
- candidate builder: `CalculateAndStoreCandidates`
- serializer: `aster-decision-input-0` / `serialize_decision_input`
- inference: `ModelPolicy` / `score_candidates`
- runtime: `run_loop`
- tools: calculator / memory.put / memory.get
- evaluator: `TaskEvaluator`
- semantic execution evidence: `Transition`
- model-side evidence: `DecisionTrace`

ACT v0のためにserializer、candidate semantics、Tool semantics、Evaluator semanticsを変更しない。

## 固定development task

ACT v0の最初のService recipeは自由task入力を受けない。次を固定する。

```json
{
  "task_id": "act-calculate-store-dev-v0",
  "kind": "calculate_and_store",
  "operation": "add",
  "left": 23,
  "right": 19,
  "store_as": "answer"
}
```

これは2026-10-02に実装前固定したfresh development probeである。独立task-family holdoutやsealed testとは呼ばない。成功後に別の値へ選び直してCapabilityを主張しない。

## Artifact契約

Service/Webはfilesystem pathを受け取らない。DecisionModelは論理ID:

```text
decision_model:<sha256>
```

で参照する。

Service用catalogは少なくとも以下を検証する。

- artifact directoryがAster root配下の所定位置
- symlinkではない
- manifest / model.pt / tokenizer必須fileが存在
- `load_decision_artifact` によるartifact ID、model digest、Tokenizer digestの検証
- `candidate_builder_id == calculate-and-store-v0`
- artifact manifestのserializer互換性はACT v0で既存 `aster-decision-input-0` を暗黙変更しないことにより維持する

既存Run内artifactをService catalogへ登録するときだけローカルCLIがsource pathを受ける。登録後のService requestはlogical IDだけを受ける。

## Service recipe

初版recipe:

```text
agent-decision-model-v0
kind: agent
inputs:
  decision_model
parameters: none
policy: model_only
task: act-calculate-store-dev-v0
candidate builder: calculate-and-store-v0
tools: calculator, memory.put, memory.get
max_steps: 8
```

requestからtask本文、tool、candidate builder、serializer、threshold、model pathを指定させない。

## model-only

ACT v0ではRuleBasedPolicy、fallback、SelectivePolicyを実行Actionの選択へ介入させない。

```text
ModelPolicy.selected Action == executed Transition.action
```

を各model decision stepで要求する。

RuleBasedPolicyをshadow evaluator/teacherとして後続で使う場合も、model Actionを書き換えない。ACT v0初版ではshadow teacher自体を必須にしない。

## canonical evidence

Runは少なくとも次を保存する。

```text
trajectory.jsonl
decision-traces.jsonl
evaluation.json
agent-bundle.json
```

`DecisionTrace` は候補、raw score、raw softmax probability、selected indexを保持する。
`Transition` は実際に実行されたAction、Observation、state before/after、step evaluationを保持する。

agent bundleには少なくとも次を追加する。

- policy_mode = model_only
- model artifact logical ID
- model ID
- candidate builder ID
- DecisionTrace列
- task
- final memory
- episode evaluation
- wiring gate result
- capability gate result

Webはこれらを再計算しない。

## 不変条件

1. artifact reload検証済み。
2. ACT中にmodel weightを更新しない。
3. 実行前後のweightsが一致する。
4. candidate builderはartifact lineageと一致する。
5. serializerは既存 `aster-decision-input-0`。
6. model-onlyの各stepで `DecisionTrace.selected == Transition.action`。
7. model decision数と対応Transitionのstep mappingを明示する。
8. fallback/rule intervention = 0。
9. sealed testを読み込まない。
10. arbitrary command/path/tool/taskをService requestから受けない。

## Gate

### ACT-Wiring Gate

次が一周し、canonical evidenceとして保存・再読込できればPASS。

```text
artifact load
→ candidate build
→ model score
→ Action select
→ real tool execution
→ Observation
→ State
→ Evaluator
→ persisted DecisionTrace/Trajectory
```

モデルがtaskを失敗してもWiring GateはPASSになり得る。
Job processの成功とtask capabilityを同一視しない。

### ACT-Capability Gate

事前固定task `act-calculate-store-dev-v0` をmodel-onlyで完了し、

- task_success = true
- goal_verified = true
- model-only
- fallback count = 0
- rule intervention count = 0

ならPASS。

Wiring PASS / Capability FAILは正当な研究結果であり、失敗RunをSEE/LEARNへ渡す。

## 非目標

- model改善
- 新Decision学習
- compact/event serializer
- 自由自然言語task
- arbitrary tool
- fallback / abstain routing
- threshold調整
- sealed test
- Intervention Learning
- candidate model promotion
- remote Service

## 次段階

ACT v0後:

1. aster-webでstepごとのcandidate/score/selected/executed/Observationを表示。
2. ACT v0.1で必要ならSelectivePolicyを別モードとして追加し、model/fallback/abstainを分離。
3. Issue #20でstudent Action / executed Action / teacher correctionを記録し、candidate v1を親と同じEvaluatorで比較する。
