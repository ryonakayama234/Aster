# Aster Service Contract v1

2026-09-28。`ServiceContract-v0` の安全境界を維持したまま、Aster Agent Runtime の最初の実行・観測経路を追加する。

v0のTokenizer/Pretrain契約は引き続き有効。v1は既存endpointと`aster-job-spec-0`を互換のまま使い、許可されたJob kind / recipe / outputを追加する。UIから任意command、Python module、filesystem pathを指定できるようにはしない。

## 目的

最初の縦切りは「自由なAgent」ではなく、既存Agent Kernelが実際にToolを実行し、状態を更新し、Evaluatorがtrajectoryから結果を判定する経路をWebから観測可能にすること。

```text
aster-web / Sites
  ↓ versioned JobSpec
Aster Service
  ↓ allowlisted recipe
Agent Kernel
  ↓
Policy → ToolExecutor → Observation → RuntimeState → Evaluator
  ↓
Run / Event / trajectory / evaluation / agent bundle
```

## Capability

v1で公開するcapability:

```text
tokenizer.train
model.pretrain
agent.run
research.observe
```

`agent.run` が任意tool実行を意味するわけではない。Serviceが明示的に登録したAgent recipeだけを開始できる。

## 最初のAgent recipe

```text
agent-calculate-store-v0
  kind: agent
  inputs: none
  parameters: none
  policy: rule-calculate-store-v0
  tools: calculator, memory.put, memory.get
  fixed task: 40 + 2 を計算し total に保存して確認する
```

実際の期待trajectoryは次の4 Action。

```text
calculator
memory.put
memory.get
stop
```

`calculator` が42を返しても、その時点ではmemoryは変化していない。`memory.put` が成功した後にだけ `total=42` がRuntimeStateへ現れる。`memory.get` による確認とEvaluatorの判定は、書込み提案やcalculator結果と区別する。

このrecipeの `RuleBasedPolicy` は配線・記録・評価境界を検証するための固定policyであり、学習済みDecisionModelの能力や一般化性能の証拠として扱わない。

## JobSpec

```json
{
  "schema_version": "aster-job-spec-0",
  "kind": "agent",
  "recipe_id": "agent-calculate-store-v0",
  "inputs": {},
  "parameters": {}
}
```

未知field、追加input、追加parameter、未知recipeを拒否する。task本文、tool名、module、command、policy class、threshold等をrequestから渡す欄はv1に作らない。

## Output

Agent Jobの`GET /jobs/{job_id}`は既存Job bundleへ実測Agent outputを追加する。

```json
{
  "schema_version": "aster-service-job-bundle-0",
  "job": {"kind": "agent", "job_id": "...", "run_id": "..."},
  "run": {"schema_version": "aster-run-0", "kind": "agent"},
  "events": [],
  "outputs": {
    "agent": {"schema_version": "aster-agent-bundle-0"}
  },
  "agent": {"schema_version": "aster-agent-bundle-0"}
}
```

`aster-agent-bundle-0` は少なくとも以下を持つ。

- recipe_id / policy_id / fixed task
- persisted `trajectory.jsonl` から読んだtrajectory
- persisted `evaluation.json` から読んだevaluation
- final_memory
- run-level summary

`evaluation.json`は既存Evaluator Contractどおりpersist済みtrajectoryから決定論的に導出し、trajectory SHA-256を保持する。Web側でtask successやgoal verificationを再計算しない。

## 保存済み研究観測

2026-10-02のResearch Observatory縦切りでは、計算Jobを開始せず、Gitへ保存済みの研究証拠をread-onlyで公開する。
capabilityは `research.observe`。初版の論理IDは `decision-failure-audit-v0` のみで、requestからreport pathや任意ファイル名を渡さない。

`GET /experiments` は `aster-experiment-index-0`、`GET /experiments/{experiment_id}` は
`aster-experiment-bundle-0` を返す。初版bundleは `reports/decision-failure-audit-v0.json` の公開projectionであり、
source report SHA-256、arm/seed/checkpoint/suite/Tokenizer、phase-zero caseの保存済み正誤・target Action・selected Action、
target candidate serializationとtoken分割を含む。

committed reportにない学習曲線とcandidate scoreは `null` と
`unavailable_from_committed_evidence` を返す。ServiceやWebで推測・再計算して埋めない。
tokenizationは全候補ではなくtarget candidate serializationの証拠であることを明示する。
testはsealedのまま公開し、repository-local path・private corpus・任意生ログをbundleへ出さない。

## Job / Run state

v0と同じ。

```text
accepted → running → completed
                   ↘ failed
accepted/running -- service restart --> interrupted
```

Job IDとcanonical Run IDを同一視しない。同時実行は引き続き1 compute jobで、queueingは追加しない。

## HTTP / local development

v0/v1の既存endpointは維持し、Research Observatory用のread-only endpointを追加する。

```text
GET  /status
GET  /capabilities
GET  /recipes
GET  /artifacts
GET  /experiments
GET  /experiments/{experiment_id}
POST /jobs
GET  /jobs/{job_id}
```

Serviceは引き続き `127.0.0.1` のみでlistenし、Host検査とBearer tokenを必須にする。

Origin allowlistには既存Sitesに加え、aster-webのローカル開発用として次を許可する。

```text
http://localhost:5173
http://127.0.0.1:5173
```

任意Origin、LAN bind、Cloudflare Tunnel、公開hostnameはv1に含めない。遠隔化するときは認証・到達範囲・TLS・公開Artifactを別途設計する。

## Security invariants

- localhost bindを維持する。
- Host / Origin / Bearer tokenを検査する。
- requestからshell command / Python module / pathを受けない。
- recipe、input、parameterをallowlistする。
- `agent-calculate-store-v0` にfilesystem、shell、Python、network toolを登録しない。
- Toolの提案、accepted、execution success、state change、goal verificationを別の事実として記録する。
- UIはcanonical trajectory/evaluationを作り直さない。
- Agent実行完了をモデル採用・学習成功・一般化性能と同一視しない。

## v1完了条件

- typed JobSpecで固定Agent recipeを開始できる。
- 任意input/parameterを拒否する。
- canonical Runがcalculator → memory.put → memory.get → stopを記録する。
- calculator直後とmemory.put直後のstate差分を記録できる。
- persisted trajectoryからevaluationを生成し、task_success / goal_verifiedを取得できる。
- Job bundleからagent bundleを読み、repository-local pathを公開しない。
- Job IDとRun IDを区別する。
- v0 Tokenizer / Pretrain jobを壊さない。
- pytestとPyrightを通す。

## 次の拡張

v1が実機ブラウザまで成立した後に、順に検討する。

1. aster-webでAgent Run / Run一覧 / Inspect / Replay。
2. fixed taskから検査済みstructured task入力へ拡張。
3. RuleBasedPolicyと学習済みDecisionModelを明示的に切り分けて選択・比較。
4. routing / confidence / fallback / abstainの観測。
5. Intervention Learningのcandidate生成と親Run比較。
6. Counterfactual Evaluation。
7. model promotionは評価・ロールバック契約を決めた後だけ追加。

自由task入力や任意tool実行を、Web UIを作る都合だけで先に追加しない。


## Learned Agent ACT v0

Issue #48の初版は、登録済みDecisionModel Artifactを固定development taskの実Tool実行へ接続する。

### DecisionModel Artifact

Service catalogへ新しいartifact kindを追加する。

```text
decision_model:<sha256>
```

登録はローカルCLIがsource pathを受け、`load_decision_artifact` でmodel/Tokenizer/artifact identityを検証した後、
`artifacts/decision_models/<digest>/` へimmutable copyする。Service/Web requestからfilesystem pathを受けない。

`GET /artifacts` の `decision_models` はlogical IDと公開可能なlineage metadataだけを返す。
source pathは返さない。

### recipe

```text
agent-decision-model-v0
kind: agent
inputs:
  decision_model
parameters: none
policy_mode: model_only
task: act-calculate-store-dev-v0
tools: calculator, memory.put, memory.get
candidate_builder: calculate-and-store-v0
serializer: aster-decision-input-0
```

固定task:

```json
{"task_id":"act-calculate-store-dev-v0","kind":"calculate_and_store","operation":"add","left":23,"right":19,"store_as":"answer"}
```

これは実装前に固定したfresh development probeであり、sealed testや独立task-family holdoutではない。

### evidence

通常の `trajectory.jsonl` / `evaluation.json` に加え、`decision-traces.jsonl` を保存する。
agent bundleの `decision` fieldはmodel artifact ID、model ID、candidate builder、serializer、DecisionTrace列、
weights unchanged、Wiring Gate、Capability Gateを保持する。

model-onlyでは各stepで次を要求する。

```text
DecisionTrace.selected == Transition.action
```

Webはscore、task success、Gateを再計算しない。

### Gate

- ACT-Wiring: artifact load → candidate → score → selected Action → real Tool → Observation → State → Evaluator → persisted evidence が一周すればPASS。task failureでもPASS可。
- ACT-Capability: 事前固定taskをmodel-onlyで `task_success=true` かつ `goal_verified=true` ならPASS。

初版ではfallback、rule intervention、自由task、任意tool、threshold変更、model update、sealed test開封を含めない。
