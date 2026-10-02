# Learned Agent ACT v0 — user WSL2 first model-only run

2026-10-03。Issue #48。これは形式例ではなく、ユーザーWSL2上のAster Serviceで実際に実行されたmodel-only ACT Runの記録。

## Provenance

- fit run: `10d679f7575b40c7ae1f78ecb2abe10e`
- arm: `one-fixed`
- source checkpoint: `decision_fit:fceed28d3a3c9c4632575c0bd4f422b2e13842e6d9510df6018714d0435b19a8`
- promoted DecisionModel: `decision_model:e28fb30d64f9b9f00c71c232e5200648eed2d208922909e065dbd65d7c5d4bf0`
- calibration: `not_run`
- routing: `not_configured`
- ACT Run: `5a8076571dca446fa07190cf4fc62509`
- task: `act-calculate-store-dev-v0` = `23 + 19` を計算して `answer` に保存し、読み出して検証する
- scope: fresh development probe。sealed test / independent task-family holdoutではない。

正規化した実測値は [learned-agent-act-user-pc-v0.json](learned-agent-act-user-pc-v0.json)。

## Gate

### ACT-Wiring: PASS

実測:

- job status: completed
- artifact reload verified: true
- model-only: true
- fallback count: 0
- rule intervention count: 0
- weights unchanged: true
- `DecisionTrace.selected == Transition.action`: true
- candidate builder: `calculate-and-store-v0`
- serializer: `aster-decision-input-0`

したがって、保存済みDecisionModel → candidate生成 → score → Action選択 → real Tool → Observation → State → Evaluator → persisted evidence の実機経路は成立した。

### ACT-Capability: FAIL

実測:

- steps: 8
- accepted actions: 8
- successful executions: 8
- failed executions: 0
- goal verified: false
- task success: false
- terminal reached: false
- stop reason: null

これはJob/Tool失敗ではなく、model-only policyがtaskを完了できなかったepisode。

## Action sequence

| step | selected | environment result | memory after |
| ---: | --- | --- | --- |
| 0 | `calculator(23 + 19)` | `42` | `{}` |
| 1 | `memory.put(answer=42)` | stored | `{"answer":42}` |
| 2 | `memory.put(answer=42)` | stored | `{"answer":42}` |
| 3 | `calculator(23 + 19)` | `42` | `{"answer":42}` |
| 4 | `calculator(23 + 19)` | `42` | `{"answer":42}` |
| 5 | `memory.put(answer=42)` | stored | `{"answer":42}` |
| 6 | `memory.put(answer=42)` | stored | `{"answer":42}` |
| 7 | `memory.put(answer=42)` | stored | `{"answer":42}` |

step 0のcalculatorとstep 1のmemory.putまではtask上有効な進行で、step 1終了時点でRuntimeStateのmemoryは既に `answer=42`。

Evaluatorは `memory.get("answer")` の実観測で値を確認したときにgoalをverifiedとするため、保存だけではgoal_satisfiedにならない。

## Candidate coverageとranking

step 0以降、`memory.get("answer")` はcandidateとして存在している。step 1以降は
`calculator / memory.put / memory.get / stop` が全て候補に存在する。

したがってこのRunで観測された主失敗は「必要Actionがcandidateに無い」ことではない。
step 1で正しい値を保存した後も、modelは `memory.get` を一度も選ばず、再計算・再保存を繰り返した。

特にstep 2ではmemoryが既に `answer=42` なのに:

```text
memory.put  score -1.047859  p=0.253250  ← selected
calculator  score -1.061407  p=0.249842
memory.get  score -1.064849  p=0.248983
stop        score -1.069108  p=0.247925
```

となり、verification Action `memory.get` は候補にあるがrankingで負けている。

## Confidence / discrimination

step 0は3候補でselected probability 0.33378、step 1以降は4候補でselected probabilityがおよそ0.2503–0.2553。
どのstepでもほぼuniform baseline（3候補なら約1/3、4候補なら1/4）に近い。

従って、この `one-fixed` checkpointではAction間のscore差は弱く、correctだったstep 0/1も強い確信を伴う選択とは読まない。

## 読み方

### 実測から直接言えること

1. 算術Toolは正しく42を返した。
2. memory.putは正しく `answer=42` を保存した。
3. Tool execution自体は8/8成功。
4. 必要なverification candidate `memory.get` は存在した。
5. modelは `memory.get` を一度も選ばなかった。
6. そのためEvaluatorのgoal verificationへ到達しなかった。
7. modelはstopにも到達せずmax stepsまで実行した。

### 現時点の解釈

このepisodeは **calculator failure / memory failure / candidate coverage failure ではなく、保存済みstateからverificationへ遷移するaction ranking / state-conditioned policy failure** と分類するのが妥当。

ただし1 Runだけから、その根本原因を「state reader」「history reader」「position」「optimization」「training coverage」のどれかへ確定しない。
`one-fixed` は極端に限定したfit armなので、広いAgent能力の失敗/成功ともみなさない。

## 次の比較

実装を変える前に、同じ固定ACT task・同じRuntime・同じEvaluatorで:

1. `one-fixed`（このRun）
2. `eight-fixed`
3. `eight-shuffle`

をmodel-onlyで比較する。

見る量:

- ACT-Wiring / ACT-Capability
- first wrong transition
- `memory.get` のrank / score / probability
- action sequence
- goal verification到達
- terminal到達

これで、学習例数/提示順の違いが実環境policyへどう現れるかを、Runtime変更なしで比較できる。
