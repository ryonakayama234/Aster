# Selective Agent Runtime v0

PR #8でAsterDecision-v0のgeneralizationとcalibrationを測れるようになりました。この段階では確率は観測値であり、Agentの実行制御にはまだ使っていませんでした。

Selective Agent Runtime v0は、calibrated confidenceを最初のcontrol signalとしてruntimeへ接続します。

```text
DecisionModel
    ↓ scores
Temperature scaling
    ↓ calibrated probabilities
Confidence gate
    ├─ high   -> model Action
    ├─ medium -> fallback Policy
    └─ low    -> abstain / stop
```

## 境界

既存のAgent Kernel契約は変更しません。

- `Policy.decide(...) -> Action` はそのまま。
- `run_loop` はmodel confidenceを知りません。
- `Transition` は実際に選ばれ、実行されたActionとObservationを記録するsemantic execution truthのままです。
- model固有のscores/probabilitiesは`DecisionTrace`、routing判断は`RoutingTrace`へ分離します。

したがって、将来fallbackを`RuleBasedPolicy`からslow reasonerや別Agentへ交換しても、tool/runtime/Transition schemaを変更する必要はありません。

## routing rule

`SelectivePolicyConfig`は3値を持ちます。

```text
temperature > 0
0 <= fallback_threshold <= autonomous_threshold <= 1
```

モデルscoreをtemperature `T` で割ってからsoftmaxし、最大確率をconfidenceとします。

```text
confidence >= autonomous_threshold
    -> route = model

fallback_threshold <= confidence < autonomous_threshold
    and fallback exists
    -> route = fallback

otherwise
    -> route = abstain
       Action.stop("low_confidence")
```

Temperature scalingはcandidateのargmaxを変えません。ただし、confidence thresholdを使うruntimeではrouteを変え得ます。これはPR #8で「calibration improvement」と「generalization improvement」を分けた理由の一つです。

## RoutingTrace

各stepで、semantic `Transition`とは別に次を保存します。

- model ID
- candidate Actions
- raw scores
- raw probabilities
- calibrated probabilities
- selected candidate
- calibrated confidence
- thresholds / temperature
- route (`model` / `fallback` / `abstain`)
- modelが選んだAction
- runtimeへ返したfinal Action
- fallback Policy ID

この分離により、Sitesは「何が実行されたか」と「なぜmodelに任せた／任せなかったか」を混同せず表示できます。

## Sites-readable run artifact

`run_logged_selective_agent(...)`は既存`RunLog`を使い、次を保存します。

```text
runs/<run-id>/
├── run.json
├── events.jsonl
├── selective.json
├── trajectory.jsonl
└── routing.jsonl
```

`selective.json`には、能力benchmarkではなく運用上のrouting summaryとして次を保存します。

- steps
- task success
- model / fallback / abstain count
- autonomous coverage
- fallback rate
- abstain rate
- mean calibrated confidence
- model routeでのtool execution failure count

`autonomous_coverage`が高いこと自体を「賢い」とは扱いません。低いthresholdを置けばcoverageだけは簡単に増やせるためです。PR #8のheld-out accuracy / NLL / Brier / ECE / risk-coverageと組み合わせて解釈します。

## v0で意図的に含めないもの

- reward / RL
- trajectoryからの自動再学習
- fallback結果を教師labelとしてdatasetへ追加する処理
- slow reasoner
- dynamic threshold tuning
- Sites UI変更
- shell / filesystem / browser tool追加

次の段階では、modelが実際に遭遇したstateでfallbackが選んだActionをintervention dataとして保存し、`model v0 -> rollout -> correction -> training -> model v+1`を閉じる候補があります。
