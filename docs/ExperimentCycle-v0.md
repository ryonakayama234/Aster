# Experiment Cycle v0 — Trial / Batch

## 目的

Asterの実験を、毎回いきなり大きく回すのではなく、**短い診断サイクルで仮説を削り、必要になったときだけ大規模Batchへ昇格する**。

速さのために証拠境界を曖昧にしない。Trialで見たcaseを、そのまま未見のconfirmatory evidenceとして扱わない。

## 2つの実験モード

### Trial — 小さく、速く、原因を探る

用途:
- wiring確認
- mechanism diagnosis
- 仮説候補の比較
- instrumentation確認
- 大きなBatchへ進む価値があるかの判断

既定budget:
- family: 最大6程度
- seed: 最大3程度
- training unit: おおむね18以下
- checkpoint evaluation: 許可
- sealed test: 開かない

既定値は上限規則ではない。超える場合は、なぜTrialのまま増やす必要があるかを記録する。

Trialでは結果を途中で見てよい。ただし、途中結果を見て条件を変えた場合は**同じTrial cycleの証拠を能力claimへ昇格させない**。変更後は新しいcycle IDとして記録する。

Trialの結論語:
- observed
- not observed
- suggests
- diagnostic lead
- mechanism not distinguished

`Supported / Not supported` のconfirmatory verdictには使わない。

### Batch — 大きく、凍結して、主張を判定する

用途:
- confirmatory hypothesis test
- capability claim
- promotion / rejection判断
- 広い比較

要件:
- manifestを測定前に固定
- family / seed / endpoint / aggregation / tie / missing / retry規則を固定
- code revisionを固定
- partial endpointを見て条件を変えない
- resume可能なunit保存
- resource audit
- resultをmachine-readableで保存
- Supported / Not supported / Inconclusiveを事前規則だけで決める

大きいBatchは、可能な限り **1 unit = 1 family × 1 seed** のように分割して実行する。これをbatch処理の基本単位とする。

## Trial → Batch の昇格規則

Trialの目的は「良い数字を探す」ことではなく、**次に何を反証すべきかを絞ること**。

Batchへ進む条件:
1. Trialで少なくとも1つの明確なmechanism signatureまたは分岐条件が見つかった。
2. そのsignatureを検証するprimary endpointを事前に書ける。
3. Trialで使ったcaseをconfirmatory evidenceから除外するか、別の独立holdoutを用意できる。
4. 失敗しても意味のあるverdictを書ける。
5. 大きなcomputeを使う理由を説明できる。

条件を満たさない場合は、別の小さいTrialを回す。

## Trialの既定テンプレート

- cycle_id
- question
- competing_hypotheses
- selected_cases と選定理由
- evidence_scope = development / diagnostic
- fixed inputs
- allowed adaptations
- metrics
- stop condition
- next decision
- run IDs / artifact lineage

結果を見てcaseを選んだ場合は `post_hoc_selected=true` と明記する。

## Batchの既定テンプレート

- protocol_id
- frozen manifest SHA-256
- source Git SHA
- unit definition
- planned units
- primary endpoint
- secondary endpoints
- aggregation rule
- statistical rule
- retry / missing handling
- resource audit
- sealed-test boundary
- final verdict

## Computeの考え方

Trialはtraining unitを減らし、必要ならcheckpointを増やして**1回の学習から情報量を増やす**。

例:
- 6 family × 3 seed = 18 training units
- 30 family × 3 seed = 90 training units

unit costが同程度なら、Trialの学習本体はBatchの約20%、約5倍軽い。
一方、step 0/10/25/50/100のcheckpoint評価を行えば、18 training unitから90 checkpoint observationsを得られる。

この比率は計算量の保証ではない。checkpoint評価、I/O、Artifact保存などの追加costは別に記録する。

## 禁止

- Trialの途中結果を見た後、その同じcaseを「未見confirmatory」と呼ぶ。
- Batch途中で良いseed/familyだけ追加・削除する。
- Batch途中でendpoint、LR、steps、horizon、serializer等を変えて同じprotocol IDを継続する。
- Trialの探索結果を、confirmatory testなしで一般能力claimへ拡張する。

## Asterでの基本サイクル

```text
Question
  ↓
Trial
  ↓
mechanism / failure mode を絞る
  ↓
必要なら別Trial
  ↓
Batch manifest freeze
  ↓
Batch execution
  ↓
verdict
  ↓
次のQuestion
```

目的は「大きな実験を減らす」ことではなく、**大きな実験を、答えが得られる段階でだけ使うこと**。
