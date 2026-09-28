# AsterDecision Benchmark v0

AsterDecision-v0は、RuleBasedPolicyのtrajectoryから作ったDecisionExampleを使って、候補Actionを1件ずつscoreする最小学習経路です。PR #6の暗記試験は「学習経路がつながっている」ことを確認するもので、未知条件へのgeneralizationを示すものではありません。

このbenchmarkは、その次の物差しを固定します。

## 何を分けて測るか

- **generalization**: 学習に使っていない数値範囲、演算、key、history、candidate構成、失敗条件でも正しいActionを選べるか。
- **calibration**: Decision Headのsoftmax確率が、実際の正答頻度と対応しているか。
- **selective execution**: confidence thresholdを上げたとき、coverageとaccepted accuracyがどう変わるか。

accuracyだけでなく、NLL、multiclass Brier score、ECEとbinごとの件数、risk/coverageを保存します。ECEはbinningに依存するため、単一値だけでなくreliability tableもartifactへ残します。

## split

`calculate-and-store-v0`は次の4 splitを明示します。

- `train`: Decision policyの教師あり学習に使ってよい例。
- `calibration`: temperature scalingの1 parameterだけをfitする例。モデル重みの更新には使わない。
- `dev`: 設計を比較するときの観測用。
- `test`: 最終比較用。temperature fitにも使わない。

各caseには`leakage_group`があり、同じgroupが複数splitへ入るsuiteは構築時に失敗します。現行suiteは小さなsynthetic taskだけであり、benchmark処理と分布shiftの配線確認が目的です。高いscoreが広いAgent能力を示すとは扱いません。

Issue #19以降のCPU baselineでは、このsplitの役割を実行経路でも固定します。Tokenizer trainingとweight updateは`train`だけ、temperature fitは`calibration`だけ、開発観測は`dev`だけに限定し、`test`は保存済みmodel artifactを評価する別entry pointまで推論しません。詳細は[DecisionBaseline-v0](DecisionBaseline-v0.md)を参照してください。

## v0のgeneralization slices

- `interpolation`: 既知のtask構造で値だけ変える。
- `range_shift`: 学習例より大きい数値。
- `composition`: 未学習の演算とkeyの組合せ。
- `key_shift`: 新しいmemory key。
- `history_shift`: task開始時から無関係なmemoryが存在する。
- `tool_failure`: divide by zeroのように実行が失敗し、その後stopすべきtrajectory。
- `candidate_set_shift`: 追加のdistractor candidateがある。
- `candidate_permutation`: candidate順だけを反転する。

## calibration

各caseの生logitを保存し、`calibration` splitだけで正のtemperature `T`を選びます。

```text
raw logits z
    ↓ divide by T
softmax(z / T)
    ↓
calibrated probabilities
```

Temperature scalingはargmaxを変えないため、accuracyを改善する手法ではありません。確率の読み方だけを調整します。Asterでは将来、例えば高confidenceならtoolを自動実行し、中程度ならslow reasonerへ送り、低ければ停止・確認するcontrol signalとして使うことを想定しています。

## Sitesへ渡すbenchmark artifact

`run_logged_decision_benchmark(...)`は既存RunLogを使い、producer=`evaluator`として次を保存します。

```text
runs/<run-id>/
├── run.json
├── events.jsonl
├── benchmark.json
├── predictions.jsonl
└── calibration.json
```

`benchmark.json`はtest全体、split別、slice別のraw/calibrated metricsを持ちます。`predictions.jsonl`はcaseごとのtarget、prediction、logits、確率を保持します。Sitesは集計値だけでなく、悪化した個別caseまで戻って観察できます。

この既存runnerはsuite全体を一度に評価するため、**開発中にtestを封印する用途では使いません**。Issue #19のdevelopment runは`training/decision_baseline.py`のsplit限定経路を使い、final-testだけが保存済みartifactに対してtestを開封します。

## DecisionModel artifactとの関係

Benchmark自身はmodel保存形式を所有しません。Issue #19では`model/decision_artifact.py`がweights、Tokenizer、model config、candidate builder、suite digest、training config、calibration/routing設定を一緒に保存します。

したがって、

```text
Benchmark = 物差し
Decision Artifact = 比較される保存済みモデル
Baseline Run = 学習・校正・dev観測をつなぐ実験手順
Final-test Run = 凍結済みArtifactをtestへ適用する評価
```

と責務を分けます。

## 現時点で含めないもの

- reward / RL
- trajectoryからの自動再学習
- curriculum / problem generator
- automatic model promotion
- optimizerを保存したtraining resume
- このsynthetic suiteを広い能力benchmarkとして固定すること

次の段階では、同じsuiteを親ModelとTrajectoryで更新した候補Modelに実行し、未知sliceの改善とcalibrationの変化を同じ物差しで比較します。Issue #19のartifactは、その比較で親と候補を独立保存するための基礎になります。
