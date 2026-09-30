# 開発の現在地

2026-09-28。Issue #19 / PR #22時点。
基準mainは `76516c6ca7e7d83b868c2e68a6232327e80b6ab1`。固定Agent Service v1（PR #18）はmainへmerge済みです。

この文書では、**実装済み、CIで検証したこと、ユーザーPCで実測したこと、まだ設計だけのもの**を分けます。

## 実装の地図

| 場所 | 現在の役割 |
| --- | --- |
| `src/aster/corpus/` | raw/canonical候補、由来、学習入力前のデータ処理 |
| `src/aster/training/view.py` | pilotの採用・分割・配合・対応表・観測bundle |
| `src/aster/tokenizer/` | byte/BPE、保存・読込 |
| `src/aster/model/tiny_lm.py` | shared TinyLM backbone |
| `src/aster/model/decision_head.py` | candidateごとのscalar scoreを出すDecisionModel |
| `src/aster/model/decision_artifact.py` | Issue #19。DecisionModel + Tokenizer + config + lineage + calibration/routingの保存・検証読込 |
| `src/aster/training/decision.py` | RuleBasedPolicy由来DecisionExampleのsupervised imitation |
| `src/aster/benchmark/` | train/calibration/dev/test、generalization slices、NLL/Brier/ECE/risk-coverage |
| `src/aster/training/decision_baseline.py` | Issue #19。train→calibration→dev、test封印、Artifact化、episode比較、CPU観測、final-test |
| `src/aster/agent/`, `runtime/`, `tools/`, `evaluator/` | Policy→Tool→Observation→State→Evaluator、Selective routing、trajectory/evaluation |
| `src/aster/service/` | localhost限定のallowlisted Job/Run/Artifact Service。固定Agent recipeまで公開 |
| `docs/UI/` | Asterを操作・観測・育成するSites/aster-webの責務・履歴 |

## 成立している縦切り

### Corpus / Tokenizer / TinyLM

```text
raw/canonical候補
→ training view
→ BPE
→ TinyLM pretrain
→ checkpoint reload
```

最初のpilotやoverfitは接続・暗記の確認であり、広い一般化性能の証拠とは扱いません。

### 固定Agent Runtime

```text
aster-web
→ Aster Service
→ agent-calculate-store-v0
→ RuleBasedPolicy
→ calculator / memory.put / memory.get
→ trajectory
→ Evaluator
```

Service v1はmainへmerge済みです。任意task、任意tool、学習済みDecisionModelのService公開はまだ行っていません。

### Decision CPU Baseline（PR #22）

```text
train
  ├→ BPE training
  └→ DecisionModel update
calibration
  └→ temperature fit
dev
  └→ Decision / episode / routing観測
→ DecisionModel Artifact
→ reload
→ testは別final-testまでsealed
```

ArtifactはweightsだけでなくTokenizer、model config、candidate builder、suite digest、training config、calibration/routing設定を関連付けます。optimizer stateは保存しないためtraining resumeは未対応です。

## Issue #19の完了に必要な実測

PR #22のコードとテストが成立した後、ユーザーPCのWSL2/CPUで次を記録します。

- 実行Git SHA
- development Run ID
- DecisionModel artifact ID
- `torch_num_threads=2` と `4` の小規模比較
- train step平均wall time
- tokenizer / train / calibration / dev decision / dev episodeの時間
- process max RSS
- devの例数、accuracy/NLL/Brier/ECE、失敗case
- rule / model / model+fallbackのepisode successとroute率
- 条件固定後のfinal-test Run IDとtest範囲

CIのpytest/Pyright成功やGitHub runnerの速度を、ユーザーPCの能力実測としては扱いません。

## 次に進める順序

1. **現在**: PR #22のCIを通し、Decision baseline implementationを固定する。
2. **実機Gate**: WSL2/CPUでdevelopment baselineを実測。必要ならthread 2/4だけ比較し、予算を決める。
3. **Final test Gate**: 設計を固定したArtifactに対してtestを一度開封する。
4. **aster-web #2**: DecisionModel artifact / fixed suiteをServiceのallowlisted recipeへ追加し、rule / model / model+fallbackのRun比較をWebへ接続する。
5. **Aster #20**: 訂正→candidate再学習→親子比較。自動promotionはしない。
6. **Aster #21**: 言語TinyLM pilotはDecision Agent系と変数を混ぜず別レーンで進める。
7. raw staging/uploadは素材準備が実際の障害になった時点で由来管理契約に沿って拡張する。

## ユーザーとエージェントの分担

- エージェント：抽出・検査・記録・実装・比較条件・再現性を構築し、未検証部分を明記する。
- ユーザー：Asterへ採用したい素材、実際の使い方、比較結果を見た採用判断を行う。
- モデル候補は自動昇格しない。改善・不変・悪化を同じ物差しで見てから採用する。

Decision baselineの仕様は [DecisionBaseline-v0](DecisionBaseline-v0.md)、Agent Service境界は [ServiceContract-v1](ServiceContract-v1.md) を参照してください。
