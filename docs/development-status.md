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


## 2026-10-02追記: Research Observatory Gate

Decision研究の次の局所実験を増やす前に、Aster #29 / aster-web #2の保存再生観測系を閉じるbranch
`feat/research-observatory-v0` をmainから開始した。

このbranchでは保存済みfailure-audit reportだけをversioned read-only bundleとしてServiceから公開し、
aster-webで同じcaseを2 arm並べて確認する。candidate scoreやlearning curveがcommit済み証拠に無い場合は未取得のまま表示する。

この作業はDecision能力改善、学習済みpolicyのService実行、Intervention Learningを意味しない。
次のGateは、観測系の実ブラウザ確認後に learned DecisionModel → typed Runtime state → real Tool → Evaluator を通すend-to-end run。


## 2026-10-02追記: SEE実機成立 / ACT v0実装中

Research ObservatoryはユーザーWSL2実ブラウザでarm/case比較まで成立した。
Aster PR #47は実機確認を記録してReady for review、aster-web PR #4はmerge済み。

次のIssue #48では新しいDecision学習ではなく、既存の保存済みDecisionModelを実行系へ接続する。

```text
decision_model:<digest>
→ load_decision_artifact
→ ModelPolicy
→ CalculateAndStoreCandidates
→ real ToolExecutor
→ Observation / RuntimeState
→ TaskEvaluator
→ DecisionTrace + Trajectory
```

branch `feat/learned-agent-act-v0` では、現時点でSpec/TODO/Progress、DecisionModel catalog登録、
model-only ACT runner、DecisionTrace永続化、Service recipeを実装中。
GitHub CIとユーザーPC実測前なので、実装済みと能力成立を区別する。

ACT-WiringとACT-Capabilityは別Gate。task失敗でもartifact→model→tool→evidenceの配線が正しければWiringはPASSになり得る。
