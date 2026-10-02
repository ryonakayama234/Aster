# Progress: Learned Agent ACT v0

## Session 1 — 2026-10-02

Task: SEEからACTへGateを切り替え、仕様を固定する。

Status: in progress

### 完了

- ユーザーWSL2実機でResearch Observatoryのarm/case比較が表示できたことをPR #47へ記録。
- PR #47をDraftからReady for reviewへ変更。
- PR #47 head `0706464afd52c9a847f4631448b77f8b3f4ea577` から `feat/learned-agent-act-v0` を分岐。
- Issue #48 `ACT v0: saved DecisionModelを実Tool実行へ接続する` を作成。
- ACT-WiringとACT-Capabilityを別Gateとして定義。
- fixed development taskを実装前に `add 23 + 19 → answer` と固定。
- serializer/candidate/runtime/evaluatorは既存実装を再利用し、ACT中に表現研究を混ぜない方針を固定。
- model-onlyを初版とし、fallback/rule interventionを非目標に固定。

### 現在わかったこと

- `ModelPolicy` は既に RuntimeState + Trajectory + candidates をscoreし、DecisionTraceを生成できる。
- `run_loop` はPolicy抽象に依存しており、RuleBasedPolicy専用ではない。
- state diagnosticsではModelPolicy + real ToolExecutorのmodel-only episodeがofflineで既に使われている。
- Service固定Agent runnerだけがRuleBasedPolicyをハードコードしている。
- Service ArtifactCatalogはtraining_view/tokenizerのみで、DecisionModel Artifactはまだlogical IDで公開されていない。
- したがってACTの主な仕事は新しい推論器ではなく Artifact → Policy → Runtime → Evidence → Service の統合。

### 次

1. `decision_model` Artifact kind/catalog/registrationを実装。
2. model-only runtime runnerとDecisionTrace永続化。
3. Service recipe。
4. CI後、ユーザーWSL2実測。
