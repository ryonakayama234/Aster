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


## Session 2 — 2026-10-02

Task: ACT v0のAster側vertical sliceを実装する。

Status: implementation complete, CI/user run pending

### 実装

- Service Artifact kindへ `decision_model` を追加。
- `ArtifactCatalog.register_decision_model` を追加し、既存DecisionModel artifactをload検証後に `artifacts/decision_models/<digest>/` へimmutable copyする。
- source artifact内symlink、tampered model state、identity/candidate-builder不一致を拒否。
- `scripts/register_decision_artifact.py` を追加。filesystem pathを扱うのはローカル登録時だけで、Service Jobはlogical IDだけを受ける。
- `src/aster/runtime/service_learned_agent.py` を追加。
- fixed task `act-calculate-store-dev-v0` = add(23, 19), store_as=answer を実装。
- `load_decision_artifact → ModelPolicy → run_loop → real ToolExecutor → TaskEvaluator` を接続。
- model weights unchangedを実行前後で監査。
- `DecisionTrace.selected == Transition.action` とstep/count alignmentをmodel-only invariantとして監査。
- `decision-traces.jsonl` とagent bundle内decision evidenceを保存。
- ACT-Wiring GateとACT-Capability Gateを別fieldで保存。
- `agent-decision-model-v0` recipeを追加し、入力を `decision_model:<digest>` だけに固定。
- ServiceContract / AGENTS / changelog / development status / READMEを同期。

### テスト追加

- DecisionModel catalog registration / idempotence / tamper / symlink拒否。
- ACT runnerのDecisionTrace/Transition alignment、weights unchanged、Gate保存。
- learned Agent Service recipeがlogical IDだけを受け、内部でregistered pathへ解決すること。

### 未確認

- GitHub Actions pytest / Pyright。
- ユーザーWSL2の既存DecisionModel artifact登録。
- 実ACT RunのWiring/Capability結果。
- aster-webのACT表示。

### 次

1. stacked Draft PRを作成してCI。
2. CI failureを修正。
3. ユーザーWSL2でartifact登録 → Service ACT Run。
4. 実測Runを基準にaster-webを接続。


### PR / CI

- Draft PR #49を作成。
- Research Observatory PR #47を実機確認・既存CI成功後にmainへmerge。
- PR #49をmainへretarget。main向けActionsを起動するため、この進捗更新をsynchronize commitとして追加。


## Session 3 — 2026-10-02

Task: ACT v0 CI gate。

Status: CI passed; user WSL2 run pending

- PR #49 GitHub Actions run #121。
- Pyright: success。
- pytest: success。
- これでコード/型/自動テストGateは通過。
- Capability成功はまだ主張しない。次はユーザーWSL2の既存DecisionModel Artifactをcatalogへ登録し、Service JobとしてACT Runを実測する。


## Session 4 — 2026-10-03

Task: 実機確認の記録漏れを修正し、ACTを閉じる。

Status: complete

- ユーザー報告により、WSL2実機でACT v0の実行経路は確認済み。
- PR #49はmainへmerge済み。
- Issue #48へ実機確認済みを追記し、completedとしてclose。
- ACT-Wiring / ACT-Capabilityは実機確認済みという理解で次Gateへ進む。
- 実Run ID / bundle ID等の詳細はrepo正本に未記録。値を推測して補わない。
- 次Gateは Issue #20 **LEARN v0 — Correction Transfer**。
