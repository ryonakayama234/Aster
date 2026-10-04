# Asterで一緒に開発するエージェントへ

## 目的

Asterは、自作TinyLMを通してAI・Python・数学・コンピューターサイエンスを学ぶプロジェクト。
ユーザーが仕組みを理解し、自分で変えられることを成果に含める。
将来の中心は外部化知能：計算、記録、検索、実行を外部の道具へ委ね、結果を使って行動できること。

## 協働の進め方

- 日本語で説明する。概念の直感、小さな具体例、実装、観測結果の順に結びつける。
- 数式や専門語は必要なときに使い、記号の意味を説明する。
- 技術的な選択肢は推奨案と理由を示す。ユーザーに未習得のアルゴリズム選択を丸投げしない。
- 変更は理解できる単位に区切る。学習の節目で「今回わかったこと」「次に確かめること」を短く残す。
- 学習目的を理由に、頼まれた作業を止めたり、必須の小テストを課したりしない。
- 事実、設計案、未実装、実験による確認を区別する。テスト成功を知能や研究成功の証拠にしない。

## GitHub・外部計算資源のコスト

- 明示的な依頼なしに、従量課金が発生し得るGitHub機能や外部計算資源を有効化しない。
- `.github/workflows/**` を変更する場合は、原則としてGitHub標準runnerを使う。
- larger runner・GPU runner・有料runnerを、明示的な確認なしに使用しない。
- Codespaces、GitHub Models、cloud agentなどの有料・従量課金機能を、明示的な確認なしに新規利用しない。
- 不要なmatrix build、scheduled workflow、大容量artifact保存、過剰なcacheを追加しない。
- CI変更時は実行頻度、runner、artifact/cache、外部service利用を確認し、コスト増加の可能性があれば事前に明記する。
- 無料枠内であっても、同じ検証を重複実行しないようworkflowのtriggerとconcurrencyを設計する。

## コーパスの所有と由来

- 日本語prose・dialogueは、ユーザーの自作文と、ユーザーが選んだ外部文章・AI対話を候補に含める（2026-09-20更新）。明示的な依頼なしに代筆・外部収集・合成して埋めない。
- ユーザーが選んだことと、ユーザーが書いたことを区別する。出典・作者・AI生成元・確認状態を記録し、読む素材と応答のお手本を分ける。
- Math・Pythonは出典と利用条件を確認した外部資料から始める。
- structured・JSON・toolはAsterプロジェクト内で作る。外部tool会話コーパスで置き換えない。
- 初期のプログラム生成と、学習後のAsterモデルによる生成を区別して記録する。
- モデルが書いたtool結果を実行済みの観測として扱わない。正解は検証器・実行器で確認する。
- 原本、出典、ライセンス、revision、ハッシュ、変換、分割、除外理由を追跡する。
- 評価用データをBPE学習にも混ぜない。元問題・章・生成テンプレートの派生は同じ分割へ置く。

## 実装と実験

- 方針案は `docs/corpus/AsterCorpus-v0.md`。案の段階を勝手に凍結済みと扱わない。
- v0を固定してTokenizerとTinyLMで評価した後、v0.1へ変更する。改善案をv0へ静かに混ぜない。
- Corpusの版とTokenizerの版は別。既存AsterTokenizer-v0.1がCorpus-v0を使っても矛盾しない。
- まず既存の単純なbyte-level BPEを使う。速度や品質に問題が出たら計測して改善する。
- このWSLチェックアウトではLinux側のPython・Gitを使う。ユーザーの未コミット作業を保護する。
- 既存の読み込み対象拡張子・特殊トークン・文書境界を確認してから接続する。
- 外部コードを収集時に実行しない。将来の実行器はネットワークなし・隔離・時間/資源制限を設ける。
- 変更に見合う確認を行い、未検証部分を明記する。大きな依存や抽象化を先回りして入れない。

## Sites変更時の継続契約（2026-09-27合意）

- Asterを操作・観測・育成するSitesに関係する変更では、最初に `docs/ServiceContract-v0.md`（存在するブランチ/PR）、`docs/UI/SitesWorkbenchContract-v0.md`、`docs/UI/SitesWorkbench-changelog.md` を読む。
- 同じ作業内で設計契約を現状へ更新し、changelogへ変更理由・ソースcommit/PR・検証結果・未検証・公開状態・次の課題を追記してGitへ保存する。
- API契約とUI責務を区別し、Job/Run/Artifactの識別、出典・採用理由、実測と保存再生の区別を保つ。
- 将来案は未実装と明示する。前提PRが未マージなら導入済みと扱わない。設計書を完成記録だけ先に更新しない。


## Learned Agent ACT v0（Issue #48, 2026-10-02）

- ACT v0の正本は `docs/LearnedAgentAct-v0.md`、作業順は `docs/TODO/LearnedAgentAct-v0.md`、進捗は `docs/Progress/LearnedAgentAct-v0.md`。
- 初版は保存済みDecisionModelのmodel-only実行。RuleBasedPolicy/fallback/abstainで成功を補わない。
- Service requestはlogical `decision_model:<digest>` のみを受け、任意model path/task/tool/serializer/thresholdを受けない。
- serializerは既存 `aster-decision-input-0`、candidate builderは `calculate-and-store-v0` を維持し、ACTの配線変更と表現研究を混ぜない。
- DecisionTraceはモデル側の候補/score/選択、Transitionは実際のAction/Observation/Stateを正本とし、model-onlyではselected Actionとexecuted Actionの一致を監査する。
- ACT-Wiring PASSとACT-Capability PASSを分ける。モデルがtaskに失敗しても配線が正しければWiringはPASSになり得る。
- ACT中にweightsを更新せず、sealed testを開かない。


## LEARN v0 — Correction Transfer（Issue #20, 2026-10-03）

- 正本は `docs/LearnCorrectionTransfer-v0.md`、作業順は `docs/TODO/LearnCorrectionTransfer-v0.md`、進捗は `docs/Progress/LearnCorrectionTransfer-v0.md`。
- 同一parentから P0 Parent / R1 Replay / C1 Correction を分岐し、Correction固有効果と単なる追加update効果を分離する。
- R1/C1はparent、Tokenizer、serializer、candidate builder、DecisionTrainConfig、seed、optimizer step数、training example列長を可能な範囲で一致させる。equal stepをequal computeと呼ばず、token数・時間・RSSも記録する。
- corrected stateの改善だけをtransfer成功と扱わない。Repair / uncorrected sibling transfer / model-only sequential transferを分離する。
- seedや同一trajectory内decision stepを独立task数に数えない。task/generator familyを基本単位とし、confirmatory manifestを結果を見る前に固定する。
- 良いseedだけを後から追加しない。既存devを独立holdoutへ名称変更せず、sealed testを開かない。
- parentを上書きせずcandidate artifactとして保存し、自動promotionしない。
- LEARN v0中にRL、reward optimization、new Tokenizer、new serializer、model scalingを同時導入しない。
