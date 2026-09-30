# Sites Workbench Contract v0

2026-09-27。ユーザー合意に基づく継続設計・作業記録の契約。
APIの正本は `docs/ServiceContract-v0.md`。この文書はSitesの責務、画面、拡張順と変更時の義務を定める。APIと矛盾する場合はAPI契約を優先し、差異を記録する。

## 目的

Asterを操作・観測・育成する個人用Web作業場にする。ユーザーが実験を開始し、何が起きたかを調べ、仮説と次の実験を残せることを成果とする。
完成像は素材の取込・採用から、前処理、Tokenizer、Pretrain、推論・対話、Agent実行、評価、再学習候補までを辿れる環境。
全段階を実装済みに見せない。添付「Aster 計画.pdf」のTinyLM → Training → Inference → Message/Tool protocol → Agent Loop → Trace/Evaluatorという段階を尊重する。

## 所有と責務

| 対象 | 正本・担当 |
| --- | --- |
| 原本、採用理由、変換、分割、学習実体 | Aster側。由来と版を保持 |
| 実行可能な能力、レシピ、入力検査 | Aster Service |
| 実験の要求 | Job。job_idで受付・状態・入力を追跡 |
| 実際に行われた処理 | Run。run_idとcanonical run/eventsが証拠 |
| 成果物 | Artifact。論理IDで参照 |
| 操作・表示・比較・考察 | Sites |
| 明示的に共有したCorpus・学習観測 | 既存Sitesの認証済みD1/R2保存 |
| API仕様 | Aster docs/ServiceContract-v0.md |
| UIソース | 既存Sites独立Git。Asterや私的素材を丸ごとSiteへ送らない |
| 製品設計と更新履歴 | 本文書とSitesWorkbench-changelog.md |

UIは実測lossや生成文を作り直さない。ユーザーの考察は実測記録と別に保存する。
Sites保存とAsterのcanonical evidenceは別物。ローカル参照はartifact IDへ解決し、filesystem pathを操作画面で要求しない。

## 初回の縦切り: Pretrain

1. 同じPCのブラウザで、固定localhostのServiceへ接続。Bearerキーはページメモリのみ。
2. GET /recipes と GET /artifacts を読み、版と必須項目を検査。GET /statusでJob一覧を取得。
3. tinylm-overfit-v0 / kind=pretrainだけを開始可能にする。
4. training_viewとtokenizerを選択。provenance_valid=trueかつtraining_view_digest一致のTokenizerだけを候補にする。
5. POST /jobsへaster-job-spec-0を送る。inputsは論理ID、parametersは空オブジェクト。shell、Python、module、path、自由な学習パラメータ欄は作らない。
6. HTTP 202のjob_idでGET /jobs/{job_id}を約1.5秒おきに取得。前の通信の完了後に次を予約。
7. JobとRunのID・状態を別表示。run=nullは開始待ち。JobからRunの状態を推定して書き換えない。
8. eventsをseq順で表示。画面は最新300件に制限するが公開用JSON書出しには取得済み全イベントを保持。
9. outputs.trainingからloss曲線、生成前後、次token候補、学習token数とcheckpoint識別情報を表示。
10. terminal状態で定期取得を止めcatalogを更新。完成したTokenizerが次のPretrainで選べる。
11. 学習結果のSite保存は従来通り明示操作か、その実行で選んだ随時保存。対応CorpusがSiteに保存されている必要がある。Job切替時は随時保存を解除する。

## 状態と異常系

- accepted/runningは未完了。completed/failed/interruptedはJobの終端。
- 通信断は監視状態でありJob失敗ではない。最後の確定情報を保持し再接続する。
- 同時に1 compute job。二重クリックを止め409を表示。キューやキャンセルを独自に追加しない。
- POSTの応答が不明なら自動再送しない。再接続してJob一覧を確認する。
- 古いpollの応答が新しい選択・保存再生を上書きしないよう世代番号で除外する。
- 再接続時はactive Jobを優先し、なければ選択していたJobを再取得。
- stepの総数がない場合は割合を捏造しない。Job終了率と学習step進捗は同一ではない。
- Serviceの任意diagnostic/error、process command、code pathを画面へそのまま出さない。UI公開用コピーはerror等を除く。詳細診断と元のcode hash対応はAsterの原記録で保持。
- overfitのloss低下は暗記の確認。一般化・Agent能力の証拠として扱わない。

## 画面

既存の濃紺・白・青のUIと左ナビを継承する。
「素材をたどる」「ことばを分ける」「学習を観察する」「共有Corpus」「実行と記録」を維持。
学習ページにレシピ・本文・Tokenizerの選択とPretrain開始。実行ページにJob/Run/events/outputs。学習曲線・生成比較は既存実測コンポーネントへ接続。
原本・canonical・学習本文・採用理由から、入力Artifact、Job、Run、出力への参照が将来の中心になる。参照できない情報は未対応と表示する。

## 既存機能の移行

- 旧POST /trainingを廃止しPOST /jobsへ統一。旧BPE要求もtyped JobSpecへ移行。
- 旧run_idという名のJob IDをcanonical Run IDへ流用しない。古いbundleは「旧形式のJob ID」と明示。
- Site同梱の独立aster_bridge.py配布を削除。実行ServiceはAsterリポジトリから導入。
- Service v0にないGET /shareは呼ばない。既存共有Corpusの閲覧、共有ファイル取込、保存、比較は維持。
- Job/eventsはServiceで永続化される。現段階のSite側永続化は従来のtraining bundleであり、完全なJob bundleは手動書出し。Site側でJob履歴を独立保存する機能は未実装。

## 接続範囲と確認の限界

現行Serviceは127.0.0.1限定。SitesサーバーからproxyしてもユーザーPCのlocalhostへは到達しないため、ブラウザ→Serviceを採用。
スマホ→PC、常時遠隔接続は未実装。実機ではWindows/WSLの到達性、ブラウザのローカルネットワーク許可・CORSを確認する。
HTTPとDOMの検証を、ブラウザ描画・実ユーザーPC接続の検証と混同しない。

## 拡張順

| 段階 | 追加する能力 | 追加前に決める契約 |
| --- | --- | --- |
| 1 | Pretrainの操作と観測 | ServiceContract-v0。今回の範囲 |
| 2 | raw staging/upload、採用・除外・変換 | 出典、作者、利用条件、revision/hash、分割、送信範囲、重複、削除 |
| 3 | checkpointのArtifact化・推論・対話 | モデル版、生成条件、入力/出力、保存範囲、訂正と学習候補の分離 |
| 4 | Agent・道具・Trace・評価 | 提案/実行結果の区別、権限、失敗・修正、評価基準 |
| 5 | 再学習・比較・モデル採用 | 候補選別、評価用隔離、親Run、採用理由、ロールバック |

アップロードしたことを学習採用と同一視しない。使用した対話を自動学習しない。実行完了をモデル採用と同一視しない。
将来の音声・画像も出典、対応テキスト、領域・区間、閲覧範囲をArtifactの由来として記録する。
遠隔接続は認証・到達性・公開範囲を別途設計してから追加し、v0のlocalhost制約を黙って緩めない。

## 継続する変更契約

2026-09-30研究接続メモ: Decision fit/状態診断はAsterのoffline CLIとcanonical Runで実測する。
現段階でService/Webへの公開は追加していない。将来の表示ではtrain fit、dev診断、model-only episodeを分離し、モデル/Tokenizer/suite/元Run/候補由来を同じbundleから辿る。
UIでdiagnostic successを再計算せず、未提供の値を推測しない。具体的な公開schemaとallowlisted recipeは対応Service PRで定義する。

今後Aster/Sitesに関連する作業をするときは、同じ作業内で次を行う。

1. ServiceContract、本書、最新changelog、関係するAGENTS.mdを読む。
2. UIの責務・画面・データ関係・API・保存/公開範囲・拡張順が変わる場合、本書を更新する。
3. 毎回changelogへ日付、要求、変更、理由、関連commit/PR、検証内容と結果、未検証、公開版、次の課題を追記する。
4. API変更はServiceContractを先に整合させる。UIだけで存在しないAPIや自由実行経路を作らない。
5. 同じ変更のソースと記録をGitへ保存する。公開失敗・未マージ・依存PR未導入も明記する。
6. 最終報告でSite URL、文書/PR、今回できる操作と必要な次の準備を示す。
7. READMEや旧計画の現状説明と矛盾を残さない。過去の観測結果は書き換えず、追記で訂正する。

本書は将来案を勝手に実装する包括許可ではない。各作業の範囲はその時のユーザー指示で決め、未実装の部分は区別する。

2026-09-30データ介入研究メモ: 将来の比較表示はarm名だけでなく、提示回数/更新回数、unique判断数、元課題group数、同じTokenizer/初期weightの識別、train fit成立、元課題保持、dev probeの由来を表示する。
反復slotの増加を新課題の増加と同一視しない。dev probeの親trainとの重なり、test scored_cases=0、reserved procedural prefixと独立task familyの違いをbundle正本から表示する。この研究CLIはService公開recipeではなく、UI接続は未実装。
