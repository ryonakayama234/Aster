# Sites Workbench 作業履歴

設計の正本: [SitesWorkbenchContract-v0.md](SitesWorkbenchContract-v0.md)。
実装した事実、検証、公開状態、次の課題を追記する。過去の実測値を最新値で上書きしない。

## 2026-09-27: ServiceContract-v0 / Pretrain接続

### 依頼と基準

- 既存SitesをAster Serviceに接続し、recipes/artifacts選択→Job開始→poll→Run/events/outputsを表示。
- 初回レシピはtinylm-overfit-v0。JobとRunは別概念。ローカルpathや任意shell/PythonをUIへ出さない。
- 今後も設計と作業をrepo内に記録する契約をユーザーが指定。
- API基準: PR #16、head `0fe179fca34b5713c0b9e1e06b6c683dd69e178d`。
- 調査時点でPR #16はDraft/open/未マージ。mainにServiceがあると仮定しない。
- 変更前Site source: `2cc49c32cd87f285e1b8b1b788635c8c13d021a3` (公開版4)。

### 実装

- versioned response検査をpublic/service-contract.jsへ集約。
- public/runs.jsでGET /recipes、GET /artifacts、GET /status、POST /jobs、GET /jobs/{job_id}を共通化。
- training_viewとprovenance-validな対応Tokenizerを論理IDで選択。pretrainのparametersは空。
- BPEもtyped JobSpecへ移行し、Job IDをRun IDとして使う旧挙動を修正。
- accepted/Run未作成、実行中、完了、失敗、中断を区別。Job中断とcanonical Run状態も別表示。
- seq順イベント、step/tokens、outputs.training、checkpoint名/hashを表示。
- 既存のloss/生成前後/次token候補/保存比較を再利用。
- in-flight POSTの連打と409後の新規開始を抑止。応答不明POSTを再送しない。
- 切断でJobをfailedにせず最終情報を保持。再接続・世代番号による古いpoll排除。
- 旧GET /share直接取得を無効化。共有ファイル取込と共有済みCorpus機能は保持。
- 旧BridgeのSite配布を削除。導入元はAsterへ一本化。
- 選択を変えた場合、前の学習曲線を新しいJobの結果に見せない。随時Site保存も新Jobへ引き継がない。
- エラー・プロセス情報の任意JSONをUIへそのまま出さず、公開コピーを作成。
- 独立Sites READMEにも設計契約の参照と更新義務を記載。

### 検証結果

- SiteのNodeテスト: 8/8成功。既存保存・認証・改ざん/分割漏洩拒否・学習描画と、新規状態遷移/不一致参照/二重送信/切断/再接続/旧poll除外を検証。
- PR #16のServiceテスト: 4/4成功。
- 合成の構造化テキストとコードのみを使い、実際のPR #16 Service、CPU PyTorch、固定200-stepレシピでUIスクリプト→HTTP→学習→DOM表示を実行。
- Job: `a6122e2583de4526972f894cd33bc74f`
- Run: `04f71d44b23c4f6db11d2c97dbcd143e`
- completed、208 events、5 observations、最終step=200、checkpoint reload_exact_match=true。
- train loss: 約5.63418→0.00211389。これは検証用少量データの暗記であり一般化性能の証拠ではない。
- ユーザーの私的コーパスは変更・収集・学習していない。検証fixtureを実コーパスへ追加しない。
- build、JavaScript構文、git diff --check成功。
- この検証はNode DOM harnessとHTTPによるもの。実ブラウザ描画、ユーザーPCのWSL到達性、ブラウザのローカル接続許可/CORSは未検証。
- 既存Worker ESMには管理プレビュー互換dev serverがなく、実ブラウザQAは実施していない。
- Aster全体のpytest/Pyrightを通したという主張はしない。実施範囲は上記。

### 公開

- Site: https://aster-learning-lab-zhong.rynaka0112.chatgpt.site
- 公開版: 5。
- Site source: `06bdf314106c06d54daad8faefb76d74e6e46354`
- Deployment: `appgdep_6ab8e2ae8a6081919bef64d27f51ff5e`
- 公開成功: 2026-09-27 18:32 JST。本人限定の共有範囲を維持。
- この文書変更はAsterのdocs/sites-workbench-contract-v0ブランチに記録し、main向けPRとして提出。PR #16を自動マージしない。

### 実機での次の作業

1. PCのAsterへPR #16のService実装を導入。Python環境に学習依存を準備。
2. Asterルートで `python aster_bridge.py --root .` を起動。
3. 同じPCのブラウザでSiteを開き「実行と記録」でキーを入力。
4. 「学習を観察する」で本文とTokenizerを選びPretrain開始。候補がなければ先に本文作成/BPEを行う。
5. 別々のJob/Run ID、進捗、loss、outputsを確認。接続制約が出た場合はその結果を履歴へ追記。
6. 設計契約のPRをmainへ取り込むと、ルートAGENTS.mdから今後の作業で参照される。

### 残る課題

- raw staging/upload、checkpoint取得/推論、Agent実行、遠隔接続は未実装。
- Site側の完全なJob bundle永続化は未実装。Service記録と公開用JSON書出しで追跡する。
- 新しいローカルCorpusをSiteへ送るService APIは未実装。既存共有済みCorpus/共有ファイル方式は利用可能。
- code_sha256と詳細diagnosticsはAster原記録が正本。公開用UIコピーとは区別する。
