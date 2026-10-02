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

## 2026-09-28: Issue #19 DecisionModel CPU baseline実装

### 依頼と基準

- `Aster #19`だけを先に実装し、学習済みDecisionModelをWebへ公開する前に、漏洩のない比較・保存・再読込の基準を作る。
- 基準main: `76516c6ca7e7d83b868c2e68a6232327e80b6ab1`。固定Agent Service v1はmainへmerge済み。
- 対応PR: #22 `feat/decision-baseline-v0`。
- UI/APIを同時に拡張しない。ServiceContract-v1のallowlisted recipe、Job/Run分離、localhost/security境界は変更しない。

### 実装

- `DecisionModel Artifact`を追加。weights、model config、Tokenizer、candidate builder、suite digest、training config、calibration/routing設定、source Git SHAを関連付け、model/Tokenizer digestをload時に検査する。
- optimizer stateは保存せず`resume_supported=false`を明示。推論再読込とtraining resumeを区別する。
- development baselineを`train → calibration → dev`に限定。BPEとweight updateはtrainのみ、temperatureはcalibrationのみ、devでDecision/episode/routingを観測し、testは`sealed`として残す。
- final-testを別entry pointにし、保存済みArtifactをloadしてtestだけを評価する。final-testではweight updateと再calibrationをしない。
- RuleBasedPolicy / ModelPolicy / SelectivePolicy(model+fallback)を同じsplit由来のtaskで比較し、episode successとmodel/fallback/abstain routeを分離する。
- 保存直後のscore再現確認に加え、pytestで別Python processからartifactをloadしてscore/argmaxを比較する検証を追加。
- CPU観測としてthread数、各区間wall time、平均step time、process CPU time、Linux max RSSをRunへ保存する。
- ローカル実行CLI `scripts/run_decision_baseline.py`を追加。

### 検証状態

- PR作成時点ではGitHub Actions pytest / Pyrightを起動済み。結果はこの履歴へ成功確認後に追記する。
- ユーザーPC上の実測Run ID、実行SHA、step速度、最大RSS、dev/final-test結果は未取得。GitHub CI成功をユーザーPCの能力実測として扱わない。
- 現行`calculate-and-store-v0`は小さなsynthetic suiteであり、高scoreを広いAgent能力の証拠にしない。
- Wolframで30分を更新だけに使う単純上限を再確認した場合、平均1秒/stepで1800 step、2秒/stepで900 step。評価・保存時間を含まない計算例であり実測値ではない。

### Sites / Serviceへの影響

- このPRではaster-webを変更しない。
- Serviceの`GET /artifacts`へDecisionModelをまだ公開しない。
- `agent-calculate-store-v0`は引き続き固定RuleBasedPolicyのrecipe。
- Sites Workbenchの責務「操作・表示・比較」、Aster側の責務「実行・評価・canonical evidence」を維持する。

### 次

1. PR #22のpytest / Pyrightを確認し、失敗があれば修正する。
2. ユーザーPCのWSL2/CPUでdevelopment baselineを実行し、Git SHA / Run ID / resource実測 / dev失敗例を記録する。
3. 条件を固定した後、保存済みartifactのfinal-testを一度実行する。
4. そのartifactと固定suiteをServiceへallowlistし、aster-web #2でrule / model / model+fallbackのRun比較へ進む。

## 2026-09-30: Decision fit実測と状態診断CLI（Issue #23 / #24）

- ユーザー依頼: 5Runの実測を#23へ記録し、#24の小さな診断課題を実装。
- 基準: PR #25 / `0fa5736e30d38c6569a1c6419e774d16748235f7`上のstacked research PR。mainへ未導入。
- ユーザーWSLの実測をreports/decision-fit-user-pc-v0.md/.jsonへ保存。3seedでLR0.001/50epoch train fit成立、旧結果やseed43を保持。
- 到達可能な遅延履歴、数値/キー変更、候補逆順の48 dev decision、train-only step対照とfirst-candidate対照、rule/model-onlyの6初期task episodeを実装。
- 保存済み6checkpointをChatGPT Linux CPUで読み込み、元の既知課題は完了、新しい数値/キーの初回probeは失敗。由来・全prediction/trajectory・weights不変・test封印を記録。
- 検証: pytest111 passed、Pyright 0 errors/0 warnings。新CLIをユーザーPCではまだ実行していない。main向けのみの既存CIをstacked PRで実行済みとは呼ばない。
- 変更はCLI/reportsのみ。ServiceContract-v0/v1、公開recipe、artifact catalog、aster-web、Site公開版は変更なし。UIへpath入力や内部Pythonを露出しない。
- 次: 小さなtrain拡張と新split/構造holdoutを設計。公開bundle/Service/Web比較は対応PRで別に進める。

## 2026-09-30: 失敗/Tokenizer監査とデータ介入4条件（Issue #24）

- ユーザー依頼: 次に提案した失敗分析と小さなデータ比較を実際に進める。
- 基準: PR #26 / faec3c4df4982a7dea1d176440fafa0df2504f14。後続stacked research PRで、mainへ未導入。
- キャッシュ済みdev predictionのdigest/正誤/token分割を検査する監査CLIと、元8判断のTokenizerを固定した4条件のequal-budget学習CLIを追加。
- 各arm32提示slot×50epoch、1,600 updates/exposures。unique判断8/16/16/32と反復を区別。seed内の初期weightとshuffle indexを共通にした。
- 元48 dev＋同時数値/キー変更8件とrule/model-only各8episodeを保存。学習後のcheckpointを再読込し、評価中のweights一致と候補coverageを検査。
- 新しい冗長計算prefix生成familyをtestに予約し、全派生をtest groupへ置く。suite内容/教師検証/digestのみ保存しmodel scoreなし。旧testも封印。devはtrain由来の診断で独立holdoutではない。
- 設計/実測はdocs/DecisionDataIntervention-v0.md、reports/decision-failure-audit-v0.md、reports/decision-data-intervention-v0.md/.json。
- 検証: pytest115 passed、Pyright0 errors/0 warnings。実測はChatGPT Linux CPUで、ユーザーWSLでの新CLI実行は未確認。GitHub main向けCIをstacked PRで実行済みとは扱わない。
- ServiceContract-v0/v1、public recipe/artifact catalog、aster-web、Site公開版は変更なし。UI接続/Job化/自動promotionは未実装。
- 次: 実測のtrain fitと元課題保持を踏まえ、Tokenizer/入力表現を独立条件で比較する。Service/Web接続は固定recipeと公開bundleを定義してから別PRで進める。


## 2026-10-02: Research Observatory v0（Aster #29 / aster-web #2）

- 目的: 新しいDecision実験を増やす前に、既存研究結果をAster Service→aster-webで調査できる最小vertical sliceを作る。
- Aster branch: `feat/research-observatory-v0`。main `098a009c35b6a023de2877e12ce6fab2e71f36a4` から分岐。
- `research.observe` capability、`GET /experiments`、`GET /experiments/{experiment_id}` を追加。
- 初版論理IDは `decision-failure-audit-v0`。保存済み `reports/decision-failure-audit-v0.json` をread-only projectionし、source SHA-256、arm/seed、checkpoint/suite/Tokenizer、phase-zero caseのtarget/selected Action、正誤、target candidate tokenizationを返す。
- reportに存在しないlearning curve/candidate scoreは `unavailable_from_committed_evidence` として欠損のまま返す。testはsealed。
- arbitrary path、任意report、private corpus、生ログ公開、評価のWeb再計算は追加しない。
- aster-web側は同branch名でExperiment/2 arm/shared case比較UIを追加。既存fixed Agent consoleは維持。
- 検証: repository unit testとAster GitHub ActionsをPR作成後に確認する。aster-webはCI workflowがないため、remote編集だけではnpm test/build未実行として残す。
- 未検証: ユーザーWSL2でのAster Service起動→実ブラウザ表示。新規learned DecisionModel実行も今回の範囲外。
- 次: 観測系が成立したらsaved DecisionModel + typed stateでlearned Agent end-to-endへ進み、その後Aster #20の親→訂正→candidate比較へ接続する。


## 2026-10-02: Research Observatory実機Gate完了 → Learned Agent ACT v0開始

- ユーザーWSL2実機ブラウザでResearch Observatoryを確認。shared case `subtract/numbers/phase-0/delay-0` を
  `seed-42--eight-fixed`（persisted false）と `seed-42--eight-shuffle`（persisted true）で左右比較できた。
- scoreはcommit済みevidenceに無いため `unavailable_from_committed_evidence` のまま表示され、Web再計算なしを確認。
- 既存fixed Agent Runも同実機で calculator → memory.put → memory.get → stop、task_success=trueを確認。
- Aster PR #47へ実機結果を追記しReady for reviewへ変更。aster-web PR #4はmerge済み。
- 次GateとしてIssue #48 / branch `feat/learned-agent-act-v0` を開始。
- Spec/TODO/Progressを追加し、ACT-WiringとACT-Capabilityを分離。
- `decision_model` Artifact kind、検証登録CLI、model-only ACT runner、`decision-traces.jsonl`、logical Artifact入力のService recipeを実装中。
- aster-webのACT表示はAster側canonical evidence契約とCIが固まった後に接続する。
- 現時点でACTコードのGitHub CIとユーザーWSL2実測は未確認。能力成功を主張しない。


## 2026-10-03: ACT v0 Artifact promotion追補

- ACT v0 PR #49はmainへmerge済み。
- 実機でformal DecisionModel Artifactが見つからず、既存研究Runの主保存物が `decision_fit:<digest>` checkpointであることを確認。
- committed summary/reportはweights本体を含まないため、reportだけから推論Artifactを再構成しない。
- verified fit checkpointをformal `decision_model:<digest>` へ変換する明示的promotion CLIを追加。
- fit checkpointはcalibration未実行なので、promotion後も `calibration=not_run` / `routing=not_configured` とし、架空の校正を作らない。
- Service/Webは引き続きlogical DecisionModel Artifactだけを扱い、local checkpoint pathはpromotion CLI境界の外へ出さない。
