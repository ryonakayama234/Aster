# Sites Workbench 作業履歴

## 2026-10-01: 停止時点の正保存・最終書込み後の取得確認に合意

- PR #34の問い0にユーザーが合意。docs/CurrentStateGoal-v1.mdを正本とし、明示選択のv1を教師・候補・独立評価器・Trace/episode保存へ追加。
- 対象キーputで達成を取消し、修復と再取得で再達成する。同値再保存も取得を要求し、他キーputでは確認を維持。
- Traceに契約ID/schemaを保存し、v0記録を維持。UI将来契約へ過去達成と停止成功の区別を追記。
- 全pytest154 passed、新規11件の最終再確認成功、Pyright0 errors/0 warnings。1,364実行経路・6,372途中判断を独立状態機械と照合。reports/current-state-goal-v1.mdに検証・限界を保存。
- PR #34へ積むfeat/current-state-goal-v1のDraft。main/Service既定recipe/aster-web/Site公開版への導入、学習済みモデルの採点と再学習は未実施。
- 次：保存済みモデルで誤値と履歴回数を分ける診断の条件を別protocolに固定。旧実測の契約を改名しない。

## 2026-10-01: 状態対照trainの全6条件測定

- PR #32 head ee61711483dd876d35728a1999f6def6966b328f上で、compact/Byte/モデル/1600更新を固定し学習データ32slotのうち16を実Toolの誤保存prefixへ置換。正常教師の頻度8件ずつとseed内全初期重み・slot permutationを保持。
- 無関係なdelayはcompactで同じ入力になることを検査。冗長計算prefixの予約test familyは学習/devに使わず、旧/予約testは構築・採点0件。
- 新dev32判断のうちunion-train同一入力4件を保持確認に分離。非重複28判断と同step/同候補主対照6組を指標にした。原48/旧56判断、初期自力8episodeも保存。
- seed42/43/44のbaseline fitは3/3、stateは2/3。新状態判断はbaseline16/17/13対state10/8/9 (各28件)、元初期完了はbaseline各2/2対state1/2・0/2・1/2。主対照改善なし、数値初期完了全arm0/2。この置換方式を採用しない。
- 事前protocolと実測MD/JSON・全evidence hashを保存。pytest143成功、Pyright0 errors/0 warnings。9,792train予測、1,152dev保存行（重複336行）、576episodeを監査。実測source clean local9b8f1ab、LinuxCPU/Python3.12.14/torch2.14.0+cpu/2threads。ユーザーWSL未測定。
- 正常例を外した影響と誤値/履歴依存の切り分けを次の保存モデル固定診断案にする。次案は未実装・未測定。
- PR #32をbaseとするDraftでmain/Service/API/公開bundle/aster-web/Site公開/モデルpromotionは未変更。観測接続は#29→aster-web#2へ継続。

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

## 2026-09-30: Decision Tokenizer比較の配線（Issue #28 / PR #30）

- PR #22/#26/#27をmainへ統合した後、同じ32 unique decisionで専用BPEとByteTokenizerだけを変える比較harnessをDraft PR #30へ追加。
- 保存済みBPEは公開digestを既定で要求し、本測定時に再学習しない。Byte側は既存UTF-8 byte ID 0..255へBOS/EOSを加えたDecision artifactとして扱う。
- shapeが共通のTransformer/Decision Head等と、意味が対応するraw-byte/BOS/EOS行だけを共有初期化し、全weight同一とは記録しない。
- token ID・UTF-8 byte range・系列長・padding込みtoken量、parameter数、learning curve、reload、56 dev decision、model-only episodeの最初の誤答をAster Runへ保存する実装。
- reserved testはsuite/digestだけを保存し、Tokenizer長さ確認にもscoreにも使わない。old testも封印を維持。
- GitHub標準CPU runnerの1-epoch debug wiringでは119 pytestが成功し、Pyrightも成功。これはIssue #28のseed42/43/44実測には数えない。
- Service/API/aster-web/Site公開版はこのPRで変更しない。将来Issue #29 → aster-web #2で比較bundleを公開するときも、UIはAsterが保存したscore/success/token境界を再計算しない。

## 2026-09-30: Decision入力比較の全6arm測定と研究契約

- PR #30の初期化条件でseed42/43/44×BPE/Byteを50epoch測定。監査に保存logits/選択Action/数値指標の照合を追加し、各arm30分のdeadlineを強制した。
- Byte train fitは3/3seed、BPEは1/3seed。数値変更episodeは両方式全seed0/2。キー変更の安定改善なし。次は#24の到達可能な状態と短い構造化入力の比較を事前設計する。
- 関連回帰64test、deadline検査を含む比較5test成功。9,792 train predictionと336 dev行を独立照合。Linux CPUの測定、ユーザーWSL測定は未実施。旧test/予約test未評価。過去の元cached auditログの再照合は未実施。
- reports/decision-tokenizer-comparison-v0.md/.jsonへ全条件と証拠を保存。devは親trainの2groupを共有し独立holdoutではない。
- 公開bundle (#29)、aster-web #2、Service recipe/UI/Site公開版は未変更。今回の計測値をUIに導入済みとは扱わない。

## 2026-10-01: raw/compact比較の実装・測定準備

- PR #30とPR #31の研究実装を統合し、#24の原入力/短縮入力比較runnerを追加。事前条件はdocs/DecisionRepresentationComparison-v0.md。
- 既存32train・Byte・全初期weight/提示順/候補順を固定。48状態devと既存56dev・8初期episode、prefixからのmodel-only継続を別記録する。
- checkpointのserializer不一致を拒否。候補のみ対照はtrain頻度lookupでありニューラル対照ではないと本測定前に定義。
- 全pytest139成功、Pyright0 errors/0 warnings。短いdebugは改善実測へ算入しない。本測定は次に開始する。
- Service/API/aster-web/Site公開版は変更なし。保存結果は#29→aster-web#2へ渡す予定で、比較画面へ接続済みとは扱わない。

## 2026-10-01: raw/compact全6arm測定完了

- 固定した条件でseed42/43/44を完了。compactはtrain fit3/3、raw1/3。48状態判断はraw22/21/17、compact39/36/33。
- 数値変更の最初からの自力完了は全arm0/2。prefix継続のinvalid（raw3/2/2、compact0/0/0）は失敗へ混ぜず、共通評価可能prefixで比較した。
- reports/decision-representation-comparison-v0.md/.jsonへ全条件・曲線・Run/source/protocol/evidence hash・次の反証可能な状態train比較案を保存。旧/予約testは構築も採点もしない。
- 保存9,792train予測と624dev予測を再照合し、実trajectory/UTF-8 byte列/選択/evaluation/reload/weights/RNG不変を検査。実測はLinux CPU/torch2.14.1でユーザーWSL測定ではない。
- #31をbaseとする後続Draft PRへ提出。#30の前提コードも統合した。ServiceContract、公開recipe、aster-web、Site公開版、モデル採用は変更なし。

## 2026-10-01: 固定6モデルのcurrent-state v1診断

- PR #35をbaseとして、旧v0学習の6checkpointを更新0で診断。考え方・固定条件・停止条件をdocs/DecisionFixedV1Diagnostics-v0.mdへ事前登録。測定source dc05bb7とPython/protocol hashをRunへ保存。
- 128ケース/modelを実Toolで生成・v1再生。正常保持、誤値×反復、完了取消、同候補・異target対照を分離。全6modelのscore/token ID/自力768episodeを保存。
- baselineは既知正常8/8、置換モデル5〜6/8。誤値変更によるAction変化0〜2/32組に対し、反復回数変更9〜28/32組。実測・限界・次の正常例保持案をreportsへ保存。
- 失敗配線Runも保持。独立cached auditで768固定予測・192対照・3,247訪問予測・5,119transition・768episodeの停止時正しさを照合。全pytest159 passed、Pyright0 errors/0 warnings。Linux CPU実測でユーザーWSLは未測定。
- Service/API/Web/公開recipe/モデルpromotionは変更なし。契約違い・candidate情報・invalidの将来表示契約だけ更新。保存結果の観測は#29→aster-web#2。次の混合学習は未実装・未測定。


## 2026-10-01：Aster全体の照合と次のv1混合比較計画

- 添付Aster計画PDF、両repo main、Aster PR #36 head 5291a377abb3140cbf4255ec094738cbf0b7c3aa、研究report/生成コード、一次研究を照合。
- docs/DecisionMixedTraining-v0.mdへ全体像、正常例を残す64slotの2条件、v1教師・候補・評価の一致、preflight、指標、停止条件・分岐を記録。
- Wolframで64×25=1,600更新/arm、6arm=9,600、Action各400提示を確認。正常例提示はcontrol1,600/mixed1,200で異なることを明記。
- 文書・算術検証のみ。生成preflight、実装、学習、能力新測定、ユーザーWSL/ブラウザ確認は未実施。pytest/CI成功を今回の成果として主張しない。
- branch docs/decision-mixed-v1-next-stepはPR #36 headを基準とする文書のみのDraft。API/Service/aster-web/Site公開/モデル採用は変更しない。
- 次：同計画の実Tool生成preflightとdigest正式登録。観測#29→aster-web#2、言語#21、育成#20を別経路で維持。ユーザー側の追加作業不要。

## 2026-10-01：v1混合学習前の実Tool生成・保存監査

- 基準PR #37 ab7f817。正常32＋復帰16、dev128、64slotの両armを生成しv1教師/候補/実行/評価へ整合。
- 176例/888transitionの独立停止時確認、衝突0、max1190/2048。train同候補異target8組、dev重複14・非重複主対照24組を再計算。
- formal clean source f3fac1b66726649ed39f2e09006c2ecea14c1ebf、Run c565a9d6d6f449448c451ae9bf438d66、Python/torch/hashをreportsへ保存。LinuxCPUでユーザーWSL未測定。
- 全pytest163、最終関連4、Pyright0/0。別saved監査で176例/128slotと分母・digestを照合、slot改変拒否。
- モデル推論/学習更新/旧予約test構築採点0。今回の成果は生成配線で、能力改善ではない。
- PR #37 baseのfeat/decision-mixed-v1-preflight。main/Service/API/aster-web/公開版/モデル採用は未変更。次は固定digestを検査する比較学習runner。ユーザー側作業不要。

## 2026-10-01：v1混合学習の全6条件実測

- PR #38をbaseとする研究runner、独立cached audit、全seed軽量reportを追加。正常32/復帰16の登録digestを実行前に検査。
- 全6条件stable fitと既知正常保持、主対照の安定改善なし。prefixはseed依存改善、数値初期は混合seed44だけ1/2。
- pytest165 passed、Pyright0 errors/0 warnings。768episode/5,441transitionを実Tool再実行で監査。正式271.082秒、9,600更新。ユーザーPC測定ではない。
- 全checkpoint/予測/順序/経路を再現bundleへ保持。旧test未使用、Service/API/aster-web/公開recipe/promotionは変更なし。次の関係明示入力は未実装。ユーザー側追加作業不要。

## 2026-10-01：観測関係入力比較の事前登録と配線

- PR #39 head 66de75cを基準に、正常＋復帰の48unique/64slotとモデル/予算を固定するcompact/relations比較を実装。
- 教師/evaluation/期待解に依存せずraw Tool観測をscanし、一致・保存取得順を外部抽出。系列長/履歴正規化も変わる診断介入として明記。
- 登録preflight衝突0、最大1190/519tokens、共通非重複主対照24組。正式能力測定は未実施。
- checkpoint serializer guard、reload、候補逆順、weights/RNG不変、独立実Tool再生監査を継承。検証と正式結果は後続追記。
- Service/API/aster-web/公開版/モデル採用は変更なし。ユーザー側の追加作業不要。

## 2026-10-01：観測関係比較の全6条件測定と負の結果保存

- clean local3538a25、同treeのAPI source18c4e2d。Run7d4660729cf74cf3a4aab123ff23d05c、正式259.895秒。
- 両入力fit/正常保持。共通非重複24主対照は4/2/2→0/0/0、共通prefixは51/58/69→37/39/39。
  個別判断110分母ではrelationsが増えたことも保持。dev重複14/18と共通除外18を区別。
- pytest170/型0 errors/0 warnings、独立監査train7488/dev768/episode768/transition6071/訪問4199。
  再読込/順序/weights/RNGと全実Tool再生を確認。ユーザーWSL未測定。
- 測定後train-only boolean関係パターン表は104/128をカバーし全正答。外部抽出/非ニューラルの探索と明記。
  次のAction種類/引数分離は未実装。全modelを保持し自動採用しない。
- reports/decision-relations-study-v0.md/.json、全記録bundleへ保存。旧/予約test0。Service/API/aster-web/公開版は未変更。

## 2026-10-02：行動種類診断の学習前検証と後続言語設計

- PR #40を基準に3数値特徴/16parameter分類器と候補引数結合を実装。48train/64slot、25epoch、3seed、lr0.01を事前固定。
- preflightは4特徴パターン、衝突0。dev128/128と新初期4/4が特徴trainと重複するので未知特徴の汎化とは扱わない。
- 5関連test成功、Pyright0 errors/0 warnings。debugは正式実測へ加算しない。全test/正式測定は後続追記。
- docs/LanguageToActionRoadmap-v0.mdへ依頼/Tool観測→事実/Task読解の接続を設計。自由会話/コード生成/言語学習は未実施。
- Service/API/aster-web/公開版/モデルpromotionは変更なし。ユーザー側追加作業不要。

## 2026-10-02：16parameterの行動種類学習・全3seed成立

- clean local0817a1d/API同tree d6f4bc6、Run212b32fd029640ce8efc89930b5c4403、正式4.464秒（全pytest併走で速度benchではない）。
- 全seed fit48/48、種類/結合128/128、対照32/32、prefix128/128、各初期variant2/2、新task4/4。特徴空間は全てtrain重複と明示。
- pytest175/新規5、型0/0、独立監査train3744/dev384/episode396/transition2088/訪問1152。reload・weights/RNG不変・source不変を確認。
- 学習器はC/M/Fから種類を選ぶ。事実抽出/引数供給は外部実装。一般的なTool失敗復帰・生履歴読解/コード生成は未確認。
- 後続言語設計は事前学習+課題用reader学習、ランダム/事前学習初期化比較。実素材/Tokenizer/context互換性の確認が次。言語学習未実施。
- reportsへ正本保存、全weights/証拠bundle保持。main/Service/API/aster-web/公開版/自動promotion未変更。ユーザー側追加作業不要。
