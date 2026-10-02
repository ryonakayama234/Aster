# Aster

**作って、動かして、仕組みを理解するための自作TinyLM。**

AIの仕組み、Python、数学、コンピューターサイエンスを、ひとつの小さな言語モデル作りにつなげます。
出来上がったモデルだけでなく「なぜこう設計したか」「どう確かめたか」を残します。

将来の目標は、外部の計算器・記録・検索・実行環境を使い、自分の処理を進められるAsterです。
道具の呼び出し方に加え、返った結果を読み、失敗を修正し、記録を次の行動に使うところまで扱います。

## コーパスの方針

| 領域 | 作り手 | 役割 |
| --- | --- | --- |
| 日本語prose | ユーザーの自作文・選定した外部文章 | 語り方・説明・視点 |
| 日本語dialogue | ユーザーの自作文・選定した対話・AI対話 | 対話表現。応答のお手本への採用は別途選定 |
| Math | 外部調達 | 問題、式、解答の関係 |
| Python | 外部調達 | 基礎文法、短い処理、入出力 |
| structured / JSON / tool | Asterプロジェクト内 | 構造、道具の実行、観測、記録と再利用 |

詳細は [AsterCorpus-v0仕様案](docs/corpus/AsterCorpus-v0.md) と
[外部調達仕様](docs/corpus/external-sourcing-v0.md) を参照してください。
これらは初回提案で、v0の完成・凍結を意味しません。

2026-09-20：ユーザー選定の外部日本語・AI対話も候補に含める方針へ更新しました。
[採用表](docs/corpus/selection-review-v0.md) と [出典台帳](docs/corpus/source-notes-v0.json) を入口に整理します。
出典台帳は人が編集する情報、採用表は `scripts/inventory_corpus.py` で再生成する調査結果です。

## 現在あるもの

- UTF-8 byte tokenizerと、隣接する頻出ペアをまとめるBPE。
- Tokenizerの保存・読み込みとCLI。[既存BPE仕様](src/docs/AsterBPE-v0.1.md)。
- テキストを `id / text / source` のJSONLへ変換する処理。
- 小さなサンプルとテスト。
- [TinyLM本体と事前学習の最小ループ](docs/TinyLM-v0.md)。CPUで少量を覚える実験、train/dev評価、重みの保存・再読込・生成。
- 外部資料の原本収集と、[canonicalへの自動変換](docs/corpus/canonical-v0.md)。
- [学習用本文の抽出・分割・配合](docs/corpus/training-view-v0.md)、BPE実験、UIで使う観測bundle。[最初の結果](reports/training-pilot-v0.md)。
- Agent KernelのPolicy → ToolExecutor → Observation → State → Evaluatorと、固定`agent-calculate-store-v0`をAster Serviceから実行・観測する経路。[ServiceContract-v1](docs/ServiceContract-v1.md)。
- TinyLM shared backbone + scalar Decision Head、Decision benchmark、temperature calibration、model/fallback/abstain routing。
- [Decision CPU Baseline v0](docs/DecisionBaseline-v0.md)。BPE/weight update=train、temperature=calibration、design observation=dev、test=別runという分離、DecisionModel Artifact、別process reload、rule/model/model+fallback比較を実装。
- [Decision fit実験](docs/DecisionFitExperiment-v0.md)と[状態診断](docs/DecisionStateDiagnostics-v0.md)。train fitと状態変化への汎化を分け、保存checkpointを更新せずに数値・キー・履歴変更を診断する。
- [失敗/Tokenizer監査](reports/decision-failure-audit-v0.md)と[データ4条件比較](docs/DecisionDataIntervention-v0.md)。候補coverage、train fit、model-only episode、旧test/予約testの封印を分離して記録する。

JSONLは一行に一つのJSONを入れる保存形式です。保存にJSONを使うことと、モデルにJSONの生成を教えることは別です。
現在のTokenizerビルダーは `.txt` を読みます。コーパス変換器のJSONLをそのまま渡す接続にはなっていません。

## 学ぶ順序

1. **Corpus**：文字列、ファイル、文字コード、出典、学習と評価の分割。
2. **Tokenizer**：bytes、辞書、ペア頻度、圧縮、可逆変換。
3. **TinyLM**：ベクトル、行列、確率、次token予測、損失と勾配。
4. **実験**：未知の入力、過学習、比較条件、再現性。
5. **道具と記憶**：型、状態、実行、検証、保存と再利用。

各段階で「小さな例を手で追う → 実装を見る → 一つ変更する → 結果を比べる」を基本にします。

## 手元で確認する

WSLのリポジトリルートで、既存の仮想環境を使う例です。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/fetch_external_seed.py
.venv/bin/python scripts/build_canonical.py
.venv/bin/python scripts/build_training.py
```

後者はGSM8Kのtrain原本とPython教材の選定章・権利表示を取得します。
[収集manifest](docs/corpus/external-seed-manifest.json) に固定revisionとSHA-256を記録し、再実行時には一致を検査します。
原本はGit管理外の `data/raw/pool/external-seed/` へ保存します。
canonical変換は台帳からpoolを直接読み、`data/canonical/builds/`に版ごとの共通データを保存します。
`data/raw/current/`は採用原本を集める用途として残しますが、変換の必須入力にはしません。
採用・train/dev/test分割を経たpilot学習用入力は`data/training/<view ID>/`へ出力します。
train本文 → Tokenizerでtoken ID列へ変換 → TinyLM内のembeddingでベクトルへ変換、という流れです。
embeddingの数値はTinyLMの学習で更新します。実行方法は [TinyLM-v0](docs/TinyLM-v0.md) を参照してください。
収集済みというだけで学習可能とは扱わず、採用前に仕様の確認を行います。

Decision baselineはtestを自動開封しません。まずdevelopment runを作り、条件を固定した後だけ保存済みartifactをtestします。

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 train
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 test \
  --artifact runs/<development-run-id>/model-artifact
```

学習支援・開発時の約束は [AGENTS.md](AGENTS.md) にまとめています。

## 状態判断の入力比較（2026-10-01、研究PR）

[固定条件](docs/DecisionRepresentationComparison-v0.md)で原入力と短縮入力の全6armを測定し、[実測](reports/decision-representation-comparison-v0.md)へ保存。48状態判断はraw22/21/17、compact39/36/33。数値変更の自力初期episodeは全arm0/2。入力整理の効果と未知の値への対応を分け、次は状態対照trainの別実験を事前設計する。PR #31をbaseとする後続Draftで、Service/Web/model promotionは未実施。

## 次に作るもの

2026-10-01: PR #31で状態判断と短縮入力のpreflightを追加。PR #30ではByteが32判断を3/3seedでfitしたが数値変更episodeは全seed0/2。次は同じByte/初期weight/学習予算でraw/compactを比較する。両PRはDraftでありService/Web公開は未実装。


1. Issue #28 / Draft PR #30で、同じ32 unique decisionを使った専用BPE対ByteTokenizer比較を実装・実測する。[比較protocol](docs/DecisionTokenizerComparison-v0.md)では保存済みBPEをdigest固定で読み、同じ更新順・共有可能な初期重み・token/parameter/CPU費用を記録する。seed42/43/44測定はreports/decision-tokenizer-comparison-v0.mdに保存済み。
2. Issue #29 → `aster-web` #2で、保存実験の学習曲線、最初の誤答、token分割、candidate score、state、episode伝播をAster正本bundleから比較表示する。Web側でsuccess/rewardを再計算しない。
3. Issue #20で、Agent実行 → teacher訂正 → candidate再学習 → 同一Evaluatorで親子比較、までの最初のLearning Loopを閉じる。改善自体は完了条件にしない。
4. Issue #21のTinyLM言語pilotはDecision研究と別レーンで進め、train暗記とheld-out言語能力を混同しない。

Aster Service / aster-webは任意shellやrepository-local pathを公開せず、Job / Run / Artifactを分離する方針を維持します。

2026-10-01: [完了契約v1](docs/CurrentStateGoal-v1.md)に合意し、明示選択の教師・候補・評価器・Trace保存を追加。[検証結果](reports/current-state-goal-v1.md)。誤上書き後の修復と再取得を要求する。既存v0測定・既定recipeは保持し、モデル能力の新測定とService/Web導入は未実施。

2026-10-01: PR #32のcompact/Byteを固定した[状態対照train比較](docs/DecisionStateTraining-v0.md)を研究ブランチに追加し、seed42/43/44を測定。
[実測](reports/decision-state-training-v0.md)では正常経路の一部を誤保存prefixで置換した条件は改善せず、元課題保持も低下。train fitはbaseline3/3・state2/3seed。数値変更の自力完了は両条件全seed0/2。
同じ候補集合で正解が変わる主対照、compact入力重複を除く診断、実Tool replayと全条件の証拠を残した。次は保存モデルを固定した正常経路/誤値/履歴長の切り分けを設計する。main/Service/Web/モデル採用には未導入。

## Decision入力比較の実測（2026-09-30）

[PR #30の条件による全6armの報告](reports/decision-tokenizer-comparison-v0.md)。Byteは3/3seedで32判断の最後3epoch fit成立、専用BPEは1/3seed。数値変更episodeは両方式全seedで0/2、キー変更はByte seed42の1/2のみ。Tokenizer置換だけで安定対応したとは扱わず、次は状態判断と短い構造化表現を比較する。反証可能な研究の継続契約をAGENTS.mdへ追加。PRはDraft、Service/Webへは未公開。

2026-10-01: 保存済み6モデルを更新せずcurrent-state v1で診断。[事前計画](docs/DecisionFixedV1Diagnostics-v0.md)と[実測](reports/decision-fixed-v1-diagnostics-v0.md)。正常例置換モデルは既知正常判断5〜6/8、対照は全seed8/8。誤値変更より関連put反復による選択変化が大きい。次は正常例保持の混合対照を別protocolで設計する。今回の診断とv1は研究Draft上で、main/Service/aster-web/公開モデル導入は未実施。

### 2026-10-01：次のv1学習比較計画（未測定）

[全体像と正常例保持の混合比較](docs/DecisionMixedTraining-v0.md)を研究ブランチへ追加。
PR #36の診断を受け、v1で正常経路を残す2条件を、各1,600更新・各Action400提示へ揃える案。
正常例の個別exposure差、旧v0からの仕様変更、同family devの限界を明記。
今回の成果は調査と計画保存。生成preflight・学習runner・新しい能力実測・Service/aster-web公開は未実施。

### 2026-10-01：v1混合比較の生成preflight成立

[生成検証](docs/DecisionMixedPreflight-v0.md)と[実測](reports/decision-mixed-v1-preflight-v0.md)を研究ブランチへ追加。正常32＋復帰16unique、dev128を実Toolで検証、衝突0・非重複主対照24組・context内。モデル推論/学習は0で能力改善ではない。` .venv/bin/python scripts/run_decision_mixed_preflight.py --root . `で再生成し、`scripts/audit_decision_mixed_preflight.py runs/<Run ID>`で保存証拠を照合する。次は固定digestで比較学習runner。main/Service/Webへ未導入。

### 2026-10-01：v1混合学習の全6条件実測

[事前登録](docs/DecisionMixedMeasurement-v0.md)と[実測](reports/decision-mixed-v1-study-v0.md)。正常例を保持する両条件で全3seed stable fit、共通正常32/32、既知初期2/2。非重複主対照24組は対照4/4/4、混合4/2/2で安定改善なし。共通有効prefix成功差は-8/+7/+29、未知数値初期完了は混合seed44だけ1/2。他5条件0/2。次は観測事実の関係を明示する入力との対照を事前登録する（未実装）。研究CLIのみ、main/Service/Web/公開モデルへ未導入。

実行：`.venv/bin/python scripts/run_decision_mixed_v1.py --root .`。保存監査：`.venv/bin/python scripts/audit_decision_mixed_v1.py runs/<Run ID>`。旧testは未使用。

### 2026-10-01：関係抽出とAction選択を切り分ける比較

PR #39の正常例保持混合trainを固定し、compact入力と観測関係を明示する入力を比較する研究ブランチ。
[事前条件](docs/DecisionRelationsComparison-v0.md)。登録preflightでは48train/128devの衝突0、
共通非重複主対照24組、最大系列長compact1190/relations519を確認。これは学習前検証であり能力改善ではない。
Service/aster-web/公開モデルへ未導入。正式学習結果は測定完了後にreportsへ記録する。

2026-10-01正式比較完了: [全6条件report](reports/decision-relations-study-v0.md)。両方式全seed stable fit/正常保持。
共通非重複24対照組はcompact4/2/2→relations0/0/0、共通prefix完了も低下。関係入力を採用しない。
測定後のtrain-only関係パターン表はdev104/128をカバーし全て正答したが、非ニューラル/外部抽出の探索対照。
次はAction種類の選択と候補引数の結び付けを分ける診断案。未実装。公開モデル/Service/Webは未変更。

### 2026-10-02：数値特徴から行動種類を学習する診断

[事前条件](docs/DecisionActionTypes-v0.md)に基づき、観測済みC/M/Fの3特徴→4Action種類を16parameterの線形分類器で学習する研究ブランチ。
引数は既存candidateへ結合。生履歴読解と引数生成は外部実装へ残す。preflightでtrain特徴4種、dev128/128・新初期4/4の特徴重複を確認。
従って未知特徴の汎化とは呼ばない。正式結果は測定後にreportsへ保存する。
[言語読解への接続設計](docs/LanguageToActionRoadmap-v0.md)も追加。次token事前学習と課題用の事実抽出学習を分け、ランダム/事前学習初期化を同条件で比較する案。言語学習は未実施。

2026-10-02正式診断完了: [全3seed結果](reports/decision-action-types-study-v0.md)。train fit、128判断/128prefix、対照32組、追加4初期taskが全seed成立。
外部抽出C/M/Fの4既知特徴パターンとcandidate結合を使う成果で、生履歴読解/引数生成/未知規則の汎化ではない。
次は文字列→事実/Task readerの同条件ランダム/事前学習初期化比較を準備。言語事前学習は今回未実施。main/Service/Web/モデル自動採用は未変更。

## 2026-10-02: 次の文字列読解課題

[HistoryFactReader-v0](docs/HistoryFactReader-v0.md)と `scripts/run_history_reader_preflight.py` に、実Tool履歴→C/M/Fの学習前検証を追加。task/current memory/Action/Observationのみを入力にし、教師/候補/評価フラグを除外します。8 train taskと4 dev task、各10prefix。モデル学習・推論、事前学習比較、Service/Web公開は未実施。PR #41に依存する研究ブランチです。

```bash
PYTHONPATH=src python scripts/run_history_reader_preflight.py --root .
```

正本は [学習前検証report](reports/history-reader-preflight-v0.md)。生成成立と読解モデルの能力を分けて報告します。

## 履歴readerの学習診断（研究ブランチ）

[固定学習protocol](docs/HistoryFactReaderTraining-v0.md)を追加。Byte/TinyLM width16/context2048のreaderでC/M/Fを読み、PR #41の保存済み16parameter分類器へ渡すCLI `scripts/run_history_reader_training.py` と保存監査 `scripts/audit_history_reader_training.py` を実装。正式3seedの結果は `reports/history-reader-study-v0.md/.json` へ記録する。一般Corpus事前学習・依頼文→Task・Service/Web公開はこの診断に含めない。

2026-10-02 reader実測: 全3seed train80/80安定fit・既知初期8/8。dev事実同時21/24/20（/40）、devprefix完了6/12/3（/40）、dev初期0/1/0（/4）。既知taskの途中prefix完了も50/48/41（/80）で、fitと訪問履歴の読解を分離。正本[実測報告](reports/history-reader-study-v0.md)。モデル採用せず、次は保存モデル更新0で値/対象/取得順の診断。

2026-10-02: 次の固定reader診断を[HistoryReaderFrozenContrasts-v0](docs/HistoryReaderFrozenContrasts-v0.md)へ登録。名前・値一致・取得順の24組/40uniqueを実Toolで生成・監査し、保存済み3readerを更新0で測定する。生成成功は読解能力を意味しない。正式実測は後続reportへ記録。PR #43依存の研究ブランチで、main/Service/Web未導入。

固定reader対照の[正式結果](reports/history-reader-contrasts-study-v0.md)：名前の既知task4/4を全3seedで保持。一致/取得順の両側正答は既知・既存devとも各0/4。取得確認済みでも同じgetを繰り返す反例を実測。全190test/型検査/独立監査成功。次は順序対照だけをtrainへ追加する更新量対照の設計候補で、追加学習は未実施。
