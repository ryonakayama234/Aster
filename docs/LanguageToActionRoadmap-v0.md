# 言語事前学習を行動へつなぐ次の設計 v0

2026-10-02 JST。ユーザー意図：「事実→行動選択が成立したら文字列読解側で言語事前学習を組む」。
本書は後続設計で、言語学習や能力確認を実施した記録ではない。

## 会話・構文・事実読解のどれか

当面の優先は**依頼文とTool観測の内容を読み、実行に必要な構造へ変換する能力**。
例えば依頼のoperation/left/right/store_asの同定、観測の成功/失敗・値・対象キーの同定、
観測履歴と現在memoryからC/M/Fの判断。自由会話の自然さや任意Pythonコード生成とは別に測る。
Pythonの構文/技術文章は事前学習素材として有用な候補だが、構文を覚えることと
正しいプログラムを選び実行できることを同一視しない。

## 次token事前学習と課題用学習の役割

事前学習: 許可されたCorpus train本文で次token予測し、文字列の表現を学ぶ。
課題用学習: 実Toolで検証したtrain履歴→観測事実/Taskの対応で読み取りを教える。
次token損失低下だけでC/M/Fの抽出能力が成立したとは扱わない。
先に**同じ読み取りモデルでランダム初期化と事前学習初期化を比べる**設計を登録する。
既存Decision幅16/ByteとTinyLM pilot幅64/BPEは契約が異なるため重みを無条件に移さない。
Tokenizer/語彙/context/architectureを固定してから同じ課題学習予算・分割で比較する。

## 接続する順序（未実装）

1. 数値特徴→Action種類の分類器を保存し、契約/入力/結合/正常保持を固定する。
2. まず実履歴の**構造化文字列→C/M/F**の読解を扱う。対象情報を外部抽出する範囲を記録。
   context長・入力の切り取りで必要情報が消える場合は、モデル学習前に直す。
   正解との衝突0、train/dev分割、データfamily/入力digestを登録する。
3. readerの出力で固定Action分類器を実行。正しい事実を渡した場合と、予測事実を渡した場合を比較。
   個々の事実のprecision/recall/未知、誤停止/誤保存、model-only完了を別々に評価する。
   readerの外した事実を黙って真値へ差し替えない。未確認/矛盾は明示した停止経路へ送る。
4. 続いて短い依頼文→Task field。数値/キー/操作の抽出、言い換え、否定/条件/曖昧さを別sliceにする。
   実行前のvalidationと観測ログは外部Runtimeで保持。言語生成したTool出力を実観測扱いしない。
5. 自由対話/コード生成は独立した追加目標とする。上の小課題成功を一般言語能力と呼ばない。

## Corpus・評価・CPU予算

既存の原本/台帳/canonical/training viewを調査してから最小pilotを選ぶ。
日本語prose/dialogueは本人の素材・本人が選んだ資料を尊重し、こちらで無断収集/代筆しない。
Python/Mathは既存の出典・版・利用条件を確認した素材を使う。structured/trajectoryは実Tool由来。
BPE/事前学習/課題用weight更新はtrainのみ。devを条件選びへ使ったら独立holdoutと呼ばない。
元文書/会話/生成familyを跨いだ分割、読解評価例が一般Corpusへ混入していないかも検査する。
既存の旧/予約Decision testを言語学習へ移さない。新読解protocolの封印testを別途登録する。
CPU pilotは短い速度/長さ確認から予算を決める。有料計算資源を追加しない。

## 成功の範囲とユーザーの役割

特徴分類器成功は「外部で関係を抽出・引数を用意した上で、学習器が行動種類を選べる」という足場。
文字列読解まで解決した証拠ではない。特徴入力が全てtrain重複なら新しい規則の獲得とも呼ばない。
次のreader比較の素材/分割/モデル互換性を確認してから、実装・言語事前学習へ進む。
今回は設計保存まで。私的日本語素材や本人の採用判断が必要になったときだけ具体的な作業を依頼する。
観測#29→aster-web#2、言語#21へ接続。公開Service/Webや自動promotionは別作業。

## 既存pilotとの互換性確認（2026-10-02）

repoのreports/training-pilot-v0.mdは本文/BPEの2026-09-20記録で、言語学習実測ではない。
train110レコード/188429bytes、BPE512+特殊2、日本語/Python/Mathを含みtoolデータなし。
configs/tinylm-pilot-v0.jsonはwidth64/layers2/heads4/context128/batch8/200stepの動作確認条件。
本checkoutには私的原文/training view/Tokenizer artifactが同梱されていないため、
本文・採用・splitを確認せずこの記録だけで言語pilotを開始しない。
読解用の入力長とTokenizer可逆性、Corpusからの評価例除外、現行素材の所在を確認するのが次の準備。

## 2026-10-02: reader学習前検証へ

[HistoryFactReader-v0](HistoryFactReader-v0.md)へ固定120件の実Tool生成・候補非依存入力・unknownラベル・独立JSON oracle・whole-task splitを登録。事前学習素材を待たず実行履歴で準備する。shared templateのparameter-transfer診断で、独立構造holdoutではない。モデル学習/推論とrandom/pretrained比較は後続。

## readerの課題用学習protocol

[HistoryFactReaderTraining-v0](HistoryFactReaderTraining-v0.md)で既存80train/40devを固定。random initializationのreaderと保存済み行動分類器を組み合わせ、oracle事実/予測事実の実行を比較する。一般Corpus事前学習との比較は今回未実施。結果はreports/history-reader-study-v0.md/.jsonへ別記録。

2026-10-02実測: random reader全3seed fit、dev/復帰履歴の汎化未成立。[報告](../reports/history-reader-study-v0.md)。次は固定モデルで対応付け/一致/時系列を切り分ける。事前学習初期化の効果は未測定で、今回の結果だけで必要性・十分性を断定しない。

2026-10-02: readerの追加診断を[HistoryReaderFrozenContrasts-v0](HistoryReaderFrozenContrasts-v0.md)へ事前登録。事前学習が名前・値・順序のどれを助けたか比較できる物差しを先に作る。今回一般Corpus学習/Task readerは未実施。

固定reader対照の[実測](../reports/history-reader-contrasts-study-v0.md)では、同長の値不一致/取得順対照を全seedが選び分けられない。事前学習の必要性は未確定。次は順序関係の課題学習を独立介入として設計し、将来のrandom/pretrained比較に同じ物差しを残す。

2026-10-02順序学習介入: [結果](../reports/history-reader-order-study-v0.md)では一部seedの既知fit改善を確認、安定関係転用なし。一般Corpus事前学習の必要性は未確定。Fだけの課題縮小を先に検討し、事前学習と課題用学習を分ける。

2026-10-02: [最小F読解](HistoryFreshnessMinimal-v0.md)を先に登録。同raw/Byte/backboneの課題教師だけを比較する。一般Corpus事前学習は未実施。F-only成功を値一致・Task読解・一般言語へ外挿しない。

最小Fの[実測](../reports/history-freshness-study-v0.md)は全3seed未fit・未知shape両側0。一般Corpus事前学習の必要性は未確定。次はイベント表現の対照を別登録する候補。F単独成功をAgent完了へ外挿する以前の学習課題が残る。
