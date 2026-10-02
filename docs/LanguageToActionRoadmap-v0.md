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
