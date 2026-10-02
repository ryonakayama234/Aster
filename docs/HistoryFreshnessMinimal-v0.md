# 取得確認Fだけの最小読解 v0

2026-10-02 JST。PR #45 / 751d42aを基準とする新protocol。

## 考え方・仮説・反証

順序追加は既知対照に16/15/0組fitし、未学習shapeの転用は弱かった。
raw JSON、対象対応、値一致、C/M/F同時学習が重なっている。
今回は値の誤りを除いた小課題で、Fだけの教師が安定fit/転用を助けるかを問う。
raw/compact変更、事前学習、容量変更、訪問訂正を同時に入れない。
Fだけでも未fitなら「多目的学習だけが原因」を支持しない。
fitするが転用しないなら順序関係の汎化が残る。成功しても一般Agent能力と呼ばない。

## 固定条件と変更点

- 実Toolで生成する既存plain/tail_put順序対照をtrain8taskへ各両側、計32件。
  calculator観測成功・現在保存値一致を固定する。両側は同memory/同event multiset/同長。
- unknown保持: 同8taskのnormal1/normal2（計算後/保存後で対象getなし）16件を追加。
  F=NoneをFalseと同一視しない。normal1のM=Falseは同時教師の対照に残る。
- train48、既知taskの未学習middle_put/tail_get/repeat_get48、
  dev4taskの同shape24＋unknown8=32。合計128unique/12task/52順序組。
  既存taskとshapeは観察済み。独立testではなく学習予算を固定した診断。
  旧/予約testの生成・読込・採点0。shape_trainはtrainのtaskを共有する。
- 入力は既存task/memory/全Action・Observationのraw JSON。関連イベントを選別しない。
  値は実Toolの正しい観測から供給する。学習器の算術/equality能力は測定しない。
- random初期化、既存HistoryReader（Byte258、width16、2head、1layer、context2048、
  学習可能な絶対位置embedding、末尾読出し、9logit head）を両armで共通にする。
  同seedの全初期tensor・全slot順を一致、初期重みを保存。
  f_onlyはFの3logitだけcross entropy、cmfは既存3事実lossの平均。
  Fのloss係数は1対1/3になる。これは教師目的変更の一部であり、損失scale/勾配干渉を分離する実験ではない。
  C/Mの未学習出力はf_onlyの評価対象にしない。
- seed42/43/44、50epoch×48=2400更新/arm、AdamW lr0.001/default、clip1.0、CPU2threads。
  同入力/同順序/同token量/同parameter数。前のcheckpoint介入とは初期化・データ・予算が異なり改善率を作らない。
- 評価epoch0/10/20/30/40/48/49/50。最終50で採点しbest選択なし。
  stable fitは最後3評価で48/48かつ全F target margin>0。全学習/最終予測と順序を保存。
  1epoch debugは正式結果へ加算しない。正式seed毎arm900秒上限、非有限/配線不一致で停止。

## 生成検証・測定・監査

実Tool再生、独立JSON oracle/既存relations照合、未知と偽、入力ラベル衝突0、
128unique、task単位split、train/eval入力非重複、全入力context内、対照multiset/長さ/共通suffixを確認。
preflightとprotocol digestを正式開始前に固定。clean source SHA/tree/Python hashesを保存。
各seedのF精度/NLL/min margin/3class confusion、対照両側、cohort/shape別を報告。
majorityと最終toolだけのtrain lookupを対照とする。正答をinputへ混入しない。
別processでcheckpoint reload/全最終logit再推論、独立算式でNLL/margin/混同行列とpairを再採点。
保存curveの全F予測も独立再採点。入力trajectoryを全件再実行し、生成/モデル能力を分離。
評価中weights/RNG不変と測定後Python hash不変を確認。
自力episode、固定Action分類器、元80例保持は今回測定しない。事実をoracleで補ってAgent成功を作らない。

## 結果別分岐・全体接続

両arm未fit: 読出し/位置/最適化を一軸診断。f_onlyのみ安定fit: 教師目的の切り分けを継続。
fit成立/shape失敗: rawとイベント表現の比較を別登録し、外部化した構文・対象同定を明示。
shape成立/task失敗: キー/数値対応を独立診断。全成立: 値不一致読解を次課題にする。
条件を結果後に同実験へ変更しない。Linux CPU測定とユーザーWSLを区別する。
研究CLIのみ。main/Service/API/aster-web/公開モデル/promotionは変更しない。
観測#29→aster-web#2ではF単独scoreと3事実/Agent完了を分ける。Refs #24/#21。
ユーザー追加作業不要。
