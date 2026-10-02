# 取得確認Fだけの最小読解 v0: 正式結果

2026-10-02 JST。PR #45依存。条件は[事前登録](../docs/HistoryFreshnessMinimal-v0.md)、
生成正本は[preflight](history-freshness-preflight-v0.json)、全条件は[JSON](history-freshness-study-v0.json)。

## 結論

F単独教師でも全3seed安定fitしなかった。最終train32/33/32（/48）、
順序対照両側0/1/0（/16）、既知未学習shapeとdev未学習shapeの両側は全seed0。
同raw入力・全初期tensor・提示順・更新量・token量のCMF対照も全seed未fit。
この条件で「3事実を同時に教えるのをやめれば解決する」という仮説は支持されない。
多目的学習が無関係、容量不足、Transformerでは不可能、事前学習が必須とは確定しない。

未知クラスをFalseへ潰さず、対象getなし16train/8devは全arm全seed正答。
一方、同じmemory/event multisetを持つ順序の区別はほぼ成立しなかった。
学習適合そのものが未成立なので、未知履歴への汎化だけに失敗を帰属しない。
モデル採用なし。一般Corpus学習、Task reader、Action選択、自力episodeは今回測定していない。

## 全6条件

|seed|教師|train F /48|train順序両側 /16|既知未学習shape両側 /24|dev shape両側 /12|取得なしdev /8|stable fit|
|---|---|---:|---:|---:|---:|---:|---|
|42|CMF|32|0|0|0|8|不成立|
|42|Fのみ|32|0|0|0|8|不成立|
|43|CMF|32|0|0|0|8|不成立|
|43|Fのみ|33|1|0|0|8|不成立|
|44|CMF|32|0|0|0|8|不成立|
|44|Fのみ|32|0|0|0|8|不成立|

F-only既知未学習shapeの単例正答23/24/24（各48）、dev shape全seed12/24。
32/48はunknown16＋順序の片側16でも到達する。単例正答と両側正答を分離する。
F-only seed43の最終33/48はepoch48/49の32/48から1増えただけで安定fitではない。
未学習shapeのNLLは約0.694（F-only seed42の既知shapeのみ約0.762）。
二値等確率のlog(2)=0.693147付近だが、3class教師・小標本であり統計的な無能力証明ではない。
CMFは全順序組で同じclassを両側へ出した。F-onlyはtrain両側同class16/15/16（/16）、
既知shape23/24/24（/24）、dev全seed12/12。異なる予測だけを正答とは数えない。

train-derived majorityはtrain16/48・順序両側0/16。
最終toolだけのtrain lookupはtrain32/48・順序両側8/16、
既知未学習shape16/24、dev未学習shape8/12。
このlookupは末尾toolが答えを示すshapeで成功するため、関係規則獲得の証拠ではない。
共通suffixのtail_put/tail_get対照を省略しない。モデルはこの単純対照も安定して上回らない。

## 条件と能力の境界

8train task/4dev task、実Tool生成128unique、52順序対照。
train48=plain/tail_putの両側32＋normal1/normal2のunknown16。
既知未学習middle_put/tail_get/repeat_get48、dev同shape24＋unknown8。
値一致は実Toolで保証し、raw入力は既存task/memory/全Action・Observationを保持。
JSON構文・対象キー対応・順序・位置の問題は依然含む。Fの教師だけに縮小した診断で、
全ての読解要因を除去した最小オートマトン実験ではない。
既存taskとshapeは研究者が観察済み、独立testではない。旧/予約test生成・採点0。

random initializationの既存Byte258/width16/2head/1layer/context2048/末尾読出し。
同seedで全初期重み・各epoch shuffle順を一致。CMF平均loss対Fだけのlossなので、
Fのloss係数1/3対1も目的変更の一部。勾配干渉と損失scaleはこの比較だけで分離できない。
元80例のcheckpoint介入ではなく、前実験との改善率を作らない。元80例保持は未測定。

50epoch×48=2400更新/arm、合計14400更新、1arm当たり1,695,900token。
全parameter44,489、trainable40,361。AdamW lr0.001/default、clip1.0、CPU2threads。
評価0/10/20/30/40/48/49/50、最終50を採点、best checkpoint選択なし。
正式wall89.892秒、Linux/Python3.12.14/torch2.14.1+cpu。
Pythonの正確な版はJSONのpython fieldを参照。ユーザーRyzen/WSL実測ではない。

## 保存・検証

正式Run `3d187a759c3246a8aa4b4ec828b80e7a`。
clean local source `c027ca626deb56b38ab94b13730456112fa222f5`、
source tree `f003b78c62da8e038a33009981934a85039010e2`。
GitHub同一tree source commit `f409004e59f78941d2067c2b64303de6ab3c1a9c`。
protocol SHA/全測定Python SHAを保存し、測定終了時も一致。条件後付け変更なし。

全pytest197 passed、Pyright0 errors/0 warnings、compile/diff check。
全128入力trajectoryを実Tool再生、独立JSON oracle/既存relations照合、
unknown/false、対照multiset/長さ/shared suffix、input衝突0/train-eval重複0を確認。
768最終予測を別processでcheckpoint reloadし全logit完全一致。
2304学習保存予測を独立softmax式NLL/margin/混同行列で再採点し、全curve/両側集計を照合。
初期重み/optimizer/提示順/全予測とcurve/入力trajectory/保存重みをRunへ保持。
評価weights/RNG不変、同seedの初期weight/order/更新/token量一致。
1epoch debug Run `9d0565277c1d4a959ce5aa4845ff89ee` は正式結果へ加算しない。
debug cached logitへ+0.25の改変を入れ監査拒否を確認し、元ファイルを復元。

## 次の判断

F単独でもfit未成立。次はraw JSONと、構文だけを除去してキー・値・順序・全イベントを
残すイベント表現の同条件比較を設計する候補。対象イベントだけを外部選別した表現は
対象対応も外部化するので別arm/別能力と明記する。読出し/位置表現も競合仮説として残す。
未fit段階で学習予算・容量・事前学習・訂正データを同時変更しない。
本変更は研究CLIのみ。main/Service/API/aster-web/公開モデル/promotion変更なし。
ユーザー追加作業不要。観測#29→aster-web#2へ将来渡す際はF単独/3事実/Agent完了を分離。
