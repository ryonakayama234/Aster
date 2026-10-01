# v1正常例保持・混合学習：全6条件実測 v0

2026-10-01。[事前条件](../docs/DecisionMixedMeasurement-v0.md)、[正本JSON](decision-mixed-v1-study-v0.json)。

## 結論

両条件全3seedでtrain stable fitが成立し、共通正常32判断と既知2初期episodeを保持した。
ただし主対照24組の両方正答は対照4/4/4、混合4/2/2で一貫した改善なし。
非重複114判断は対照55/51/44、混合51/52/51。prefix継続はseed43/44で改善しseed42で悪化。
混合seed44だけ数値変更の初期完了1/2、他5armは0/2。キー変更は対照seed44だけ1/2で混合全0/2。
単一seedの改善だけで採用しない。**学習と正常保持の両立は成立、状態関係の汎化は未成立**と評価する。
「復帰例を足せば必ず改善」「容量不足」とは結論しない。

## 条件と全結果

- normal-repeat-v1：正常32uniqueを2回で64slot。
- normal-plus-recovery-v1：正常32+復帰16+正常get/stop再提示16の64slot、unique48。
- Byte/compact、width16/heads2/layers1/context2048、LR0.001、AdamW既存default、CPU2threads、25epoch=1,600単例更新/arm。
- seed内全初期weightと64slot index permutationをhash照合。同じindex順でも内容は異なる。
- v1教師・候補・Tool実行・replay・Evaluatorを一致。calibrationなしT=1。学習契約をcheckpoint別sidecarで検査。
- 各Action400提示だが正常calculator/put個別提示は対照50/混合25。固定総予算の配分介入。
- epoch0〜25の全unique予測6,240件を保存。最終3epoch全正答/正target marginでfitを判定。

|seed|arm|安定fit|共通正常 /32|非重複判断 /114|主対照両方 /24|既知初期完了 /2|数値初期完了 /2|キー初期完了 /2|
|---|---|---|---:|---:|---:|---:|---:|---:|
|42|normal-repeat-v1|True|32|55|4|2|0|0|
|42|normal-plus-recovery-v1|True|32|51|4|2|0|0|
|43|normal-repeat-v1|True|32|51|4|2|0|0|
|43|normal-plus-recovery-v1|True|32|52|2|2|0|0|
|44|normal-repeat-v1|True|32|44|4|2|0|1|
|44|normal-plus-recovery-v1|True|32|51|2|2|1|0|

主対照は同じ候補集合で異targetを要求し、train unionと非重複の24組。
14/128の重複判断は保持確認として別に保存。24組はcaseを共有し独立24taskではない。
同時変更の初期完了は全arm0/2。正常保持32/32はtrain内で、未知正常dev32と区別する。
教師Action一致と停止時task_successは別。理由が異なるstopも、終状態が正しければ成功になる。

## 自力継続とinvalid

全128prefixから最大8行動の自力継続を保存。context超過はinvalidで通常失敗へ合算しない。
両arm共通の有効開始例に限定した比較：

|seed|共通有効 /全128開始|対照成功|混合成功|混合−対照|
|---|---:|---:|---:|---:|
|42|117|57|49|-8|
|43|119|51|58|7|
|44|121|38|67|29|

各arm全開始でのnormal/factorial/completion成功・invalid件数、その他sliceはJSONに全て保存。
これは途中prefixからの復帰であり、初期状態からの完了に加算しない。
完了取消・同値再保存は今回trainに含まれず、そこだけの失敗で表現/容量の不足を断定しない。
候補集合のみtrain頻度lookup（未知集合abstain）は別対照。候補のみoracleは学習済み精度ではない。

## 検証と実行資源

- 全pytest165 passed、Pyright0 errors/0 warnings、diff check成功。
- 登録済みpreflightのsuite/input/target/candidate/pair/slot/Tokenizer/serializer digestを学習開始前に一致確認。
- 全6armのtrain最終予測とreload完全一致。dev固定128scoreを別reloadで完全一致。
- 候補逆順はAction一致、score許容差1e-5で検査し実際の最大差を全arm保存。
- 評価前後weights/RNG一致。測定後の全Python source hashも一致。
- 別cached auditで6,240trainのaccuracy/NLL/margin、768dev予測、3,569訪問予測、768episode、5,441transitionを照合。
  入力UTF-8 byte列/argmax/教師/候補を確認し、保存Actionを実Toolで再実行。停止時値と最後put後get順を別計算。
- 保存slice/初期episode表を別集計し全6arm一致。Wolframで主対照差0/-2/-2、共通prefix差-8/7/29、非重複判断差-4/1/7、9,600更新/6,240train評価行を照合。
- 正式Run `819891926f88471d80d5448265232c37`、clean local source `55ca4a563487d852cca278c48e622d6aceee8806`、dirty=false。
- Python 3.12.14 / torch 2.14.1+cpu / ChatGPT Linux CPU / 2threads。ユーザーWSL測定ではない。
- 全正式wall 271.082秒。全arm30分deadline内。parameter数はJSON。process maxRSSはarm専用memory比較ではない。

|seed|arm|更新秒|unique評価秒|全学習+評価秒|token数|padding込みtoken数|
|---|---|---:|---:|---:|---:|---:|
|42|normal-repeat-v1|28.163|4.336|45.149|3559200|3657900|
|42|normal-plus-recovery-v1|28.582|5.757|45.238|3789600|3888300|
|43|normal-repeat-v1|27.227|4.292|41.519|3559200|3657900|
|43|normal-plus-recovery-v1|28.960|5.877|45.383|3789600|3888300|
|44|normal-repeat-v1|28.253|4.360|41.807|3559200|3657900|
|44|normal-plus-recovery-v1|31.929|6.286|50.266|3789600|3888300|

## debugと保存

1epochの最初のdebug Run51bc713a610e46939501fe2d79e97c6dはfloat score完全一致チェックで停止。
選択Actionは同じで約6e-8の丸め差だった。正式測定前に既存診断同様の1e-5許容へ修正、Action一致は必須のまま。
次のdebug Run8bd871568f9f4f9bb737b8f2c1e41e71は両armcompleted。いずれも正式結果へ加算しない。
修正理由と失敗記録を保持し、修正後clean sourceから全6armを一度測定した。
checkpoint/全Runログは別ZIP、Gitには軽量summary・hash・全seed結果を保存。旧modelを上書きしない。

## 解釈と次の一段

学習内容を増やしても、同候補集合の未知状態の選び分けは安定改善しなかった。
一方、既知正常例を損なわず48判断をfitできるので、この条件でtrain未fitだけを原因とする説明は弱まる。
途中prefixの改善は限定された成果として保持する。未知数値の全般対応が成立したとは扱わない。
次の優先案は、同じ混合train/予算のまま、compactと**観測事実の関係を明示する表現**を別条件で比較すること。
必要情報を読む難しさと、情報から行動を選ぶ難しさを区別する。未実装・未測定。

一次研究の視点：Battaglia et al. (2018), https://arxiv.org/abs/1806.01261 は対象/関係の構造化を設計視点として論じる。
Lake & Baroni (2018), https://proceedings.mlr.press/v80/lake18a.html は近い例への成功と組合せ汎化を区別する。
いずれもAsterの失敗原因の証明ではなく、次の診断対照を選ぶ根拠。

2親groupの派生dev probeで独立holdoutではない。seed数や768episodeを独立課題数へ加算しない。
旧/予約test構築・点検・採点0、追加調整・自動promotionなし。main/Service/aster-web/UI/公開モデルは未変更。
PR #38をbaseとするstacked Draft。今回ユーザー側追加作業不要。
