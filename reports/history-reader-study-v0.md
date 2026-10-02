# 履歴→事実 reader：固定3seed学習・実行診断 v0

2026-10-02 JST。[事前条件](../docs/HistoryFactReaderTraining-v0.md)、[正本JSON](history-reader-study-v0.json)。

## 結論
全3seedでteacher-prefixのtrain80件を安定fitし、既知8taskの通常初期からの完了を保持した。
一方、dev事実同時正答は21/24/20（40件）、devprefix完了は6/12/3、初期4task完了は0/1/0。
既知taskの途中状態からでも50/48/41（80件）しか完了せず、見たprefixのfitと
その後に自力で訪れる未学習履歴への対応が分かれた。
**課題用学習の成立は確認。関係読解と自力復帰の汎化は未成立。モデル採用しない。**

|seed|安定fit|train事実同時|dev事実同時|dev対照両方|trainprefix完了|devprefix完了|既知初期|dev初期|
|---|---|---|---|---|---|---|---|---|
|42|True|80/80|21/40|4/16|50/80|6/40|8/8|0/4|
|43|True|80/80|24/40|4/16|48/80|12/40|8/8|1/4|
|44|True|80/80|20/40|5/16|41/80|3/40|8/8|0/4|

初期はprefixの部分集合で、件数に加算しない。対照は16組/devで同一4task派生。
seed数も独立task数へ加算しない。train80/dev40は8/4task、共有10templateのparameter transfer。
構造的holdoutではない。oracle事実→保存済み分類器→候補結合は全120prefix完了。
reader成功と外部candidate引数供給、固定分類器の成功を区別する。

## どこで読解が崩れたか
|事実別正答 /40|seed42|seed43|seed44|
|---|---:|---:|---:|
|C：該当計算を観測済み|34|38|35|
|M：現在memoryが観測値と一致|29|31|29|
|F：最終put後に正しくget|31|32|31|
|F=trueの再現 /8|0|3|1|

F全体の正答はunknown/falseで上がり、取得確認済みを拾う能力とは異なる。
現在memory不一致のM=false 8件は全seedで1/8正答。dev誤上書きfamilyは全seed0/4事実同時正答。
Cだけでなく、対象の対応付け・値の一致・取得順の組合せが弱点として残る。
どの入力手掛かりを使ったか、容量不足/キー暗記/位置/長さのどれが原因かは未確定。

保存ログからの測定後診断では、最初の誤読が発生した入力は全seed・両splitともtrain入力非重複。
train開始prefixでも行動後に新しい履歴へ出て崩れる。複数事実が同時に誤るため事実別件数は重複する。
一度誤読しても回復して完了するケースがあり、最初の誤読を必ず最終失敗とは扱わない。

## 失敗と範囲外の扱い
|dev episode /40|seed42|seed43|seed44|
|---|---:|---:|---:|
|完了|6|12|3|
|abstain|1|2|2|
|candidate結合失敗|2|0|2|
|範囲外Tool失敗invalid|2|6|4|
|誤stopを含むepisode|2|10|7|
|誤putを含むepisode|11|5|1|

各行は排他的分類ではない。上限内に完了しない経路もある。
invalidはモデルの提案から実Tool失敗に到達した結果で、評価対象外件数を隠さない。
成功6/12/3は全40開始件数を分母に表示し、invalid除外だけで成功率を高く見せない。
trainではinvalid4/7/6。このうちseed43/44各1がreader context超過、他は実Tool失敗。
全failed Observationを実際に記録し、失敗Tool結果を生成で補わない。rule fallbackなし。
unknownは未観測というラベル、abstainは矛盾した予測組の拒否。合法unknownは固定feature契約で0へ写す。
未校正確率をconfidenceやpromotionの根拠としない。

## 条件・保存・検証
Byte258/context2048、TinyLM16/heads2/layers1、9logit head（3事実×3class）、総44,489/学習40,361parameter。
AdamW lr0.001/default、clip1.0、50epoch×80=4,000更新/seed、計12,000。毎epochtrain80。
固定分類器はPR #41正式Run212b32fd…/seed42保存weight。model de12ec2…を全seedで固定・更新0。
一般Corpusでの事前学習なし、自由会話/Task生成/引数生成/算術の学習実験ではない。
多数クラス対照はtrain8/80・dev4/40同時正答。各事実を独立に最頻classへする非学習baseline。

正式Run `1cc57fcd7b6e41819bdd95d35c272003`、clean local source
`3584053ac50bd20c16e27b6c43b4c627177d5e0c`。source tree/Python/protocol/suite hashは正本JSON。
Linux CPU/Python3.12/torch2.14.1+cpu/2threads、正式89.799秒。ユーザーRyzen/WSL測定ではない。
全pytest184 passed（64.85秒）、新reader4test、Pyright0 errors/0 warnings、diff/compile成功。
正式測定と全pytestは併走させていない。測定後の全既存Python hash変更0。

独立cached audit：train12,240/final360/episode360/transition2,543/訪問1,656を照合。
全oracle120経路も実Tool再実行。NLL/marginは保存logitsから別算式で照合し、
入力/真値/提案/結合/teacher/実transition/停止時memory・順序を再確認。
reload全score一致、reader/classifier weightsとRNG不変を確認。
全initial/final weight、shuffle、curve/score、case、trajectoryを実測bundleへ保存。
optimizerなし・学習再開非対応。debug1epochは正式結果へ加算せず別Runとして保持。
研究runnerでは旧/予約testを生成・採点しない。全pytestの仕様テストとは別範囲。

## 次の問いとAster全体への接続
次は保存3readerを更新0で診断し、(1)未知taskの値/対象対応、(2)既知taskでも復帰後に出る
新しいget履歴の読解を分ける。関係を保つキー一貫改名、値一致/不一致、
同じイベント数でget/put順だけが違う到達可能prefixを別protocolへ事前登録する。
原因を絞ってから、trainのみの自力訪問履歴への訂正を採る比較や事前学習比較を設計する。
既存devをそのままtrainへ移して改善を主張しない。候補は育成#20、言語#21、親研究#24。

main/Service/API/aster-web/公開モデル/promotionは変更しない。PR #42依存のstacked Draft。
観測#29→aster-web#2へ将来渡せる証拠は保存したが、Web表示の実装は今回未実施。
ユーザー側追加作業不要。改善しなかった部分を含め、研究の次の境界が明確になった。
