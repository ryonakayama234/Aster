# 固定履歴reader：名前・値一致・取得順の診断結果 v0

2026-10-02 JST。[登録protocol](../docs/HistoryReaderFrozenContrasts-v0.md)、
[機械可読正本](history-reader-contrasts-study-v0.json)。

## 結論

**既知taskでの対象キー一貫改名は保持した。一方、値の一致と最終書込み後の取得順を
選び分ける対照は、既知taskでも全3モデルが両側正答0/4だった。**
未知キーの名前だけでは失敗を説明できない。既知の名前・数値の課題でも、
新しい誤値/イベント順への関係読解が成立しない反例を確認した。
ただし内部の手掛かり、容量、最適化、必要な学習量の唯一原因は確定しない。
モデルの更新・追加学習・採用は行わない。

## 固定したものと分母

PR #43正式Run1cc57fcd7b6e41819bdd95d35c272003の保存reader seed42/43/44と同じ固定行動分類器。
Byte258、width16/heads2/layers1/context2048、旧input/goal/class契約を維持。校正/学習0、fallbackなし。
元train8taskからindex0/1/4/5と元dev全4taskを使用。各taskで3対照を生成。
24組/48参照slot/40unique入力/8task。各split20unique、旧train重複8、前回dev重複8。
各axisのtrain側は4組、dev側4組。同一taskからの派生を独立課題に数えない。
既存devを研究者が観察済みなので独立holdoutと呼ばない。旧/予約test生成・採点0。
今回と前回のdev suiteの内容と分母は異なり、6/20と前回6/40を改善率として比較しない。

各対照の両側はbyte長/イベント数が同じ。最大832tokensで切り詰めなし。
順序対照は同じtask/current memory/Action・Observation multiset。
一致対照は前三イベントとtaskが同一、最後のputと現在memoryの値だけを同符号・同長の誤値へ変更。
名前対照は全出現の一貫改名で事実不変。全40prefixを実Tool生成・再生し、
期待ラベル/独立JSON oracle/既存関係scan/v1教師を照合した。oracle継続40/40完了。

## 対照の両側を正しく読めた組数

|axisとtask由来|seed42|seed43|seed44|
|---|---:|---:|---:|
|名前：既知task /4|4|4|4|
|名前：既存dev task /4|0|1|0|
|値一致：既知task /4|0|0|0|
|値一致：既存dev task /4|0|0|0|
|取得順：既知task /4|0|0|0|
|取得順：既存dev task /4|0|0|0|

今回の各組では両側の事実正答とAction正答の件数が同じ。一般に同じ指標とは限らない。
名前の既知側は正常取得済みの原入力と、同長の別名への変更。全3seedで両側正答。
devの名前では予測不変は4/3/4組だが、両側正答は0/1/0。間違ったまま不変を成功と呼ばない。
value/orderでは全split・全seedで4/4組の予測が同じ。正解はMまたはFが変わるため、
両側を正しく区別できなかった。
旧trainと両側とも入力非重複なのはdev各axis4組。既知側は原入力を片側に含む保持＋転用診断。

## 具体的な反例

train/task0はadd(2,3)→total。観測された計算値は5。

- 計算→put(5)→get→put(5)：M=true/F=false。次はget。
- 計算→put(5)→get→put(6)：M=false/F=false。次はputで修復。
- 計算→put(5)→put(5)→get：M=true/F=true。次はstop。

全3モデルは3入力すべてC=true/M=true/F=falseと読み、同じgetを提案した。
2番目では誤値を一致と読み、3番目では最後の書込み後の確認を取りこぼす。
元trainの誤上書き-999はfitしたが、今回の同長誤値は全seed・全8taskでM=trueと誤読。
固定の誤値を覚えた可能性は残る。ただしこの結果だけでその内部原因を証明しない。

名前のtrain成功は特定の改名/正常経路4taskに限る。全未知名/多キー/日本語/任意schemaへの能力ではない。
順序対照の同長失敗は、入力長変化だけで今回の対照失敗を説明する説を弱める。
位置表現への依存や構造読解の不足のどちらかをこの実験だけで選ぶことはできない。

## 個別読解と自力継続

|指標|seed42|seed43|seed44|
|---|---:|---:|---:|
|既知由来の事実完全一致 /20|12|12|12|
|既存dev由来の事実完全一致 /20|4|7|4|
|既知由来prefix自力完了 /20|10|11|8|
|既存dev由来prefix自力完了 /20|6|7|2|
|dev誤stopを含むepisode /20|2|2|0|
|dev誤putを含むepisode /20|2|0|0|

今回の全120自力episodeでinvalid/abstain/結合失敗0。
値・順序の初手誤読は既知taskでも発生しているため、自力訪問分布の変化だけでは説明できない。
初手正答後に新しい履歴で崩れる例もあり、正しい初手を完了能力と同一視しない。
全3seedで8episodeずつ、最初の誤読は初手の次に発生。初手誤読は24/21/24episode。
反復getでstep上限に達する経路なども全trajectoryへ保存。seed数を独立taskへ加算しない。

## 保存・数値再現・監査

正式Run b631e0d1dee643ac85abd66f893ec99a。
clean source c47af593195f0f8cebfa5615a6e4313dec2f22e0、source tree/Python/protocol hashは正本JSON。
正式3.198秒（学習なし。精密な速度benchmarkではない）。
Linux x86_64 / Python3.12.14 / PyTorch2.14.1+cpu / CPU2threads。ユーザーRyzen/WSL未測定。
全weights/RNG不変、元checkpoint files不変、測定後Python hash変更0。
同一processのreloadは全40入力×3seedでscore完全一致。

元環境との120入力×3seedのlogitはbitwise一致0/120ずつ。
最大絶対差1.9073486328125e-6 / 1.430511474609375e-6 / 1.9073486328125e-6、全事実class一致。
新対照を採点する前に絶対許容1e-5/相対0・class変更0へprotocolを改版した。
許容内一致と完全一致を区別し、元score/復元score/差を保存している。

全pytest190 passed（68.95秒）、新規6test、Pyright0 errors/0 warnings、compile/diff check成功。
正式測定中にpytestを併走しない。
独立監査：元score照合360、診断予測120、対照72、episode120、transition1114、訪問682、oracle40。
保存logitsを別算式で再採点。分類器はPython scalar計算で種類を照合。
全記録Actionを実Tool再実行し、停止時memoryと最後対象put後getを別走査で確認した。
生成/配線/仕様test成功と学習モデルの読解能力を区別する。

## 次に採用したい問い（設計候補、未実装）

**順序を変えた対照を学習に含めると、元の正常経路を保ちながら取得順を読めるか？**
値一致と順序を一度に追加せず、まず同じmemory/イベントmultisetで正解が変わる順序を一軸にする。
元80trainを残し、同じ親checkpointから更新なし/元例の追加反復/元例＋train taskの順序対照を比較。
学習予算・提示回数・class頻度・optimizer再初期化などは正式実装前に別protocolを固定する。
既存dev対照をそのままtrainへ移さない。追加生成はtrain taskだけとし、旧testは封印。
trainで両側をfitできなければ最適化/表現/容量へ、fitしても転用できなければ関係の汎化へ問いを戻す。
自力訪問履歴への訂正(#20)やrandom/pretrained比較(#21)は、その結果から別々に決める。

PR #43依存のstacked Draft。main/Service/API/aster-web/公開モデル/promotionは変更なし。
正本・全weights・全実行証拠を再現bundleに保持。Web観測#29→aster-web#2への公開は未実装。
今回ユーザー追加作業不要。
