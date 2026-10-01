# 固定モデルのv1状態診断：実測 v0

2026-10-01。事前計画: [DecisionFixedV1Diagnostics-v0](../docs/DecisionFixedV1Diagnostics-v0.md)。
結果の正本: [軽量JSON](decision-fixed-v1-diagnostics-v0.json)。6checkpoint更新0。

## 結論

PR #33の正常例置換モデルは既知正常8判断も5〜6/8へ低下し、baselineは全seed8/8。
同じ次Actionを要求する1回/2回の誤保存で、置換モデルの選択が24〜28/32組変わる。
誤値-999/-777による選択変更は置換モデル0〜1/32組、baseline0〜2/32組。
この2値の違いだけで悪化を説明する説は弱まり、正常経路の学習支持の欠落と
関連イベント反復/入力長への依存を次の対照として優先する。
回数と入力長・EOS位置が同時に変わるため、位置符号化だけの原因とは断定しない。

## 対象と実行条件

- PR #33のbaseline/state×seed42/43/44。Byte・compact・width16/heads2/layers1/context2048。
- source bundle: Aster_Decision_State_Training_v0_20261001.zip。元library IDとZIP SHA-256はJSONに保存。
- 推論source: local commit `dc05bb7`、全Python/protocol SHA-256はRun protocol.sourceに保存。
  後続のPython変更はpreflightの戻り値型注釈と独立cached audit scriptのみ。判断/入力/採点の変更なし。
- clean sourceで正式Run `e697aa10155e40c9a0c14eecb2953e19`、ChatGPT Linux CPU、Python3.12.14、torch2.14.1+cpu、2threads。
- 旧学習torch2.14.0+cpuとは版が異なるが、6×32最終train判断の選択とlogitsは完全一致（最大差0）。ユーザーWSL測定ではない。
- 128判断/model×6=768判断。各prefixから最大8行動の自力継続768本。学習更新・再校正・promotionなし。
- 旧v0学習に対する新v1診断。完了契約・候補供給も変わるため、旧reportの数値と単純な改善率を作らない。
- 元trainの2親groupからの開発probe。独立holdoutではない。旧/予約testは構築・点検・採点0。

## 全6モデル

|seed|arm|正常 /32|誤値×回数 /64|完了契約 /32|既知正常 /8|非重複主対照の両方正答 /24|
|---|---|---:|---:|---:|---:|---:|
|42|baseline|20|35|18|8|5|
|42|state|14|21|19|6|0|
|43|baseline|20|38|14|8|4|
|43|state|13|26|12|5|2|
|44|baseline|18|28|13|8|2|
|44|state|15|28|16|6|2|

非重複主対照は候補集合が同じで教師は修復put/getを選び分ける24組。
32組のうちtrain unionと入力同一のcaseを含む8組を保持確認へ分離した。
固定入力14/128がtrain unionと同一。全体precisionだけから状態読解を帰属しない。
このsuiteのin-sample候補集合oracle上限104/128はラベルを使った情報上限で、学習済み精度ではない。
各モデルのtrain-only候補集合lookup、全score/token IDs/予測・同候補対照を保存。

## 誤値と反復の分離

|seed|arm|誤値を変えて選択が変わる /32組|1→2回で選択が変わる /32組|計算前・1回誤保存 /16|計算前・2回誤保存 /16|
|---|---|---:|---:|---:|---:|
|42|baseline|2|15|8|8|
|42|state|1|25|6|2|
|43|baseline|1|9|10|12|
|43|state|1|24|12|4|
|44|baseline|0|20|8|6|
|44|state|0|28|12|2|

この対照ではtask・計算前後・現在memory・候補集合・正しい次Actionを保ち、
関連putイベントの反復回数を変える。compactにstep番号は含まれない。
同じ数字だけを学び直す案より、正常経路を残した比較と反復への頑健性を測る案が妥当。
値依存が一般にないという証明ではなく、この2値/8taskでの診断結果である。

## 自力継続と新仕様

|seed|arm|正常prefix成功 /32|誤保存prefix成功 /64|完了prefix成功 /32|完了prefix invalid /32|
|---|---|---:|---:|---:|---:|
|42|baseline|20|26|19|2|
|42|state|13|23|21|2|
|43|baseline|22|25|23|2|
|43|state|12|18|16|1|
|44|baseline|18|25|23|4|
|44|state|16|21|20|4|

invalidはmodel自身の反復でcompactイベントが増えcontextを超えた経路。
成功件数とinvalid件数を分けて示し、invalidを通常失敗率へ合算しない。
教師Action完全一致とEvaluatorの停止時成功は別。理由がpolicy_stopの停止でも
v1終状態が正しければ成功するため、教師不一致と能力失敗は一対一ではない。

初期状態からのseen成功: baseline全seed2/2、state1/2・0/2・1/2。
数値変更は全6model0/2。キー変更baselineは1/2・1/2・0/2、state全0/2。
同時変更は全6model0/2。反復prefix継続の結果を初期からの完了に加算しない。

同値再保存後の再getや達成後の誤上書きは旧学習契約との差を含む。
この部分の失敗だけでモデル容量不足を断定しない。各mutationの判断はJSONに全件保存。

## 検証と実行履歴

- 実Toolで作った128prefixを再生し、state/observation/evaluation/candidates/teacher一致。
  rule128/128完了、候補coverage128/128、入力正解衝突0、固定入力超過0。
- 6モデル全caseの候補逆順と別reloadのscore一致。weights/RNG/元checkpointファイル不変。
- 独立cached audit: prefix768予測、対照192組、自力訪問3,247予測、768episode、5,119transitionを照合。
  保存scoreからargmax/Action/marginを再計算、UTF-8 byte列を照合し、保存memoryとget/put順から停止時成功を別計算した。
- 最初の配線Run `1916d622aebb4683af45c67ed843d984`は道具失敗後の候補1個のmarginで打切り。
  競合なしmargin=Noneへ修正し回帰検査。失敗Runも保存し、全6modelは同じ事前条件で最初から測定。
- 正式測定wall約56秒（preflight・reload・全診断を含む）。各model30分deadline内。
- 検証: 全pytest159 passed、Pyright0 errors/0 warnings。numPyなしのtorch警告は推論を妨げず、旧cached logitsの一致を確認した。

## 次に選ぶ一段

**正常例を保持する混合学習対照**を、v1教師へ揃えた別protocolとして設計する。
学習データの支持範囲を確かめる実験で、広いAgent能力を保証しない。
案: baseline元32判断を2回提示する64slot、mixedは正常32+誤保存16+正常get/stop反復16の64slot。
双方calculator/put/get/stopを各16slot、25epoch=1,600更新へ揃える。
混合側のunique48判断と元32判断、個別exposure差、token費用を明記する。
同seed全初期weight・提示slot permutation・Tokenizer・compactを固定。条件は実装前に正式登録する。
今回の新dev全caseをtrainへ流さず、旧train task上の1回誤保存だけを混合候補に使う。
2回反復と完了取消の対照は引き続き診断として残す。
正常保持だけ回復し反復判断が崩れるなら、その次に関連事実の表現比較へ進む。
混合で既知fitが成立しなければ汎化議論よりfit診断へ戻る。再学習は今回未実施。

main/Service/Web/公開recipe/採用モデルには未導入。PR #35をbaseとする後続Draftで提出。
ユーザー側の追加作業は不要。保存ログ閲覧は#29→aster-web#2の後続。
