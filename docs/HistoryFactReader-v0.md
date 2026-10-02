# 実Tool履歴→観測事実 reader v0：学習前登録

2026-10-02 JST。基準 PR #41 / 5f3cfde873792b85727dbfa8773c0c26ae28fae4。

## 考え方・仮説・達成目標
数値特徴→Action種類は成立したが、事実抽出は外部実装だった。次に文字列からC/M/Fを読む境界を測る。
まず実Tool生成・独立ラベル・保存再生・入力衝突・分割・長さが成立することを達成目標とする。
今回モデル初期化/学習/推論0。成功はreader能力の証明ではない。

## 入力と正解
入力はtask、現在memory、全履歴のAction/Observation（出力とok）のJSON文字列。
対象キー・関連イベントの選別、値の比較、時系列要約は入力側で行わない。
step、state_before/after、candidate、teacher、target、evaluation、goalフラグを含めない。
Runtimeがtaskとmemoryを構造化して供給する範囲を外部化し、依頼文→Taskは後続とする。
固定schema/JSON serializerとUTF-8 bytesを使用。候補ごとの入力ではなく履歴1件に対する読解。

C=現taskの成功calculator観測が存在する。
M=現在対象memoryが最後の該当calculator観測値に一致する。Cなしではunknown。
F=最後の対象putより後の成功getが存在し、その返値が観測値に一致する。
Cなし/対象getなしではunknown。unknownはfalseと別に保存し、既存分類器への接続時だけ0へ写す。
教師/評価器/未観測の算術期待解はreaderラベル生成に使用しない。
独立JSON readerをラベルoracleとし、既存observable_relationsとの一致とv1教師Action種類との対応を検証する。
oracleは非学習の検証器でありニューラルreaderの実測ではない。

## 固定データ・分割
8 train task（add/subtract各4）、4 dev task（各2）。各taskの全10prefixを同じsplitへ置く。
10family: normal0/1/2/3、誤上書き、同値再保存、無関係キー書込み、修復未取得、別task計算、未計算誤保存。
全prefixを実ToolExecutorで実行し、失敗観測は範囲外として拒否。
別task計算は現taskと異なるoperationの実観測で、予約済み冗長計算test familyには触れない。
train80/dev40。devは未見task identityのparameter-transfer診断でテンプレートは共有。
独立した構造family holdoutと呼ばない。既存dev/testは読まず、新封印testは今回生成しない。
元chapter/会話groupを用いる事前学習Corpusとの混入確認は後続。

## 検証・停止条件
入力署名の跨split重複0、同入力/異C-M-F衝突0、UTF-8可逆、最大bytes+2 <= 2048を要求。
切り詰め不可。全prefix再実行のTransition/state一致、oracleと既存抽出一致、教師種類一致、
v1ルール継続の停止時memoryと最終put後getを独立確認。全cases/digest/長さ/ラベル/契約をRunへ保存。
失敗ならpreflight失敗として停止し、測定前にprotocolを改版。改善を目指す後付け条件変更はしない。

## 次の学習（未実装）
まず同じ固定入力からC/M/Fを出す小型readerのtrain fit/正常保持/状態対照を測る。
正解事実→固定分類器と予測事実→固定分類器を比較。真値差し替えやrule fallbackの成功を算入しない。
未知/矛盾は明示abstain、精度/事実別precision-recall/誤停止/結合失敗/episode成功を別記録。
その後に同じarchitecture/Tokenizer/context/課題予算のrandom/pretrained比較を別登録。
一般Corpusの採用/split/所在確認が未完なので、今回言語事前学習は開始しない。
CPU速度確認後にepoch/LR/seed/時間上限を学習前登録する。200stepを完了根拠としない。

## 成果と責務
CLI生成/独立oracle/保存監査/正本reportをGitへ保存。モデル/Service/API/aster-web公開は後続。
ユーザー側の素材収集・WSL学習不要。私的日本語素材の採用が必要な時に具体的に依頼する。
Refs #24 / #21、観測#29→aster-web#2。PR #41依存のstacked Draft。
