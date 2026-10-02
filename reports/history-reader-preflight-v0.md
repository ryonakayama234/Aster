# 履歴→事実 reader 学習前検証 v0

2026-10-02 JST。[事前登録](../docs/HistoryFactReader-v0.md)、[正本JSON](history-reader-preflight-v0.json)。

## 結果
実Toolでtrain80/dev40を生成し、保存後に全120prefixを再実行した。候補/教師/評価フラグを入力から除外。
独立JSON oracleと既存observable_relations、およびv1教師のAction種類が全120件一致。
全120prefixからルールで継続し、停止時memoryと最終対象put後getを独立確認。

|確認|結果|
|---|---:|
|train / dev task数|8 / 4|
|train / dev prefix数|80 / 40|
|共有生成family|10|
|状態対照（異事実/同事実各2種類×12task）|48組成立|
|入力/ラベル衝突|0|
|跨split入力重複|0|
|最大Byte token（BOS/EOS分2含む）|1,005 / 2,048|
|保存再監査|一致|
|モデル学習 / 推論|0 / 0|
|旧・予約test生成 / 採点|0 / 0|

C/M/Fのunknownを残すとtrainに6種類のラベル組がある。既存分類器への0写像後は4パターン。
unknownの事実ラベルと行動分類器の数値特徴を混同しない。別task計算はC=false、
同値再保存はM=true/F=false、無関係キーへの書込みはF=trueを維持する。

## 意味と限界
成立したのは読み取り課題の生成/教師/保存/評価配線で、学習済みreaderの能力ではない。
非学習JSON oracleは正解を作る検証器。Task/current memoryの構造化供給はRuntimeの責務。
reader自身が任意依頼文を読んだ、算術を解いた、道具を選べたとは扱わない。
devはtask identityを丸ごと分けた数値/キー移送診断で、trainと生成templateを共有する。
独立構造familyのholdoutではない。データ追加/学習条件選択に使えば設計用devである。
Tool失敗は範囲外。予約済み冗長calculator prefix familyは生成せず、既存testを読まない。

## 検証と出典
新規5testを含む関連pytest34 passed（0.67秒）。本モジュールのPyright0 errors/0 warnings。
compileall/diff check成功。改変ラベル/Transition/余分なschema fieldを拒否。
モジュールimportがtorchを読み込まないことを別processで確認。
Torchなし環境のため全training suiteは実行していない。学習依存を追加せず生成前処理のみ測定。
測定はChatGPT Linux/Python3.12、ユーザーWSLでの実測ではない。

正式Run `648099f20e944cec94844e73ab60d363`、clean source
`a0a0e54`、source tree `bf017ac7952d8862c236eb1a31afa21139adfb9d`。
全Python hash/protocol hash、suite/manifest digestは正本JSONへ保存。
source測定後の変更は報告/現状文書のみ。初回dirty配線Runは正式結果へ加算しない。
Wolframで8×10=80、4×10=40、合計120を確認。
CLIを再実行すると全cases.jsonl/trajectory、report、Run/eventsを再生成する。
保存後の同じauditでsuite/manifest digestが一致することを検証済み。

## 次の一段
固定readerの課題用学習。正解事実→固定行動分類器と予測事実→固定分類器を比較し、
事実精度/誤停止/自力完了を測る。学習前にarchitecture/LR/epoch/seed/CPU上限を登録する。
事前学習済み初期化との比較は一般Corpusの素材/split/Tokenizer/context確認後に別登録。
今回はPR #41依存のstacked Draftで、main/Service/API/aster-web/公開モデルの変更なし。
ユーザー側の追加作業不要。観測#29→aster-web#2、言語#21、研究#24。
