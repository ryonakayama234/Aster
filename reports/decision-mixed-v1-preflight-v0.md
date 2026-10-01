# v1混合学習：生成preflight実測 v0

2026-10-01。[事前計画](../docs/DecisionMixedPreflight-v0.md)、[機械可読正本](decision-mixed-v1-preflight-v0.json)。

## 結論と次の問い

正常例保持の混合比較に必要な生成・到達可能性・入力分離は成立した。
モデル推論/学習更新は0。学習成功・汎化・復帰能力は今回未測定。
次は同じsuite/slots/serializer/Tokenizer digestを要求する2条件×3seedの学習runner。
問いは「正常経路を保持し、候補が同じでも現在状態で修復put/getを選び分けられるか」。

## 実際に生成・確認したもの

|項目|件数・結果|
|---|---|
|train task|8。add/subtract各4、元2親groupの派生|
|正常unique判断|32|
|復帰unique判断|16（計算前/後に実Toolで-999を1回保存）|
|モデルから見えるtrain unique入力|48、異target衝突0|
|normal-repeat-v1|64slot、32unique、正常64slot|
|normal-plus-recovery-v1|64slot、48unique、正常48+復帰16slot|
|正解Action頻度|両条件calculator/put/get/stop各16slot/epoch|
|train候補coverage・rule継続|48/48|
|dev候補coverage・rule継続|128/128|
|独立終状態確認|176/176、合計888transition|
|同候補・異targetのtrain対照|8組（caseを共有、独立taskではない）|
|devのtrain union重複|14/128判断、保持確認へ分離|
|dev主対照|32組中、非重複24組。旧件数を流用せず再計算|
|最大BOS/EOS込みByte系列|train812、dev1190。context2048内|
|dev candidate-only oracle|104/128。ラベル由来情報上限で学習済み精度ではない|

正常例は全て含むが混合の正常calculator/put提示回数は対照の半分。
各25epochなら両1,600更新、Actionごと400提示、正常提示control1,600/mixed1,200。
Wolframで64slot、1,600/arm、9,600/全6arm、各16slot、正常1,200提示を照合。
固定総予算の配分介入であり、復帰データだけの純粋な効果とは扱わない。

## 検証の意味

既存v1 Sessionの実Toolでprefixを作り、状態・観測・候補・教師をreplay。
rule継続後はgoal flagだけに依存せず、taskの演算結果、停止時memory、最後の対象キーputより後の正しいgetを別検査した。
teacher/target_index/evaluationを変えてもcompactが変わらないことを点検。Byte roundtrip、保存例の再読込/replay、全train/dev間の入力正解衝突0を確認。
空履歴は既存Trajectoryの互換上v0を返すため、履歴のない初期状態はv1 Session/Run契約で指定。非空履歴にはv1を要求する。旧履歴をv1と改名していない。

別CLIは保存JSONから176caseの入力/target/byte長/digest、128slotと各例exposure、dev重複/pair分母、suite digestを再計算した。
提示slot改変のコピーを拒否。全pytest163 passed、最終metadata変更後の関連4 passed、Pyright0 errors/0 warnings、diff check成功。
これらは配線/記録の検証であり、モデルが判断を学べる証拠ではない。

## 実行環境と証拠

- formal Run `c565a9d6d6f449448c451ae9bf438d66`。clean source `f3fac1b66726649ed39f2e09006c2ecea14c1ebf`、source dirty=false。
- source全Python/protocol hashとcase/input/target/candidate/slot/suite/pair digestをJSON正本へ保存。
- ChatGPT Linux CPU、Python 3.12.14、torch 2.14.1+cpu。ユーザーWSL測定ではない。
- preflight内部wall約0.339秒。Python起動/import/環境導入時間を含まない。
- NumPy未導入warningは出たが、Torch model計算なし・Byte生成/実Tool/replayに影響なし。
- source初期実行はdirty working tree（Run11b1a16da5c34381a0bae0b131a8be14）として正式Runと分離。
  型の戻り値注釈と環境/Tokenizer metadataを確定後にclean sourceから再実行。
- Run全例はCLIで決定的に再生成可能。Gitへ軽量summary/176case manifest/全source hashを保存。
- 旧/予約testの構築・点検・採点0。モデル初期化/推論/更新・temperature fit0。

## 限界と次の一段

2親group由来のdev probeで独立holdoutではない。反復slot/派生判断数を独立課題数へ加算しない。
混合は誤保存1回だけを教える。2回反復と完了後上書き/同値再保存の失敗は、未知のprefix/未学習の仕様差として別sliceへ残す。

次のrunnerは、この生成物digestを検証して正常保持・非重複主対照・teacher-prefix・自力継続・初期episodeを分ける。
fit未成立なら汎化議論に進まず、予算を変えずに全seedを報告する。
Service/aster-web/公開recipe/モデル採用は未変更。観測#29→aster-web#2、言語#21、育成#20を別経路で継続。
PR #37をbaseにするstacked Draft。ユーザー側の追加作業は不要。
