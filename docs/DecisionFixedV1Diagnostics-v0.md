# 固定モデル・current-state v1 診断 v0

2026-10-01。PR #35 head fe8a49fcbf927ad6ae034a2825dfaec3a48e2a07を基準とする事前計画。

## 考え方・仮説・反証

PR #33の正常例置換による悪化を、正常経路の欠落、誤値の文字列、関連履歴回数に分ける。
候補生成器は確認済み時だけgoal_verified候補を供給する。候補集合から得る情報と
状態から得る情報を区別し、同じ候補集合で異なる教師Actionを要求する対照を重視する。
完了契約はユーザー合意済みのcalculate-and-store-current-v1。
旧v0で学習したモデルへの仕様変更診断であり、旧実測の再採点・v1再学習ではない。

- 正常例欠落説: stateモデルが正常経路を保持し新状態だけ失敗すれば、欠落だけの説明を弱める。
- 誤値依存説: 同じtask/phase/回数で-999と-777の予測が等しければ、この2値の依存説を弱める。
- 回数依存説: 同じtask/phase/値で1回と2回の誤保存の予測が等しければ、この2回数の依存説を弱める。
- 状態選択説: 同候補集合・異targetの両方正答で診断する。候補集合だけのoracleは両方正答できない。
- 自力分布説: 最初の誤り、教師prefix判断、model-only継続を分離。初期判断から失敗する場合は自力分布だけでは説明できない。

## 固定条件・変更点

重み/Tokenizer/serializerを更新しない。PR #33のbaseline/state×seed42/43/44の6checkpointを
Aster_Decision_State_Training_v0_20261001.zipからloadし、checkpoint identity/model/tokenizer digestを検査する。
保存元torch2.14.0+cpuと今回の環境版は別記録し、旧cached train最終32判断のAction/logitsを照合する。
serializerはaster-decision-compact-input-0。候補順を維持し、逆順のargmaxも確認する。
教師・候補・評価器・prefix replayは同じv1を明示。旧v0 helperの既定に依存しない。
評価用新serializerや事実flag、teacher正解をモデル入力に加えない。

## 事前固定ケース

既存development_tasksの8task（add/subtract×seen/numeric/key/joint、親2group）だけを使う。
予約testの冗長calculator family、旧testは構築/点検/採点しない。

- 因子診断: 8task×計算前/計算後×誤値(-999/-777)×誤保存(1/2回)=64判断/model。
- 正常対照: 8task×calculator/put/get/stop前の4状態=32判断/model。
- 完了契約対照: 正取得後の誤上書き、同値再保存、他キーput、誤上書き修復直後（再get前）の4種×8task=32判断/model。
- 合計128判断/model、6model=768。全caseから最大8行動のmodel-only継続を別記録する。
- 主対照: 計算後の誤保存4条件と正常get前を各taskで対応させる32組。
  候補集合一致・異targetを実行前に検証。1回書込みの16組のみstepも一致する。
- Wolframによる件数照合を実施。算術一致はモデル能力の証拠ではない。

## 学習前検証・採点・保存

実Toolでprefixを生成し全transition/state/observation/evaluationを再実行で照合。
rule継続128/128、target coverage、compact入力正解衝突0、固定入力長context内、保存再読込を確認。
各caseの入力は6armのtrain入力unionと照合し、重複/非重複を別集計。
入力同一でも教師が異なる場合は黙って混ぜず測定を止める。
同じ候補集合ごとの多数targetからin-sample candidate-only oracle上限を示す。
これは診断データのラベルを使う情報上限で、学習済みbaseline精度ではない。
各モデルのtrain候補集合頻度lookupも別に保存（未知集合はabstain）。

全score/入力/UTF-8 Byte token ID/target/選択/最初の誤り/trajectoryを保存。
Action完全一致の教師精度と独立Evaluatorによる停止時成功を区別する。
stop reason違いが成功を変えない場合も教師不一致として記録し、能力失敗へ機械的に合算しない。
context超過はinvalid、未停止は非成功。invalidを失敗率へ混ぜない。
weightsとtorch RNGの評価前後一致、別reloadスコア一致、元artifactファイル不変を確認。
Runへ契約ID・suite/protocol/source/checkpoint digest・環境・CPU費用を残す。

## 達成目標・停止条件・次の分岐

6モデル全条件の診断・対照・限界・次の判断が残ること。改善は完了条件にしない。
モデルごと30分deadline。timeout/例外はfailed/interruptedとして保存し、条件を延長しない。
条件変更が必要なら推論前に改版、debug測定は正式測定から分離する。
正常経路低下が残れば正常例保持の混合学習を次の別protocolへ。
名前/値に強く依存し正常経路だけ成立するなら関係を明示する表現診断へ。
教師prefixが成立して自力継続が崩れるなら訪問状態への訂正を次に検討する。
新仕様対照だけ失敗ならv1学習データ導入を別実験化し、容量不足と断定しない。
同じ2親group由来の開発診断で独立holdoutではなく、派生数/seed数を独立task数へ加算しない。
Service/aster-web/公開モデル/promotionは変更しない。ユーザー側追加作業は不要。
