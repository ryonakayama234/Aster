# Decision状態対照train v0：実測と次の判断

2026-10-01 JST。親Issue #24、基準PR #32。条件は[事前protocol](../docs/DecisionStateTraining-v0.md)に固定した。

## 考えたことと作業

短縮入力/Byte/モデル/更新予算を固定し、決まった経路に偏る学習例を実状態の対照へ置換すれば判断が改善する、という仮説を検証した。無関係なdelayはcompactから消えるため、遅延だけのデータ変更は元入力の反復になることをコードと署名で確認。前の提案にあった再計算prefixは既存test予約familyに属するため採用しなかった。

対象キーへ誤値を書き込む操作を実Toolで実行する。8 taskのcalculator/put教師各1件を置換して16slotを変更し、get/stopの正常prefixを保持。32slot・各教師8件・trainの数値/キー・serializerを揃える。teacher正解を入力へ追加しない。初期重み・slot permutation・1600更新をseed内で共有したfresh trainingで、旧checkpointの継続学習ではない。

## 学習前検査・分割

- train2条件64判断、新dev32判断、原状態dev48判断、旧dev56判断の合計200記録について実行/教師/候補replayとrule継続を検証。candidate順置換は集合一致を確認してreplayに限り正規順へ戻し、評価では元順を維持。
- 入力正解衝突0、短縮後の変更16slot。新devは同step2のcalculator/put/getとstep3のstop。主対照は候補集合が一致するput/getの8組。
- 新dev4判断はunion-trainとcompact入力が同一の保持確認。新しい状態への指標は残り28判断、両側とも非重複の主対照6組。正規化署名は全候補の入力文字列を候補順非依存に並べたもの。Byteは可逆なので同じ署名は同じtoken内容になる。全candidate token ID/長さは別途auditに保存。
- train/devは同じwrong-write生成family、元task親2group由来。train誤値-999、dev-777。case名・値変更による独立holdoutではなく、既存生成経路への診断。旧test/冗長計算prefix予約testは構築・長さ確認・採点0件。
- 新devのstep-onlyは16/32、first-candidateは8/32。主対照の同候補集合から一定Actionを選ぶだけでは両方正答できない。train-only exact candidate頻度lookupの全予測は各armに保存し、ニューラルcandidate-only対照と区別。

## 全6条件の結果

新判断・新prefixはunion-train非重複28判断、主対照は6組。prefix成功はteacherが作った開始状態からのmodel-only継続で、最初からの自力完了と区別。原48判断には保持例を含み、独立holdoutではない。fitは最後3epochの全正答・正marginを要求する。

|seed|学習条件|安定fit (最終train)|主対照両方正答|新状態判断|新prefix完了|元初期課題完了|数値初期課題完了|原48状態判断|
|---|---|---|---:|---:|---:|---:|---:|---:|
|42|baseline|成立 (32/32)|1/6|16/28|18/28|2/2|0/2|39/48|
|42|state|成立 (32/32)|0/6|10/28|14/28|1/2|0/2|30/48|
|43|baseline|成立 (32/32)|1/6|17/28|18/28|2/2|0/2|36/48|
|43|state|未成立 (31/32)|0/6|8/28|10/28|0/2|0/2|24/48|
|44|baseline|成立 (32/32)|0/6|13/28|17/28|2/2|0/2|33/48|
|44|state|成立 (32/32)|0/6|9/28|15/28|1/2|0/2|27/48|

今回の置換方式による改善仮説は支持されなかった。新状態判断・新prefix継続・原状態判断は全seedで低下し、主対照は改善なし。state条件は2/3seedでfitし、seed43は31/32で未fit。そのseedの失敗を汎化だけに帰属させない。数値/jointの最初からの完了は全arm0/2、stateのキー完了も全seed0/2。全episodeのinvalidは0。

正常経路のcalculator/put教師を置換したため、空履歴から計算する元入力と正常な計算直後に保存する元入力の提示が減った。この介入は「新しい状態を足して元例も保持する」実験ではない。元初期2課題の完了がbaseline2/2からstate1/2・0/2・1/2へ低下しており、元経路を外した影響を原因候補とするが未確定。誤値をtrainで固定し、devで別値/別履歴長へ変えたため、値依存と経路依存も単独で識別できない。「状態例は不要」「モデルが状態を学べない」という広い結論にはしない。

teacher-prefixの誤りとgoal失敗は別に保存した。引数はcandidate builderが作り、モデルによる数値計算や自由引数生成とは扱わない。fallbackなし。invalidはepisode成功率に混ぜない。

## 次の反証可能な段階

追加学習やモデル拡大へ進む前に、保存済み6checkpointを固定して、既知8taskにおける正常な空履歴/計算直後の保持を確認する。誤保存prefixは誤値-999/-777と誤保存回数1/2の交差を事前固定した別診断にし、値変更だけ・履歴長変更だけ・両方変更を分ける。これは今回の結果を見て決めた後続案で、未実装・未測定。今回の本測定に追加した事前登録結果とは呼ばない。

正常経路保持の低下だけなら元32判断を保持する混合データ対照を別protocolで設計。値変更だけで崩れるなら値の役割/表現の仮説へ進む。今回の失敗を根拠に複数条件を同時変更しない。改善を保証せず、どの範囲で復帰判断が成立するかを先に絞る。

## 検証と再現性

全pytest143 passed（新規4件）、Pyright0 errors/0 warnings、diff check成功。能力実測と配線テストの成功を分ける。
保存9,792 train予測から全306epoch曲線のaccuracy/NLL/min margin/argmax/Actionを再計算して照合。devログは1,152行を照合したが、旧56devを二箇所へ保存するため336行が重複しており、実際のdecision採点は816行（136×6）。seed反復を独立task数へ加算しない。
全576episode記録の連番・最終goal判定・選択Action・訪問入力Byte/BOS/EOS、主対照集計、weight/RNG不変、reload、同一初期weight/提示順を照合。詳細件数は[監査JSON](decision-state-training-audit-v0.json)。

測定sourceはclean local commit `9b8f1ab`。各RunにPython全file hashとprotocol hashを保存。ChatGPT Linux CPU/Python3.12.14/torch2.14.0+cpu/2threadsでユーザーWSL測定ではない。PR #32のtorch2.14.1と違うため今回はbaselineも同じ環境で再測定し、今回の条件間差を比較する。全arm30分deadline、最大RSSは同processの最大値でarm別メモリの比較には使わない。NumPy未導入のtorch初期化warningは出たがNumPy APIは使用せず完了。
全Run/curve/input/trajectory/weightsを再読込可能な実測bundleへ保存。軽量summaryと全証拠hashは[実測JSON](decision-state-training-v0.json)。

## 再実行

```bash
.venv/bin/python scripts/run_decision_state_training.py --root . --seed 42
.venv/bin/python scripts/run_decision_state_training.py --root . --seed 43
.venv/bin/python scripts/run_decision_state_training.py --root . --seed 44
```

`--epochs`の変更はdebug扱いで本測定へ加算しない。source差分と環境はRun metadataに残る。旧Runを上書きしない。

## Aster全体・公開状態

今回の到達点はAgent Loop→Trace→Evaluatorで失敗を比較できる研究実装と実測。#20訂正親子比較、#21言語pilot、#29→aster-web#2保存再生は別の達成条件。PR #32に依存するDraftとして提出し、main/Service公開recipe/aster-web/UI/モデル採用には導入していない。ユーザー側の追加作業は現時点で不要。
