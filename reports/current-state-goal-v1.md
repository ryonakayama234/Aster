# 完了契約v1：実装検証と次の問い

2026-10-01 JST。ユーザー合意に基づく[事前契約](../docs/CurrentStateGoal-v1.md)。
PR #34の研究メモを前提にした仕様の実装検証。学習実験ではない。

## 考え方と変更

v0は「過去に達成」を保持し、達成後の上書きでも完了扱いだった。
v1は現在の保存値と、最後の対象キーput後の成功getを独立評価する。
教師は誤値を修復して再取得し、候補は現在の確認が成立する場合のみgoal_verified停止を含む。
教師はtaskと引数が一致するcalculator観測を使い、評価器はtaskから独立計算する。

Traceの達成フラグの単調性もv0限定にした。v1は取消を記録できる。
異契約Trace混合を拒否し、v1 Transition/episode保存にschemaと契約IDを付ける。
旧JSON・旧単調性・既存既定動作・旧実測は保持する。契約を明示選択して使用する。

## 確認結果

|境界|v1の確認結果|
|---|---|
|通常の計算→保存→取得→停止|4行動で成功|
|正取得後に対象キーへ誤値をput|達成取消、修復put→get→stopで成功|
|対象キーへ同じ正値を再put|取得確認が失効、getが必要|
|再put直後にgoal_verifiedという理由で停止|停止理由にかかわらずtask_success=false|
|別キーへput|取得確認を維持|
|引数不足で失敗したput|現在値と確認を維持。既存のtool_failure停止方針は継承|
|初期memoryが正値|未取得では不成立、成功get後に成立|
|別引数のcalculator結果|教師・候補は対象taskの計算結果として使わない|
|保存・再読込|取消を含むTraceと契約IDが一致|
|異契約混合・schema不一致・未知契約|拒否|
|v0での達成後誤上書き|従来の完了扱いとJSONを保持|

4行動（対象キー正put/誤put/get/他キーput）の長さ1..5の全1,364経路を
実Toolで実行。別実装の状態機械により6,372途中判断を照合した。
これは単一taskの有限な境界検証であり、独立課題の汎化評価ではない。
Wolframでは現在値・取得存在・取得の時間順・取得値の4条件の真理値表16行を確認。
全条件trueの場合のみ達成する論理の確認であり、時間順の実装検証は上記Python実行が担う。

- 全pytest：154 passed（78.86秒）。新規境界テスト11件。
- 最終のboolean値比較の修正後、新規11件を再実行して成功。
- Pyright（実venv interpreter指定）：0 errors / 0 warnings。
- 追加学習0、学習済みモデル採点0、予約test構築/採点0。
- ChatGPT Linux環境での検証。ユーザーWSL・公開UIでの実機確認は今回未実施。

## わかったこと／次に確かめること

合意した完了条件を実行器・教師・候補・評価器・保存で表現できた。
学習済みモデルが復帰できると確認したわけではない。
episodeのgoal_verifiedとfirst_goal_verified_stepは過去の達成を記録し、
task_successだけが最後の成功停止を表す。過去trueがあるだけではv1成功にならない。

次は保存済みモデルを固定し、正常経路と誤値×誤保存回数を分ける診断。
既存checkpointはv0由来なので、v1へ移す場合には新protocolで候補・教師・入力署名を記録し、
再採点を新測定として明示する。旧結果をv1へ改名しない。
新しい学習・モデル拡大・Service/Web採用は未実施。依存PRが未マージのstacked Draftで保存する。

## 再確認

```bash
.venv/bin/python -m pytest tests/evaluator/test_current_state_goal.py -q
.venv/bin/pyright --pythonpath .venv/bin/python
```
