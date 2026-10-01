# calculate-and-store 完了契約 v1

2026-10-01。ユーザー合意：「停止時点でも正しく保存され、最後の変更後に取得確認できている」。

## 着手前：考え方・作業・達成目標

過去の成功を保持するv0では、達成後の誤上書きが完了を取り消さない。
復帰課題には現在の状態で成立する目標が必要。これは仕様の実装検証であり、学習改善実験ではない。
教師・候補・評価器・Trace保存をv1の選択で整合させ、v0の既定動作と既存実測を維持する。
達成目標は正常経路の完了、上書きによる達成取消、修復・再取得後の完了、保存再読込の一致。
反証は誤値で成功する、同値再保存後に取得せず成功する、他キーの書込みで確認が失効すること。
全条件を満たしたら終了。失敗した場合は仕様・実装を直し、学習へ進まない。

## 正確な意味

- 契約ID：`calculate-and-store-current-v1`。旧契約：`calculate-and-store-ever-v0`。
- 現在の対象キーが存在し、値がtaskから独立計算した正解と一致する。
- 最後の成功した対象キーへのputより後に、対象キーの成功したgetがあり、その最新getが正解を返す。
- 同値のputも再確認を要求する。失敗したputと他キーへのputは確認を取り消さない。
- 初期memoryに正値がある場合も取得確認が必要。記録されたputがない場合、成功getで確認できる。
- 確認後の対象キーputでgoal_satisfiedはfalseへ戻りうる。stop時のtrueだけがtask_successを支える。
- 道具の失敗時にstopする既存方針は継承し、あらゆる失敗からの復帰は今回の目標ではない。
- 全変更をTraceへ記録する実行器を前提とする。未記録の変更と再復元の検出は範囲外。

## 固定条件と版

学習・Tokenizer・モデル・既存測定データ・予約testは変更しない。新契約は明示選択のみ。
`RuleBasedPolicy(goal_contract=...)`、`TaskEvaluator(goal_contract=...)`、
`CalculateAndStoreCandidates(goal_contract=...)`へ同じIDを渡す。
v1のEvaluationResultに契約IDを保存し、Transitionは`aster-transition-1`。
v0は従来JSONを保ち、契約IDのない旧記録も旧契約として読む。異なる契約のTrace混合は拒否する。
episodeのtask_successは最後の停止状態、goal_verified/first_goal_verified_stepは履歴上の達成記録。
v1 episode書出しは`aster-episode-evaluation-1`と契約IDで区別する。
PR #33までの数値はv0のまま保存し、v1の結果として転記しない。

## 次の分岐・公開範囲

教師による復帰の成立とモデルによる復帰の成立を分ける。
保存済みモデルの誤値×履歴回数診断は別protocolで、未測定。新旧契約を混ぜて比較しない。
Service既定recipe、aster-web、公開モデルへの導入は別作業。今回自動採用しない。
仕様は合意済みで、ユーザー側の追加作業は不要。
