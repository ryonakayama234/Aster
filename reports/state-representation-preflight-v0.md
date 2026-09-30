# 状態判断・短縮入力の学習前検証：実測

2026-10-01 JST。[事前計画](../docs/DecisionStateRepresentation-v0.md)、Issue #24。
ChatGPT Linux CPU / Python3.12で実行。ユーザーWSLでの速度・モデル能力の実測ではない。

|検査|実測|
|---|---:|
|dev判断case / 元train親group|48 / 2|
|同じstep・異なる正解の対照組|32|
|異なるstep・同じ正解と短縮入力の対照組|32|
|prefixの状態・観測・評価・候補・教師の再実行一致|48/48|
|ルールによるprefixからの追加実行完了|48/48|
|教師候補coverage|48/48|
|全短縮入力が同一なのに正解が違う衝突|0|
|同じ全候補集合に異なる正解がある集合 / case|4 / 24|
|元train手順のstep-only対照|16/48|
|first-candidate対照|12/48|
|候補集合だけを見る決定的分類のin-sample oracle上限|36/48|
|全case・全候補の現行入力 UTF-8 bytes合計|153,402|
|同じ対象の短縮入力 bytes合計|108,138|
|短縮/現行比|0.7049321391|
|モデルによる採点 / test採点|0 / 0|

## 何が分かったか

たとえば計算済みだが未保存の状態と、保存済みだが未取得の状態では、候補一覧が同じでも正解はmemory.put/memory.getで異なる。この対照は候補だけからは全件を解けず、状態・関連履歴を使う効果を検査できる。

同じstepの対照には候補一覧自体が変わる組もある。その32組全部を「候補から正解が分からない」とは扱わず、上記の候補集合別の曖昧性を別に検査した。oracle上限はdev正解で算出した説明用上限であり、学習済みcandidate-onlyモデルの測定ではない。

短縮入力は対象キーの現在値、関連する生のAction/Observation、taskと候補を保持する。教師の段階・正解・Evaluator結果を追加せず、無関係操作による入力変化を除けた。この48件で異なる正解を潰す衝突はなかった。ただしルールも候補生成も限定課題の専門設計であり、汎用エージェント能力の証明ではない。

Wolfram Languageで`108138/153402`、削減率、対照組件数、対照の割合を算術照合した。約29.5%の削減はbytes/token長の効果であり、accuracyの改善ではない。

## 保存・再現・検証

実行Run：`27e4fc84f41a4f398a51a0a3e77745b6`。
base main：`098a009c35b6a023de2877e12ce6fab2e71f36a4`＋今回の未commit実装。
測定Python source snapshot：`9250980a12fa3d432b9c84dc34f36f72e18da4307da295b5e8b957297558f636`。
[機械可読記録](state-representation-preflight-v0.json)にfile hash、suite/pairs digestを保存。Run全件ログはローカル生成で、repoには再生成コードと要約を残す。

新規5testを含む関連21test成功、Pyright 0 error/0 warning。改ざんした観測/step/teacherの拒否、対照関係、連番のprefix継続、target非依存の入力、dev-only、torchを読み込まないimportを確認。テスト成功はモデル学習の成功を意味しない。

preflight CLI・対照課題・短縮serializer・監査は実装済み。学習比較runner、model-only prefix継続の測定、Service/Web公開は未実装。PR #30には依存せずmainからの独立変更。次は事前計画のraw/compact同条件比較を実装し、train fit・状態判断・自力継続を分離して測る。
