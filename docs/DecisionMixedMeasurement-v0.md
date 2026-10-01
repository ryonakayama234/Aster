# v1正常例保持・混合学習の正式測定 v0

2026-10-01。PR #38 head 21b6f5e55898f2be9a64765834f73ec1b33b62bfに基づく着手前登録。
考え方/生成規則/仮説/分岐はDecisionMixedTraining-v0.mdを継承する。
達成目標は全2arm×seed42/43/44を同条件で測定し、正常保持と状態対照を分離する。
学習改善を完了条件にせず、fit未成立/悪化/invalid/打切りも報告する。

両64slot×25epoch=1,600単例更新、Byte/compact、width16/heads2/layers1/context2048、LR0.001、CPU2threads、AdamW既存default。
seed内全初期weight/slot index permutationを共有。事前生成JSONのtrain/dev/pair/slot/Tokenizer/serializer/input digestを要求し、違えば測定開始前に停止する。
元正常32を両armに含む。混合は復帰16追加と正常get/stop16再提示。正常calculator/put提示量差は配分介入の一部。

epoch0〜25でunique train32/48を評価し、正常32の共通保持指標を別保存。
最終3epoch全unique正答かつ正target marginでfit成立。train候補1個があればNone marginで別計数し正margin条件対象外。
既存fit helperはepoch評価/RNG復元/保存に再利用可能。提示slotの重複とunique評価を混同しない。
旧/予約testは構築・点検・採点0、calibrationなしT=1、promotionなし。

reload後にv1固定128prefixの教師判断/最大8行動自力継続を評価。初期8task episodeは別枠。
非重複主対照24組、正常/誤値×回数/完了取消、重複14判断と非重複114判断を別に報告。
候補集合のみtrain頻度lookup（未知集合abstain）とrule、候補逆順score、weights/RNG/reload不変を確認。
invalidは通常失敗へ合算せず全開始数と両arm共通有効caseの比較を併記。
教師Action完全一致と停止時successを別評価。

学習/評価各arm30分deadline。配線・速度debugは正式結果から分離し、時間が成立したら全seedを結果によらず実行。
異常時はfailed/interruptedとして部分曲線/モデルを保存し、予算延長や成功seed選別はしない。
資源として時間/token数/padding量/maxRSS/parameter数を保存。プロセスmaxRSSはarm専用メモリとは呼ばない。
事前条件の変更が必要なら正式測定前に改版、条件を固定した後はdevを見て追加学習しない。

Runには全生成例/manifest、全epoch予測と順序、全dev score/入力/token/実行経路、checkpoint、source/Python/protocol hashを保存。
Gitには軽量全seedreport、詳細記録とcheckpointは実測bundleへ保存。
独立holdoutではなく元2親group由来probe。3seed×派生caseを独立task数へ足さない。
今回API/Service/aster-web/UI/公開モデルは変更しない。ユーザーPC測定は未実施。

学習前debug補足：最初の1epoch Run51bc713a610e46939501fe2d79e97c6dは候補逆順のfloat完全一致チェックで停止した。選択Actionは同じで最大差は約6e-8。既存診断と同じscore許容差1e-5＋Action一致へ揃え、最大差を保存する。これは正式測定前の配線修正で学習条件変更ではない。次debugの成功後に正式測定する。
checkpointは既存fit schemaに加えtraining-contract.jsonのv1/serializer/slot/checkpoint identityをload時に要求する。
