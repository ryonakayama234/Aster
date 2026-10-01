# v1混合学習：生成preflight v0

2026-10-01。基準 PR #37 head ab7f817767bd5e142943241aaacfc2a6aad5ecd7。

考え方：学習より先に、正常32/復帰16の到達可能性、v1教師/候補/評価一致、model-visible衝突、dev重複を確かめる。
作業：既存Sessionと実Toolで生成→replay→rule継続→Byte可逆/長さ→union重複→manifest保存。
達成目標：64slotの両armが予定頻度を満たし、全例が正解候補を持ち、独立終状態検査に通る。
停止条件：衝突/教師不一致/頻度不一致/context超過があれば失敗として保存。学習や容量変更で回避しない。

生成規則と次の学習条件はDecisionMixedTraining-v0.md。変更はpreflight実装だけ。
train8task×正常4=32、同8task×計算前/後誤保存1回=16。normal-repeat正常32×2、mixed正常32+復帰16+正常phase2/3再提示16。
devは既存fixed_v1_diagnostics.build_casesの128例/32組。新train unionへ照合して保持確認を分離する。
同一候補集合・異targetのtrain組も別manifestへ残す。候補のみoracleは診断ラベルを使う情報上限で学習済み精度ではない。
Byteは既存Decision artifact interface、BOS/EOS込みcontext2048。全候補列を可逆検査する。
実Tool prefixをreplayし、rule継続後のtask値・対象キー最後put後のget順・停止を独立に点検する。
教師/phase/evaluationのメタデータを変えてもcompact入力が同じか、入力へのラベル依存を検査する。
旧/予約testは構築・点検・採点0。モデル初期化/推論/更新0、temperature fit0。

Runへ全train/dev例、提示slot、case/pair/target/candidate/input digest、source/Python/protocol hash、環境、費用を保存。
Gitには軽量summaryとcase manifest、生成コードを保存する。全例はCLIで決定的に再生成可能。
配線成功は能力改善ではない。同family devは独立holdoutではなく、最終pair件数は検証後に確定する。
次：preflight成功時のみ、同じdigestを要求する学習runnerを別実装。ユーザー側作業不要。
