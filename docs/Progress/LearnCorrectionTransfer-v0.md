# Progress: LEARN v0 — Correction Transfer

## Session 1 — 2026-10-03

Task: ACTを閉じ、Correction Transferの反証可能な研究境界を固定する。

Status: protocol fixed; matched-control implementation next

### 完了

- ユーザー報告によりACT v0はWSL2実機確認済みとして扱う。Issue #48へ記録しclose。
- 実ACT Run IDは `5a8076571dca446fa07190cf4fc62509` と後から補完。bundle ID等、未記録の追加値は推測して補わない。
- branch `feat/learn-v0-correction-transfer` をAster mainから作成。
- Issue #20を **LEARN v0 — Correction Transfer** へ再定義。
- 中心仮説を「student-visited correctionが、同budget Replayより未訂正siblingへ転移するか」に固定。
- P0 Parent / R1 Replay / C1 Correctionの3 armを固定。
- Repair / Local Transfer / Sequential Transferを別Gateに分離。
- seed/decision stepを独立task数として数えず、task/generator familyを基本単位とする。
- 良い結果を見てseedを追加する任意停止を禁止。
- RL / new Tokenizer / new serializer / model scaling / sealed testを非目標とした。

### 既存実装の確認

- `run_logged_intervention_agent` はstudentが訪れた各stateへshadow teacher labelを保存できる。
- `train_intervention_candidate` はparentをdeepcopyし、base + intervention examplesからcandidateを作れる。
- 現状はCorrection固有効果を分離するmatched Replay armがない。
- 次の最小実装はReplay control helperとbudget invariant test。

### 次

1. `src/aster/training/intervention.py` にmatched Replay controlを追加。
2. testsでparent不変・同example列長・同step budgetを固定。
3. 1 familyのdevelopment wiring probeを固定してP0/R1/C1を一周。


## Session 2 — 2026-10-03

Task: matched Replay controlを実装する。

Status: implementation complete; CI pending

### 実装

- `replay_decision_examples(base_examples, count)` を追加。
- Correctionのtrainable example件数と同数だけbase supervisionをdeterministicに再提示できるようにした。
- `DecisionReplayUpdate` を追加し、replay-only candidateのtraining examples / before-after metrics / loss列を保持。
- `train_replay_control_candidate` を追加。parentをdeepcopyし、元modelを変更しない。
- 同じ `DecisionTrainConfig` とadded-example countなら、Correction armとtraining example列長・optimizer step budgetを一致させられる。
- equal stepはequal FLOPsと主張しないことをdocstring/specへ残した。

### テスト

- base例とCorrection例の内容が等価な人工条件で、Replay/Correctionのtraining lengthとstep数が一致することを検査。
- 同じparent / seed / config / model-visible supervisionなら両candidateの全weightが一致することを検査。
- parent weightsが両arm学習後も不変であることを検査。

### 未確認

- GitHub Actions pytest / Pyright。
- 実student rollout由来のcorrectionを使うdevelopment wiring probe。
- Artifact保存・reloadを含むP0/R1/C1実験。


## Session 3 — 2026-10-03

Task: P0/R1/C1を同一benchmarkへ通すLEARN-Wiring runnerを追加する。

Status: implementation complete; CI pending

### 実装

- `run_logged_correction_transfer_experiment` を追加。
- student rolloutはmodel-onlyを要求し、各traceで `route == model` と `executed_action == student_action` を監査。
- student-visited stateのtrainable Teacher label数をCorrectionのadded-example countとする。
- R1 ReplayとC1 Correctionを同じparent / train config / optimizer-step count / training-example countで作る。
- P0 Parent / R1 Replay / C1 Correctionを同じBenchmarkSuiteで測定。
- parent→replay、parent→correction、replay→correctionの3比較を保存。
- `equal_flops_claimed=false` を明示し、同step数を同計算量と誇張しない。
- candidateはpromotionせず `candidate_only`。

### テスト

- fallbackなし、threshold 0のmodel-only policyで三者比較runnerを一周。
- replay/correctionのadded example数とtraining example数が一致することを検査。
- 3比較が保存されることを検査。
- 親model weightsが実験後も不変であることを検査。

### 境界

- このtest fixtureはresearch development probeではない。
- sibling transfer / model-only sibling episode / Artifact save+reloadは未実装。
- sealed testや既存devの結果を能力主張に使っていない。


## Session 4 — 2026-10-03

Task: LEARN-Wiring runnerのCI gate。

Status: CI passed

### 初回失敗

- GitHub Actions run #129でPyrightは成功、pytestは1 failure / 124 pass。
- 新しいmodel-only wiring testで、未学習parentが最初に誤ったTool Actionを選択した。
- tool failureを含むtrajectoryが次stateの入力へ入り、serialized decision inputが88 tokenへ伸長。
- test fixtureがbase/benchmark入力だけから `context_length=25` を決めていたため、`88 > 25` で明示的に失敗した。
- runner側でtruncateせず、failure-historyを含む逐次実行では入力長が伸びるという実際の制約として扱った。

### 修正

- correction-transfer wiring testだけ `context_length >= 256` を確保。
- runtime / serializer / truncation semanticsは変更していない。
- context超過時に黙って切らず失敗する既存挙動を維持。

### 最終CI

- GitHub Actions run #130。
- Pyright: success。
- pytest: **125 passed in 47.89s**。
- これでprotocol + matched Replay control + P0/R1/C1同一benchmark配線の自動テストGateは成立。
- まだ実parent Artifactを使うdevelopment probe、sibling transfer、model-only sibling episodeは未実測。


## ACT lineage補完 — 2026-10-03

- LEARN v0 development probeのparent lineage起点として、実ACT Run IDを `5a8076571dca446fa07190cf4fc62509` に固定。
- 次のprobeでは `runs/5a8076571dca446fa07190cf4fc62509/run.json` の `inputs.decision_model_artifact_id` を検証し、同じ登録済みDecisionModel Artifactを再利用する。
- Run IDからArtifact IDを推測・再生成しない。Run evidenceに記録されたlogical IDだけを使う。


## Session 5 — 2026-10-03

Task: 実ACT Runから同じparentを再利用するdevelopment probe入口を固定する。

Status: implementation complete; CI/user WSL run pending

### Lineage

- ACT Run IDを `5a8076571dca446fa07190cf4fc62509` と補完。
- Issue #48 / ACT Progress / LEARN Progressへ記録。
- probeはACT Runの `inputs.decision_model_artifact_id` だけをparent参照として使う。
- completed agent / ACT recipe / model-only / fixed ACT taskを検証してからArtifactCatalogでresolveする。

### Fixed development family

- family: `learn-dev-key-shift-family-v0`
- correction: add(2,3) → `diagnostic_total`
- uncorrected sibling: add(2,3) → `sibling_total`
- 数値・operationを固定し、key shiftだけを見る。
- seed=42、100 steps、LR=0.003、fallbackなし。
- confirmatory evidenceには算入しない。

### 実装

- `src/aster/training/learn_probe.py`
- `scripts/run_learn_correction_transfer_dev.py`
- ACT Run → registered parent Artifactの厳格な逆引き。
- exact student-visited correction stateについてP0/R1/C1 Repair metricsを保存。
- sibling benchmarkはproject sealed testを開かず、development-only suiteとして生成。
- training条件はCLIから変更不可。

### 次

1. CI。
2. ユーザーWSLで固定CLIを1回実行。
3. Repair / uncorrected sibling結果を読んで配線を監査。
4. P0/R1/C1のsibling model-only episode保存を追加。
5. その後にconfirmatory family manifestを事前固定。


## Session 6 — 2026-10-03

Task: fixed real-parent development probe CI gate。

Status: CI passed; user WSL run pending

- GitHub Actions run #138。
- Pyright: success。
- pytest: success。
- ACT Run ID lineage resolver、fixed key-shift family、exact Repair evidence、development probe CLIを含むlatest headで成功。
- 次のGateはユーザーWSL上で `scripts/run_learn_correction_transfer_dev.py` を1回実測すること。
- このprobe結果はconfirmatory family勝率・能力主張へ算入しない。


## Session 7 — 2026-10-03

Task: user WSL development probe first attemptのlineage failureを修正。

Status: resolver fix implemented; CI pending

### 実機で観測した失敗

固定CLIは学習開始前に
`ValueError: ACT run does not exist`
で停止した。

NumPy未導入のPyTorch warningも表示されたが、例外原因ではない。

### 原因

- LEARN probe resolverはACT Runを `<repo>/runs/<RUN_ID>/run.json` と仮定していた。
- ACT v0はAster Service Jobとして実行される。
- `JobManager` は各Jobを `<repo>/runs/workbench/<JOB_ID>/` に隔離し、recipeへそのJob directoryを `--root` として渡す。
- したがって実ACT Runは
  `<repo>/runs/workbench/<JOB_ID>/runs/<RUN_ID>/run.json`
  に存在する。
- `--root .` を渡したユーザー操作は正しかった。resolverの保存構造モデルが誤っていた。

### 修正

- direct Run `runs/<RUN_ID>/run.json` とService Run `runs/workbench/*/runs/<RUN_ID>/run.json` の双方を探索。
- Run store外へのescape/symlinkを受理しない。
- 同じRUN_IDが複数見つかれば曖昧として停止。
- Service workbench構造を再現する自動テストを追加。

この失敗は能力結果ではなくLEARN-Wiringのlineage bugとして記録する。


## Session 8 — 2026-10-03

Task: user WSL second attemptのparent calibration mismatchを修正。

Status: implementation fix complete; CI pending

### 実機で観測した失敗

Service workbench内ACT Runの解決には成功し、registered parent Artifact loadまで進んだ。その後、
`ValueError: Parent artifact temperature must be positive`
で学習前に停止。

### 設計確認

- ACT v0 runtime `service_learned_agent.py` は `ModelPolicy` を直接実行する。
- ACTのAction選択はraw DecisionModel scoreのargmaxであり、Artifact calibration temperatureを使用しない。
- LEARN development probeはmodel-only保証のため `SelectivePolicy` をwrapperとして使うが、threshold=0 / fallbackなしではrouteは常にmodel。
- この条件ではtemperatureはrouting confidence表示にしか影響せず、selected Actionには影響しない。
- したがってArtifact calibrationをprobe実行の前提条件にするのはACTとの意味論を不必要に狭める。

### 修正

- LEARN model-only probeのrouting temperatureをidentity `1.0` に固定。
- parent Artifact calibrationは実行条件ではなくdiagnostic metadataとして保存する。
- missing / invalid_type / invalid_value / validを区別。
- Action selection、parent weights、Tokenizer、candidate builder、training条件は変更しない。

NumPy未導入warningは引き続き今回の例外原因ではない。


## Session 9 — 2026-10-03

Task: fixed real-parent LEARN development probe WSL実測。

Status: completed; metric interpretation pending

- ユーザーWSLで固定CLIが最後まで完了。
- LEARN development probe Run ID: `43d3966e1b0049dc904e2c0330d8abf0`。
- lineage起点はACT Run `5a8076571dca446fa07190cf4fc62509`。
- これにより実parent Artifact load → model-only student rollout → teacher correction収集 → matched Replay / Correction training → P0/R1/C1 sibling benchmark → canonical Run保存まで実機で一周した。
- CLIがRun pathを返して完了したため、LEARN-Wiringの「実parentで一周」は確認済み。
- Repair / Local Transferの数値解釈はRun artifact内容を読み取るまで保留。
- model-only sibling episode（Sequential Transfer）とcandidate Artifact save/reloadはまだ未実装。
- この1-family probeはconfirmatory能力結果へ算入しない。
- GitHub Actions run #145: Pyright / pytest success。
