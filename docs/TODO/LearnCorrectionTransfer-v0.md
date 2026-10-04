# TODO: LEARN v0 — Correction Transfer

## Gate 0 — Contract

- [x] ACT v0実機確認を記録しIssue #48をclose。
- [x] Issue #20をCorrection Transfer研究として更新。
- [x] Spec / TODO / Progressを作成。
- [x] AGENTS / READMEへ現在Gateを反映。

## Gate 1 — Matched control — COMPLETE

- [x] base exampleからdeterministic replay追加例を作るhelper。
- [x] parentをdeep-copyしReplay candidateを学習するhelper。
- [x] Replay/Correctionでtraining example列長・optimizer step・configが一致するtest。
- [x] どちらもparent weightsを変更しないtest。
- [x] matched-control成立条件をsummaryへ保存。example数 / optimizer stepsに加え、encoded token数 / padded token positions / candidate sequence数 / wall time / process max RSSを各armで実測し、`equal_flops_claimed=false`を維持。

## Gate 2 — Development wiring probe — COMPLETE (2026-10-04)

- [x] 1つの固定task family / seedを実装前固定。
- [x] model-only student rolloutからteacher correctionを取得（実ACT parent WSL Run `43d3966e1b0049dc904e2c0330d8abf0`）。
- [x] P0/R1/C1を生成・保存・reload（WSL Run `720c9e43a8a248a48829fc6b789ba75f` で3 armともreload_verified=true）。
- [x] corrected state / uncorrected sibling / model-only episodeを別評価（WSL Run `720c9e43a8a248a48829fc6b789ba75f`）。
- [x] このprobeをconfirmatory能力結果に算入しない。

## Gate 3 — Confirmatory manifest — COMPLETE (2026-10-04)

- [x] task/generator family単位のleakage groupを30 familyで明示固定。
- [x] correction memberと未訂正sibling memberを明示列挙し、store_as keyだけを変える規則で固定。
- [x] 全60 task / 240 teacher decisionのcandidate coverageを学習前testで確認。
- [x] seedsを42/43/44へ事前固定。良いseedだけ追加しない。
- [x] episode horizonを8 stepsへ固定し、development結果を見て延長しない。
- [x] primary=L1 sibling teacher-prefix accuracy、L0 manipulation check、L1 NLL/L2 task_success/goal_verified=secondaryへ固定。
- [x] family-level win/tie/loss、exact sign test、欠測・失敗・retry規則を固定。
- [x] sealed testを開かない境界をmanifestへ固定。

## Gate 4 — Measurement

### Gate 4a — Measurement runner

- [x] frozen manifestを唯一のprotocol inputとして90 unit（30 family × 3 seed）をmanifest順に実行するrunner。
- [x] 1 unit = 1 family × 1 seed。seedを独立familyとして数えない。
- [x] completed unitのresume、endpoint未出力failureだけretry、endpoint出力済みfailure/running duplicateを停止。
- [x] clean Git working treeと全unit同一measurement Git SHAを要求。
- [x] partial campaignでは件数/run IDだけを保存し、family effect / p-value / verdictを出さない。
- [x] exact two-sided sign testを整数演算で実装し、Wolfram reference値でregression test。
- [ ] PR #52 CI / review完了後にmainへmerge。
- [ ] merge後の同一commitをWSLへpullし、preflightを再実行。

### Gate 4b — Frozen measurement

- [ ] Repair。
- [ ] Local Transfer。
- [ ] Sequential Transfer。
- [ ] first error / failure propagation。
- [ ] token count / wall time / RSS。
- [ ] family単位のpaired P0/R1/C1 comparison。
- [ ] 全90 unit完成後だけprimary family effect / exact sign test / verdictを生成。

## Gate 5 — Record

- [ ] report + machine-readable result。
- [ ] Artifact lineage。
- [ ] Supported / Not supported / Inconclusiveを明示。
- [ ] 次にRL/curriculumへ進む根拠があるか判断。
