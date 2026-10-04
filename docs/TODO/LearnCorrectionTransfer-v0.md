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
- [x] matched-control成立条件をsummaryへ保存（example数 / training sequence length / optimizer steps / config、`equal_flops_claimed=false`）。token数 / wall time / RSSはGate 4 resource auditで別測定。

## Gate 2 — Development wiring probe — COMPLETE (2026-10-04)

- [x] 1つの固定task family / seedを実装前固定。
- [x] model-only student rolloutからteacher correctionを取得（実ACT parent WSL Run `43d3966e1b0049dc904e2c0330d8abf0`）。
- [x] P0/R1/C1を生成・保存・reload（WSL Run `720c9e43a8a248a48829fc6b789ba75f` で3 armともreload_verified=true）。
- [x] corrected state / uncorrected sibling / model-only episodeを別評価（WSL Run `720c9e43a8a248a48829fc6b789ba75f`）。
- [x] このprobeをconfirmatory能力結果に算入しない。

## Gate 3 — Confirmatory manifest — NEXT

- [ ] task/generator family単位のleakage groupを定義。
- [ ] correction memberと未訂正sibling memberを固定。
- [ ] candidate coverageを学習前に確認。
- [ ] seedsを事前固定。良いseedだけ追加しない。
- [ ] episode horizonを事前固定し、development結果を見て延長しない。
- [ ] endpoint hierarchyを事前固定（strict terminal task_success / goal_verified等の扱い）。
- [ ] family-level win / tie / loss規則と欠測・失敗runの扱いを固定。
- [ ] sealed testを開かない。

## Gate 4 — Measurement

- [ ] Repair。
- [ ] Local Transfer。
- [ ] Sequential Transfer。
- [ ] first error / failure propagation。
- [ ] token count / wall time / RSS。
- [ ] family単位のpaired P0/R1/C1 comparison。

## Gate 5 — Record

- [ ] report + machine-readable result。
- [ ] Artifact lineage。
- [ ] Supported / Not supported / Inconclusiveを明示。
- [ ] 次にRL/curriculumへ進む根拠があるか判断。
