# TODO: LEARN v0 — Correction Transfer

## Gate 0 — Contract

- [x] ACT v0実機確認を記録しIssue #48をclose。
- [x] Issue #20をCorrection Transfer研究として更新。
- [x] Spec / TODO / Progressを作成。
- [x] AGENTS / READMEへ現在Gateを反映。

## Gate 1 — Matched control

- [x] base exampleからdeterministic replay追加例を作るhelper。
- [x] parentをdeep-copyしReplay candidateを学習するhelper。
- [x] Replay/Correctionでtraining example列長・optimizer step・configが一致するtest。
- [x] どちらもparent weightsを変更しないtest。
- [ ] added supervision以外の条件差をsummaryへ保存（example数/step/configは保存済み。token数/time/RSSは未実装）。

## Gate 2 — Development wiring probe

- [x] 1つの固定task family / seedを実装前固定。
- [ ] model-only student rolloutからteacher correctionを取得（固定CLI実装済み、実ACT parentでのWSL実測待ち）。
- [ ] P0/R1/C1を生成・保存・reload。
- [ ] corrected state / uncorrected sibling / model-only episodeを別評価。
- [ ] このprobeをconfirmatory能力結果に算入しない。

## Gate 3 — Confirmatory manifest

- [ ] task/generator family単位のleakage groupを定義。
- [ ] correction memberと未訂正sibling memberを固定。
- [ ] candidate coverageを学習前に確認。
- [ ] seedsを事前固定。良いseedだけ追加しない。
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
