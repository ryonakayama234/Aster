# Progress: LEARN v0 — Correction Transfer

## Session 1 — 2026-10-03

Task: ACTを閉じ、Correction Transferの反証可能な研究境界を固定する。

Status: protocol fixed; matched-control implementation next

### 完了

- ユーザー報告によりACT v0はWSL2実機確認済みとして扱う。Issue #48へ記録しclose。
- 実ACT Run ID / bundle IDはrepo正本に未記録のため、値を推測して補わない。
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
