# DIAG v0 — Correction Interference Curve

## 目的

LEARN v0で観測した

```text
Correction
  → corrected-state Repair は強い
  → uncorrected sibling Transfer はしばしば悪化
```

の時間発展を追い、negative transferの原因候補を絞る。

これはまず **Trial / diagnostic experiment** とする。能力claimやconfirmatory verdictには使わない。

共通実験モード規約は [Experiment Cycle v0](ExperimentCycle-v0.md) に従う。

## 競合仮説

### H1 — Correction overfit / local interference

学習が進むほど:
- Repair ↑
- sibling margin / accuracy ↓

### H2 — operation-specific representation

addとsubtractで曲線が系統的に異なる。
特にsubtractでnegative transferが早く・強く現れる。

### H3 — optimizer instability

同一familyでもseed間分散が大きく、Repair/Sibling曲線の方向が安定しない。

### H4 — broad forgetting

Correctionでsiblingだけでなくbase supervision retentionも低下する。

## Trial 0 — Representative interference curve

### Evidence scope

- development / diagnostic only
- post-hoc selected cases
- confirmatory capability claimへ算入しない
- sealed testは開かない

### Case selection

LEARN v0結果を見た後に、operation × outcomeの代表例を選ぶ。

候補:
- add loss: family 03
- add win: family 15
- add tie: family 21
- subtract loss: family 02
- subtract win: family 12
- subtract tie: family 22

この選択は `post_hoc_selected=true` と記録する。

### Seeds

- 42
- 43
- 44

6 family × 3 seed = 18 training units。

### Checkpoints

- step 0
- step 10
- step 25
- step 50
- step 100

同じtraining trajectoryからcheckpoint evidenceを保存する。
checkpointは独立sampleとして数えない。

### Metrics at every checkpoint

Correction / Replay双方で:

1. corrected-state Repair
   - accuracy
   - raw NLL
   - correct-vs-best-wrong logit margin

2. uncorrected sibling
   - accuracy
   - raw NLL
   - correct-vs-best-wrong logit margin

3. base retention
   - fixed base supervision accuracy / NLL

4. parameter drift
   - Decision Head L2 distance from parent
   - backbone L2 distance from parent

5. resource / lineage
   - seed
   - family
   - step
   - source Git SHA
   - parent Artifact ID
   - training examples presented
   - encoded token count
   - wall time where meaningful

## Diagnostic signatures

### Signature A — local interference

Repair improves while sibling margin decreases within the same family/seed trajectory.

### Signature B — operation-specific interference

subtract trajectories show systematically earlier/larger sibling degradation than add trajectories.

### Signature C — broad forgetting

base retention degrades together with sibling performance.

### Signature D — instability

direction or timing is dominated by seed variance and no stable curve pattern appears.

## Stop / next-decision rule

Trial 0は18 training unitsで停止する。
結果を見てfamily/seedを追加して同じTrial 0を延長しない。

次は以下のいずれか1つへ進む。

- Aが明確 → train_backbone / replay ratio / LR等のcausal intervention Trial
- Bが明確 → numeric / operation representation Trial
- Cが明確 → forgetting-control Trial
- D → optimizer / initialization stability Trial
- どれも明確でない → instrumentationまたは仮説を再設計

新しい条件は新しいTrial cycle IDで扱う。

## Batchへの昇格条件

Trialでmechanism signatureが絞れた後、新しい独立familyを使ってprimary endpointを事前固定できる場合だけBatch manifestを作る。

Trial 0で選んだ6 familyは、そのBatchの未見confirmatory evidenceには使わない。
