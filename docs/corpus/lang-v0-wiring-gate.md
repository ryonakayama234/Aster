# LANG v0 — Gate 2 TinyLM Wiring

Gate 1で実測・reload確認したtraining viewとTokenizer artifactを固定し、
既存TinyLMのone-window overfit経路へ接続する。

固定入力:

- training view: `6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d`
- tokenizer artifact: `37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513`
- model/training config: `configs/tinylm-overfit-v0.json`
- context 128 / width 64 / heads 4 / layers 2 / seed 42
- greedy generation policy unchanged

実行:

```bash
.venv/bin/python scripts/run_lang_v0_wiring.py
```

このGateの目的はgeneralizationではなく、step 0 -> one-window overfit -> checkpoint save ->
checkpoint reload -> exact logits match -> greedy generationまで配線が一周すること。

PASS条件:

- step 0 checkpoint/evaluationが保存される
- training lossが有限で低下する
- checkpointが保存される
- actual checkpoint reloadでpredictions exact match
- greedy generationが実行可能
- selected training window manifestが保存される
- testは使用しない

held-out dev改善はGate 3 LANG-Learningで別に判定する。
