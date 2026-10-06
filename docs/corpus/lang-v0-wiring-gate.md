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

## PR #74 review follow-up — artifact identity / run provenance

Gate 1のartifact IDは`artifact.json`の保存済みバイト列のSHA-256であり、
モデルpayloadだけから作る`tokenizer_id`とは別の識別子である。
同じBPE payloadでもaudit/provenanceが違えばartifact IDは異なる。

- LANG provenance形式では`artifact.json`を必須にし、そのdigestとディレクトリ名を照合する。
- wiring runnerが指定した`tokenizer_artifact_id`をloaderの期待IDにも渡す。
- `evaluation.json`が併存してもLANG provenance/identity検証を迂回しない。
- identityにはtokenizer本体4ファイル、`provenance.json`、`audit.json`のhashを必須とする。
- `wiring_spec_sha256`はrunnerが解析したspecそのもののバイト列から計算する。
- 選択したartifact IDとspec digestを開始時の`run.json.inputs`へ保存し、
  `experiment.json`、checkpoint metadata、training bundleにも引き継ぐ。
  入力検証で失敗した場合も選択した入力はrunに残る。
- 旧evaluation形式は期待artifact IDを指定しない既存呼び出しで維持する。

今回わかったこと：payloadの同一性だけではGate 1の監査・由来の同一性を保証できない。
次に確かめること：固定した実データでGate 2を再実行し、上記2識別子とspec digestが
保存記録から辿れ、従来のreload一致とtest不使用を維持していること。
合成fixtureでの回帰テストは配線の確認であり、言語能力やGate 2実測の代替ではない。

検証（2026-10-07、Python 3.12 / PyTorch 2.14.1+cpu）：
`PYTHONPATH=src python -m pytest -q`は245 passed。
Pyrightは0 errors / 0 warnings。固定したGate 1実データでの再実行は未実施。

追加レビューへの対応：runnerの事前検証はtrainerを呼ぶ前にも失敗し得るため、
runner自身が`lang_v0_wiring` Runを入力読み込み前に作る。
読めたspecのdigest、解析できた選択IDを検証前に保存し、
成功時には別の`pretrain` Runを`training_run_id`で参照する。
失敗・中断はrunner Runにも記録する。spec自体が読めない場合はpathとエラー、
JSON解析できない場合はpath・digestとエラーまでを残し、未知のIDを推測しない。
view/tokenizer欠落、config不正、schema不正、JSON不正、spec欠落、KeyboardInterruptを
回帰テストで確認する。
追加修正後の全体テストは254 passed（57.86s）、Pyrightは0 errors / 0 warnings。
