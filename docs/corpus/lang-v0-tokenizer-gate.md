# LANG v0 — Tokenizer Gate

Gate 0で固定したcanonical buildとgroup splitを使い、既存のUTF-8 byte BPEを
**train splitだけ**で学習する。

## Fixed inputs

- canonical build:
  `239fff3a6ac9c4d796954ee23c81fae15fff3e205a9655e7e184fea2c454f587`
- split contract: `configs/lang-v0-split.json`
- training recipe: `configs/lang-v0-training.json`
- target BPE vocab: 512
- min pair frequency: 2
- context length for audit: 128

splitのhash fallbackは禁止し、recipeのoverrideはfrozen split contractと完全一致させる。

`dialogue/ja/014.txt` と `dialogue/ja/015.txt` はsplit上はtrainだが、
execution/display境界の確認が終わっていないため、このtraining selectionではdeferする。
これはsplit変更ではない。

## One command

mainをpullした後:

```bash
.venv/bin/python scripts/run_lang_v0_tokenizer.py
```

このコマンドは:

1. pinned canonical buildからtraining viewを構築する。
2. frozen group splitを検証する。
3. selected `train/` textだけで512 BPEを学習する。
4. train/devについてbyte baselineとBPEを同一textで比較する。
5. round-trip、bytes/token、tokens/character、document token length、
   context=128の非overlap window数を記録する。
6. BPE mergeのrank/長さをhexで記録する。
7. tokenizer artifactとresource measurementをlocal ignored pathへ保存する。

test sampleはtraining viewにはmaterializeされ得るが、BPE fittingとTokenizer audit metricには使わない。

## Local outputs

- tokenizer artifact: `artifacts/tokenizers/lang-v0/<artifact-id>/`
- run record: `runs/.../`

GitHubへprivate本文や生成training textをcommitしない。

## PASS / STOP

Gate 1で確認するもの:

- train-only fittingが成立
- train/dev round-trip failure = 0
- tokenizer artifact reload可能
- byte baselineとの圧縮差を記録
- document token長 / context-window数を記録
- training time / peak RSSを記録

このGateでTokenizer方式、vocab size、context length、model sizeは変更しない。
