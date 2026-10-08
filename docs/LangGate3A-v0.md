# LANG v0 Gate 3A — Frozen held-out scorer

Issue #92 (implementation preflight); parent #78. This is **code-only preparation**, not a successful LANG-Learning result.

## Frozen scope

- training view: `6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d`
- train-only tokenizer: `37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513` (514 IDs, including BOS/EOS)
- configuration: `configs/tinylm-pilot-v0.json`, exact SHA-256 `445df549ccd4e62359248354b48dacf6796ce11cedf9e5839f59cc96971e83de`
- Transformer: 64 width / 4 heads / 2 layers / 128 context; seed 42; 200 optimizer steps, fixed checkpoints 0, 50, 100, 150, 200.
- training views and selected tokenizer live **only on the user's WSL2 checkout**. Do not publish them in GitHub or assume they are accessible to CI.

## What is measured?

Three mutually comparable predictors use the **identical dev window target sequence and mask**:

1. Step-0 TinyLM.
2. Fixed 50/100/150/200 step TinyLM checkpoints.
3. Train-only token-frequency unigram with Laplace add-one smoothing: `p(id)=(train_target_count(id)+1)/(all_train_targets+vocab_size)`. The unigram uses exactly nonoverlap train-window targets including EOS; it does not see dev/test for fitting.

For each predictor: mean NLL in nats/valid target, target count, total NLL, text-target count, EOS count and EOS NLL, text bytes and text-only bits per byte. Byte length comes from the BPE vocabulary's **actual byte strings**, not Unicode code points. The numerator for text BPB includes **non-special targets only**; EOS is kept in the NLL overall, reported separately, and excluded from the text byte denominator. Padded positions and BOS are never targets. Domain breakdown uses sums then divides (token-weighted), never mean-of-document-means.

`bits_per_byte = text_NLL_sum / (ln(2) * UTF8_byte_count_of_text_targets)`.

The text metric is directly comparable across the **same fixed tokenizer/targets**. It is not a claim about dialogue or general Japanese language quality. Dev is only one frozen prose document.

## Local WSL2 commands

No new shell script installs dependencies or downloads any training data. In an existing checkout with prepared private artifacts:

```bash
cd ~/src/Aster
.venv/bin/python -m pytest -q tests/training/test_lang_gate3.py
.venv/bin/python scripts/run_lang_v0_gate3.py preflight --root .
```

The preflight validates pinned identifiers, tokenizer artifact hashes, train/dev view integrity, cross-split metadata and window support. It does **not** read sealed test text, train weights or produce a result claim.

**Gate 3B only, after separate review:** Run the existing 200-step pilot **once**, without any hyperparameter changes:

```bash
.venv/bin/python scripts/train_tinylm.py --root . \
  --view data/training/6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d \
  --tokenizer artifacts/tokenizers/lang-v0/37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513 \
  --config configs/tinylm-pilot-v0.json
# Use the exact runs/<id> printed by the command:
.venv/bin/python scripts/run_lang_v0_gate3.py evaluate --root . --run runs/<id>
```

The analysis reads only an **existing completed** run, verifies all five checkpoint SHA-256 values and source config/tokenizer/view provenance, then writes `runs/<id>/lang-gate3-evaluation.json`. It never re-trains or touches sealed test. Training writes `training-bundle.json`, `experiment.json`, checkpoint hashes, token-presentation counts, reload equality, elapsed seconds and Linux peak RSS in `run.json`.

## Failure modes and limitations

- A missing local view/tokenizer, wrong config digest or wrong checkpoint lineage fails closed.
- `prepare_windows()` uses metadata and train/dev byte hashes while **skipping reads of sealed test text**; new code does not unseal test.
- Existing validation fixtures in CI use synthetic public text only; CI success is **not** an observation of LANG Gate 3 learning.
- The current training loop reports raw batch loss but does not promise optimizer-state resume. Re-executing training is a new run, not a continuation.
- No data-dependent selection of the most favorable checkpoint; 200 steps are the frozen endpoint. The intervening checkpoints are diagnostic.
- Do not combine with persona/SFT, Decision, DIAG/KL, corpus rebalancing, new tokenizer or model scaling.
- Do not merge this Draft before review and automated tests; the private WSL result is a separate milestone.
