# LANG v0 Gate 3A — frozen held-out scoring preflight

Issue #92; parent #78. Status: **code-only and synthetic tests; private WSL training not executed**.

## Research question

Does the frozen Japanese TinyLM improve held-out next-token NLL relative to (a) its step-0 checkpoint and (b) a train-only smoothed unigram baseline? Gate 3A implements the scoring contract, not a language capability verdict.

## Frozen identity

- Training view: 6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d
- BPE artifact: 37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513
- Total vocabulary: 514 (includes BOS/EOS)
- Existing pilot config: configs/tinylm-pilot-v0.json — context 128, width 64, heads 4, layers 2, batch 8, 200 updates, eval every 50, LR .003, seed 42.
- Source/config Git blob SHA pins: configs/lang-v0-gate3.json.
- Trainer instrumentation change: save Linux ru_maxrss in completed local Run summary; no optimizer/model changes.

## Metrics

- NLL: valid target-token-weighted cross entropy, includes EOS but not BOS or padding.
- Text BPB: sum of non-special target token NLL in nats / [ln(2) * sum of the corresponding BPE token UTF-8 byte lengths]. EOS has its own NLL and target count and is not included in text BPB.
- Laplace unigram: (train target count + 1) / (all train targets + vocabulary size), trained only on train token-target positions. The public report contains the count-vector digest, not raw text.
- Domain-level NLL/byte/target accounting uses the identical mask for the model and unigram.
- No dev fitting, no sealed-test target scoring, no model promotion.

## WSL2 preflight — read only

Run from the local Aster Linux checkout after switching to the reviewed branch:

    cd /home/zhong/src/Aster
    .venv/bin/python scripts/run_lang_v0_gate3.py --root .

It audits pinned file identities, config, tokenizer provenance and split hashes. It reads train/dev text only, not test bytes. It prints aggregate counts and returns without weight updates.

## Gate 3B — later, not started in this PR

After an independent review, a separate approved 200-update local pilot Run can use the existing train_tinylm.py command with the pinned view/tokenizer and the unchanged configs/tinylm-pilot-v0.json.

After that Run completes, the read-only reporter command is:

    .venv/bin/python scripts/run_lang_v0_gate3.py --root . --run runs/<id>

It validates the completed Run and selected training windows, checks checkpoint SHA-256 and metadata, reloads all five frozen checkpoints and writes only local runs/<id>/lang-gate3-report.json. Existing differing reports are never overwritten.

**Known legacy caveat:** the existing pretrain.prepare_windows -> verify_view code reads test file bytes *for integrity verification*, although it never trains or scores test targets. The new Gate 3A preflight/report does not perform that read. Before a Gate 3B training run, this provenance-only legacy read must be audited/changed or explicitly acknowledged. It does not justify opening sealed-test outcomes.

## Interpretation and stop rule

Gate 3B will classify results as LANG-Learning observed, Overfit only, or Inconclusive. The current held-out dev is only one Japanese prose document, so improvement would not demonstrate broad dialogue capability. No hyperparameter search, corpus edits, new Tokenizer or Decision/DIAG modification is part of this gate.

Gate 3A local synthetic tests are not evidence of language-learning success. GitHub CI and independent review must be checked before merging. Do not commit private corpus, local model checkpoints, or measurement Run data.
