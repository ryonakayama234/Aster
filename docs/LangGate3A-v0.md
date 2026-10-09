# LANG v0 Gate 3A — frozen held-out scoring preflight

Issue #92; parent #78. Status: **code-only and synthetic tests; private WSL training not executed**.

## Research question

Does the frozen Japanese TinyLM improve held-out next-token NLL relative to (a) its step-0 checkpoint and (b) a train-only smoothed unigram baseline? Gate 3A implements the scoring contract, not a language capability verdict.

## Frozen identity

- Training view: 6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d
- BPE artifact: 37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513
- Total vocabulary: 514 (includes BOS/EOS)
- Existing pilot config: configs/tinylm-pilot-v0.json — context 128, width 64, heads 4, layers 2, batch 8, 200 updates, eval every 50, LR .003, seed 42.
- Source/config Git blob SHA pins: configs/lang-v0-gate3.json, including **all 17 files captured by the trainer's code_sha256 provenance**, including transitive shared modules and executed Python package initializers (TinyLM, Transformer, LM head, checkpoint, inference, Tokenizer, records, and training/view preparation). The read-only report checks the complete code-file set and both the Run-recorded SHA-256 and frozen Git blob hash for each source. A mismatch stops reporting instead of silently comparing different implementations.
- **Frozen spec trust chain (latest P1 fix):** the public WSL entrypoint `scripts/run_lang_v0_gate3.py` checks the reviewed Git blob SHA of `src/aster/training/lang_gate3_lock.py` before importing Aster code. The independent lock module pins the Git blob of the canonical `configs/lang-v0-gate3.json`, and `preflight()` refuses either a noncanonical `--spec` path or modified spec bytes before accepting any identities/paths/hashes from JSON. The frozen spec continues to pin the evaluator Git blob and training source closure. This prevents a spec/evaluator self-hash cycle and prevents silent v0 input substitution. The authority is the reviewed CLI source revision; programmatic calls using monkeypatched checkers in synthetic tests are not an alternate public validation interface. Regression tests cover canonical pass, alternate-path refusal, spec-byte drift, and launcher lock pin.
- Independent-review finding (PR #94): the Gate 3 evaluator itself is pinned by `evaluator_git_blob_sha` in `configs/lang-v0-gate3.json`. Before reading any view or creating a report, preflight compares the actual `src/aster/training/lang_gate3.py` Git blob SHA with the frozen value and aborts on missing/drifted evaluator code. Synthetic tests cover matching, tampered, missing, and preflight-refused hashes.
- Subsequent Codex reviewer finding: the read-only reporter runs as a separate CPU process, so before scoring checkpoints it now applies frozen `threads=2` with `torch.set_num_threads` and the trainer's `torch.use_deterministic_algorithms(True)`. A synthetic test checks the exact runtime settings. This limits thread-related scoring drift; it does not promise byte-identical results across arbitrary CPU/PyTorch versions.
- Third Codex review finding: before loading/scoring any checkpoint, the reporter now requires the evaluation PyTorch version (including its build suffix) to match the training version in `experiment.json`. It also records the training and evaluation PyTorch and Python versions in `dependency_versions` in the local report. A synthetic regression test covers a version mismatch and missing provenance. This does not guarantee bitwise reproducibility across CPU hardware or numerical kernels.
- Trainer instrumentation change: save Linux ru_maxrss in completed local Run summary; no optimizer/model changes.
- Latest independent review finding: trainer and reporter source pins must cover transitive local imports. The frozen 17-file closure now includes `src/aster/corpus/pipeline.py`, `src/aster/training/view.py`, and `src/aster/training/extract.py` in addition to prior files. A source-graph regression test walks `aster.*` imports and existing parent-package `__init__.py` files, compares the full closure to both trainer provenance and frozen spec, and re-hashes every pinned source. In this repository, the three added initializers are `src/aster/__init__.py`, `src/aster/corpus/__init__.py`, and `src/aster/tokenizer/__init__.py`. This is a provenance-only hardening change, not a new experiment configuration.

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

It first authenticates the fixed spec lock, then audits pinned file identities, config, tokenizer provenance and split hashes. The CLI's optional `--spec` flag may not select a different v0 spec; any alternate path fails closed. It reads train/dev text only, not test bytes. It prints aggregate counts and returns without weight updates.

## Gate 3B — later, not started in this PR

After an independent review, a separate approved 200-update local pilot Run must record the pinned BPE artifact ID at the training boundary (the ID is now an optional CLI argument for legacy runs, but required for this Gate 3 Run). Use the frozen inputs and unchanged optimizer/model configuration:

    .venv/bin/python scripts/train_tinylm.py --root . \
      --view data/training/6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d \
      --tokenizer artifacts/tokenizers/lang-v0/37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513 \
      --tokenizer-artifact-id 37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513 \
      --config configs/tinylm-pilot-v0.json

The read-only reporter verifies both `experiment.json` and `run.json` carry exactly this artifact ID, and validates each checkpoint's embedded `model_config` and parameter count (174,080) against the frozen model, not just checkpoint metadata. A Run made with the previous untagged CLI must not be retroactively relabeled.

After that Run completes, the read-only reporter command is:

    .venv/bin/python scripts/run_lang_v0_gate3.py --root . --run runs/<id>

It validates the completed Run and selected training windows, checks checkpoint SHA-256 and metadata, reloads all five frozen checkpoints and writes only local runs/<id>/lang-gate3-report.json. Existing differing reports are never overwritten.

**Sealed-test audit:** Gate 3A also changes the existing `prepare_windows` path to call `verify_view(..., verify_test_text=False)`. It still verifies the full manifest, all listed file paths, train/dev text bytes, and cross-split leakage metadata, but does not open sealed test text. The default `verify_view(view)` continues to perform full-integrity test-byte hashing for non-training callers. Synthetic regression tests enforce the no-read property.

## Interpretation and stop rule

Gate 3B will classify results as LANG-Learning observed, Overfit only, or Inconclusive. The current held-out dev is only one Japanese prose document, so improvement would not demonstrate broad dialogue capability. No hyperparameter search, corpus edits, new Tokenizer or Decision/DIAG modification is part of this gate.

Gate 3A synthetic tests also cross-check the actual TinyLM's NLL against the original training evaluator, padding invariance, and adversarial changes to the Transformer code hash. They are not evidence of language-learning success. GitHub CI and independent review must be checked before merging. After any new source-pinning commit, rerun the read-only WSL preflight at the *new exact head*; older successful logs do not verify newer code. Do not commit private corpus, local model checkpoints, or measurement Run data.

## Full synthetic Run integrity audit (review hardening)

The reporter now rejects a mixed training-bundle experiment, mismatched Run input configuration, wrong/nonmonotonic checkpoint step or `tokens_seen`, unexpected checkpoint filenames, mismatched checkpoint metadata/provenance (including the token counter), and checkpoint train/dev NLL or target counts that disagree with the trainer's observations. A synthetic five-checkpoint report test exercises successful serialization and idempotent read-only reruns, then deliberately corrupts the token counter, observed NLL, and bundle provenance to verify fail-closed behavior. The standalone CLI now mirrors the TinyLM trainer's `src` import fallback. All test inputs are synthetic and do not include private corpus text.
