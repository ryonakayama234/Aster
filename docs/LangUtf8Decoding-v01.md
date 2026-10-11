# LANG v0.1 — one UTF-8 prefix-constrained decoding diagnostic

Follows [Issue #99](https://github.com/ryonakayama234/Aster/issues/99), completed Gate 4 Issue #97 and merged PR #98.

This is a **post-hoc decoding-only diagnostic** on one existing fixed small CPU TinyLM. It does **not** revise the Gate 4 result, retrain a model, retune seeds/prompts, change the Tokenizer/corpus, or inspect sealed test text. Byte-level BPE itself correctly encodes Japanese text; unconstrained greedy decoding can concatenate its byte pieces into malformed UTF-8.

## Frozen inputs and expected comparison

- Existing private Run: `runs/f884ea724e854ab09ed6a3944f03f8f2` only.
- Same two SHA-verified checkpoints (steps 0 and 200), Gate 3 report SHA-256 `08f1c62ee949e45ccc32f67f5ded9c9c48044804c15587d45672571e8ba555f1`, six Gate 4 fixed prompts, context 128, deterministic CPU and 64-new-token maximum.
- Existing private `lang-gate4-private.json` serves as a read-only **unaltered baseline**; its SHA is recorded in the new private output. Its policy/checkpoint SHA, prompts and six-pair aggregate counts must match the known Gate 4 observation before executing the new policy.
- New named policy: **strict UTF-8 prefix-masked greedy**. For each candidate token's *entire byte payload*, use strict incremental UTF-8 decoding from the current pending-byte state. Allow valid partial codepoint prefixes, reject malformed continuations, overlong sequences, Unicode surrogates, out-of-range scalar encodings and invalid leads. Never emit BOS or other special byte payloads. Allow EOS only after a complete Unicode scalar; keep the same logits/model and deterministic argmax over allowed candidates.
- The 64-token limit is unchanged. If the 64th token leaves a partial character, `pending_utf8_hex` is preserved in the **private** output, `utf8_valid=false` and `incomplete_utf8_at_budget` increments. This is **not silently repaired** into a valid string. The private preview uses replacement decoding only to display it.
- Record counts of valid UTF-8, pending UTF-8 at budget, controls, repeated token trigrams, EOS, 32-byte exact train-window overlap, and train-window prompt overlap. Training text is SHA-verified in a single-read snapshot; sealed test bytes are never opened.

Maximum new policy generation: 6 fixed inputs × 2 unchanged checkpoints × 64 new tokens = **12 outputs**, at most **768 generated tokens**. The old policy is not regenerated.

## WSL2 execution after reviewed merge

Use the exact **reviewed and merged commit SHA** while protecting any uncommitted work and original checkout, as in Gate 4. The flag is **required**; it must be the actual 40-digit merged commit SHA, not `HEAD` or a branch name. From that checked-out exact revision run once:

```bash
cd ~/src/Aster
.venv/bin/python scripts/run_lang_v01_utf8.py --root . --expected-commit <40-hex-merged-commit-SHA>
```

The runner checks `git rev-parse HEAD`, root checkout identity, clean tracked working tree, then compares the bytes of BOTH `scripts/run_lang_v01_utf8.py` and `src/aster/inference/utf8_greedy.py` against their exact Git blobs from the supplied commit **before generating**. The commit SHA and both SHA-256 source hashes are stored in the private audit and aggregate stdout, so the audited code can be identified after execution. It fails closed on a mismatch or dirty tracked checkout.

The CLI emits **only aggregate counts**. Complete per-prompt results are stored **only locally** in the existing Run as `lang-v01-utf8-private.json` with mode 0600, using exclusive create (no overwrite); Gate 4 `lang-gate4-private.json` is read but never rewritten. Never commit or upload private prompts, generated text, weight files, the full Run, or training corpus.

## Decision rule

Compare the exact same six pairs for (1) malformed UTF-8 (technical correctness), (2) repetitions/EOS/control characters, (3) human-readable Japanese structure and coherence. **Valid UTF-8 alone cannot demonstrate fluency**; it may simply make malformed output printable. Inspect every pair, not just successful examples. Publishing aggregate-only counts and a short verdict concludes this one experimental intervention. No repeat seed search, no max-length or decoding-policy tuning, no retraining, no promotion to conversational competence.
