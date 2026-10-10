# LANG v0 Gate 4 — fixed local generation and memorization flags

Issue #97. **No new training.** The previous held-out NLL improvement is on one dev prose document only; this experiment does not claim dialogue ability.

## Immutable comparison

- Completed private Run: `runs/f884ea724e854ab09ed6a3944f03f8f2`.
- Canonical Gate 3 report SHA-256: `08f1c62ee949e45ccc32f67f5ded9c9c48044804c15587d45672571e8ba555f1`. The runner refuses any changed report, model checkpoint hash, step, tokenizer, configuration or source provenance.
- Only checkpoint steps **0 and 200** are used. The checksum is verified against checkpoint bytes **read once**, and PyTorch loads that very same in-memory byte snapshot to eliminate the check/use file-swap race. No weights updated, no sealed test bytes read.
- **Six fixed prompts**, in order: 「朝、窓を開けると」「雨がやんだので」「机の上には」「昨日読んだ本には」「静かな駅で」「それから、私は」.
- Decoding: existing `generate_ids`, greedy argmax, BOS excluded, stop at EOS, rolling context 128, up to 64 new tokens. Deterministic torch runtime with 2 CPU threads, no seed search. Total 6 × 2 = 12 outputs, theoretical maximum 768 generated tokens. No edits after previewing outputs.

## WSL2 after reviewed PR merge

Run from the existing Aster working copy switched to the exact reviewed/merged revision:

```bash
cd ~/src/Aster
.venv/bin/python scripts/run_lang_v0_gate4.py --root .
```

The command reports only non-sensitive aggregate counts and saves a **private** local JSON file under the same Run at `runs/f884ea724e854ab09ed6a3944f03f8f2/lang-gate4-private.json` with mode `0600`. It fails closed if the private result exists; it does not overwrite anything. Open this JSON **on WSL2** for paired comparisons. Do not commit/upload the Run, prompts, generated excerpts, or model weights.

## Precommitted inspection rubric

Inspect **all six pairs**, not only attractive results. Judge (1) valid Japanese text and readability, (2) whether the continuation follows its prefix coherently, (3) repetition/loops, and (4) exact train-text copying. Summarize how many improved, stayed similar or worsened from step 0 to 200, including unclear cases. These human judgments are separate from the automatic flags.

Automatic flags for each output:

- `utf8_valid`: no UTF-8 decoding errors.
- `repeated_token_trigram`: any repeated token 3-gram in the generated continuation.
- `train_exact_32_byte_match`: generated continuation (excluding prompt) contains at least one exact 32-byte span found in a train-token window, **not** the sealed test.
- `prompt_in_train_window`: this exact prompt appears within a train-token window.

An overlap flag is **not proof of memorization**; absent flags do not establish absence of memorization (shorter or cross-window copying is missed). These automatic flags do not quantify grammar or semantic understanding. The underlying evaluation has only one held-out dev document.

## Next decision

Keep source, report, Run/checkpoint hashes and a non-sensitive tally of all six fixed comparisons on Issue #97; keep actual text and full private audit local. Finish this limited audit once, then separately scope LANG v1 dialogue continuation if the observed outputs justify it. **No retraining, model-size/seed changes, SFT or model promotion in Gate 4.**
