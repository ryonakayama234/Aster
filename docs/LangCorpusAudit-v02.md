# LANG v0.2 — Frozen train-only corpus quality audit

Status: implementation proposal for Issue #101. New data and training are out of scope.

## Question

Measure whether actual LANG v0 train text overrepresents Lean-related prose,
numbered references and markup. Separate source concentration from model
capacity and undertraining; observational correlation is not causal proof.

## Frozen inputs and provenance

- Fixed LANG v0 train-view ID and 514-token train-fitted byte BPE tokenizer.
- Existing Gate 3 verifier checks train/dev file hashes, index, and cross-split
  source grouping, but never opens sealed test text (manifest metadata only).
- Per-train-document snapshots are SHA-256 checked using the exact bytes later
  decoded and tokenized.
- configs/lang-v0-corpus.json provides raw source *size metadata only*;
  private train-view text provides post-extraction bytes and actual BPE tokens.
- Require exact 40-hex reviewed commit, clean tracked worktree/index, and
  exact checkout root identity.

## Metrics

Each selected train document has raw-original bytes, extracted train bytes,
BPE text tokens excluding BOS/EOS, prediction targets including EOS, and
non-overlapping context-128 windows. Private reports group metrics by source
group and genre/domain.

Literal scanners count case-insensitive lean substrings, [3], numbered
brackets, triple-backtick code fences, URL prefixes, HTML-like tags and
non-whitespace control characters. These are *review indicators*, not proof
that any markup is unwanted. The source-tag Lean group share and single
largest-document share are computed from **actual BPE text tokens**.

Potential near-duplicates are flagged by 64-character shingle Jaccard >= 0.20.
This can miss paraphrases and some partial overlap. It does not imply
semantic equivalence or leakage without examining source/session identity.

## Output boundary

- Terminal: small global numeric aggregates only. No source text, prompt,
  document IDs, group IDs, private hashes or credentials.
- Private: new runs/lang-v02-corpus-audit/private-report.json (0700 directory,
  0600 file) with per-document labels, metrics and candidate overlap pairs.
- Never rewrite previous Runs or the frozen data, model or tokenizer.
- The private file is fsynced in a temporary 0600 file, then published
  atomically. I/O failures delete incomplete artifacts and the new directory.
- Second successful run fails instead of overwriting private report.
- No Drive writes, raw text transfers to GitHub, new ingestion or training.

## WSL2 command after review and merge

Switch temporarily to the exact reviewed merge commit (restoring the original
checkout in a shell EXIT trap, as previously done for Gate 4), then run:

    .venv/bin/python scripts/audit_lang_v02_corpus.py --root . --expected-commit "$AUDITED_MERGE_COMMIT"

Set AUDITED_MERGE_COMMIT to the **full exact 40-hex SHA** after merging.
Do not infer it from HEAD or a mutable branch. Verify output directory does
not already exist before starting. Share only aggregate stdout.

## Limitations and next decision

The same frozen model has ~174k parameters, 514-token vocabulary, context 128,
and 200 optimizer steps. Distorted generation and repetition may reflect all
these factors; a corpus-mix audit alone cannot isolate causation. LANG v0 has
only a prose dev group, so this does not establish dialogue generalization.

A separately approved, versioned data-only trial can follow, using prospective
private Drive inbox intake in Issue #95. Rights/provenance and session grouping
must be reviewed first. Modern dialogue and authored prose are separate genres;
Aozora Bunko can supply selected public-domain literature only after per-work
author and translator status verification and cleaning ruby/annotation metadata.
Neither literature nor formatting fixes are equivalent to dialogue SFT. Do not
open the old sealed test or alter LANG v0 artifacts for the future trial.
