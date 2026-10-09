"""Immutable authority for the Gate 3 v0 spec, separate from the evaluator.

The spec pins the evaluator source SHA; this lock pins the spec itself.
Keeping the two hashes in different modules avoids a self-hash cycle.
The WSL CLI additionally pins this lock module's own Git blob.
"""

CANONICAL_SPEC_RELATIVE_PATH = 'configs/lang-v0-gate3.json'
FROZEN_SPEC_GIT_BLOB_SHA = '0000000000000000000000000000000000000000'
