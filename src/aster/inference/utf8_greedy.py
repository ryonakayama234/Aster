"""UTF-8 prefix-valid greedy decoding, separate from the frozen Gate 4 policy.

Each byte-BPE token is accepted only if its complete payload can extend a
well-formed UTF-8 prefix. This changes decoding, NOT model weights or logits.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Utf8Generation:
    token_ids: list[int]
    pending_bytes: bytes
    stop_reason: str  # 'eos', 'budget', or 'no_valid_token'


def extend_utf8_prefix(pending: bytes, payload: bytes) -> bytes | None:
    """Return the incomplete trailing UTF-8 prefix, or None if malformed.

    Python's strict incremental UTF-8 decoder also rejects overlong codepoints,
    surrogate encodings, > U+10FFFF, unexpected continuations and invalid leads.
    A token may straddle codepoint boundaries, so validate its ENTIRE payload.
    """
    if not payload:
        return None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    decoder.setstate((pending, 0))
    try:
        decoder.decode(payload, final=False)
    except UnicodeDecodeError:
        return None
    remaining, flag = decoder.getstate()
    if flag:
        raise ValueError("Unexpected nonzero UTF-8 decoder state flag")
    return remaining


@torch.no_grad()
def generate_utf8_greedy(model, tokenizer, prompt_ids: list[int],
                         max_new_tokens: int = 64) -> Utf8Generation:
    """BOS is forbidden; EOS is allowed only between complete UTF-8 scalars.

    This does NOT force a complete final character at the fixed token budget.
    An incomplete final prefix is explicitly reported as pending_bytes, and
    must be classified as invalid complete UTF-8 (not silently repaired).
    """
    if not prompt_ids or type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 4096:
        raise ValueError("Nonempty prompt and max_new_tokens in [0,4096] required")
    model.eval()
    device = next(model.parameters()).device
    vocab = tokenizer.model.vocab
    if tokenizer.bos_id in vocab or tokenizer.eos_id in vocab:
        raise ValueError("Special token IDs cannot overlap byte-token vocabulary")
    ids = list(prompt_ids)
    generated: list[int] = []
    pending = b""
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.config.context_length:]], device=device)
        logits = model(x)[0, -1].clone()
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        next_pending: dict[int, bytes] = {}
        for token_id, payload in vocab.items():
            new_pending = extend_utf8_prefix(pending, payload)
            if new_pending is not None:
                allowed[token_id] = True
                next_pending[token_id] = new_pending
        if not pending:
            allowed[tokenizer.eos_id] = True
        # Only EOS may be emitted from the special tokens, never BOS.
        scores = logits.masked_fill(~allowed, float("-inf"))
        if not bool(torch.isfinite(scores).any()):
            return Utf8Generation(generated, pending, "no_valid_token")
        token = int(scores.argmax())
        generated.append(token)
        ids.append(token)
        if token == tokenizer.eos_id:
            if pending:
                raise AssertionError("EOS after an incomplete UTF-8 scalar")
            return Utf8Generation(generated, pending, "eos")
        pending = next_pending[token]
    return Utf8Generation(generated, pending, "budget")
