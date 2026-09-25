from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aster.tokenizer.bpe import BPEModel, train_bpe


DEFAULT_SPECIAL_TOKENS = ("<|bos|>", "<|eos|>")


@dataclass(frozen=True)
class TokenizerManifest:
    name: str
    version: str
    tokenizer_type: str
    encoding: str
    base_vocab_size: int
    target_vocab_size: int
    actual_bpe_vocab_size: int
    special_token_count: int
    min_pair_frequency: int


@dataclass
class AsterTokenizer:
    model: BPEModel
    special_tokens: dict[str, int]
    manifest: TokenizerManifest

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.model.vocab) + len(self.special_tokens)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.model.encode(text)

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        special_ids = set(self.special_tokens.values())

        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id not in special_ids]
        elif any(token_id in special_ids for token_id in ids):
            raise ValueError(
                "Cannot decode special-token IDs as UTF-8 text. "
                "Use skip_special_tokens=True."
            )

        return self.model.decode(ids)


def train_aster_tokenizer(
    texts: list[str],
    *,
    target_vocab_size: int,
    min_pair_frequency: int = 2,
    name: str = "AsterTokenizer",
    version: str = "0.1",
) -> AsterTokenizer:
    model = train_bpe(
        texts,
        target_vocab_size=target_vocab_size,
        min_pair_frequency=min_pair_frequency,
    )

    next_id = max(model.vocab) + 1
    special_tokens = {
        token: next_id + offset
        for offset, token in enumerate(DEFAULT_SPECIAL_TOKENS)
    }

    manifest = TokenizerManifest(
        name=name,
        version=version,
        tokenizer_type="byte_bpe",
        encoding="utf-8",
        base_vocab_size=256,
        target_vocab_size=target_vocab_size,
        actual_bpe_vocab_size=len(model.vocab),
        special_token_count=len(special_tokens),
        min_pair_frequency=min_pair_frequency,
    )

    return AsterTokenizer(
        model=model,
        special_tokens=special_tokens,
        manifest=manifest,
    )


def save_tokenizer(tokenizer: AsterTokenizer, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": tokenizer.manifest.name,
        "version": tokenizer.manifest.version,
        "tokenizer_type": tokenizer.manifest.tokenizer_type,
        "encoding": tokenizer.manifest.encoding,
        "base_vocab_size": tokenizer.manifest.base_vocab_size,
        "target_vocab_size": tokenizer.manifest.target_vocab_size,
        "actual_bpe_vocab_size": tokenizer.manifest.actual_bpe_vocab_size,
        "special_token_count": tokenizer.manifest.special_token_count,
        "min_pair_frequency": tokenizer.manifest.min_pair_frequency,
    }

    vocab = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in tokenizer.model.vocab.items()
    }

    merges = [
        {
            "left": pair[0],
            "right": pair[1],
            "new_id": new_id,
        }
        for pair, new_id in tokenizer.model.merges
    ]

    _write_json(output_path / "manifest.json", manifest)
    _write_json(output_path / "vocab.json", vocab)
    _write_json(output_path / "merges.json", merges)
    _write_json(output_path / "special_tokens.json", tokenizer.special_tokens)


def load_tokenizer(input_dir: str | Path) -> AsterTokenizer:
    input_path = Path(input_dir)

    manifest_data = _read_object(input_path / "manifest.json")
    vocab_data = _read_object(input_path / "vocab.json")
    merges_data = _read_array(input_path / "merges.json")
    special_tokens_data = _read_object(input_path / "special_tokens.json")

    vocab: dict[int, bytes] = {}
    for raw_token_id, token_hex in vocab_data.items():
        token_id = _parse_token_id(raw_token_id, "Tokenizer vocabulary key")
        if not isinstance(token_hex, str):
            raise ValueError("Tokenizer vocabulary values must be hex strings")
        if token_id in vocab:
            raise ValueError(f"Duplicate tokenizer vocabulary ID: {token_id}")
        vocab[token_id] = bytes.fromhex(token_hex)

    merges: list[tuple[tuple[int, int], int]] = []
    for raw_entry in merges_data:
        entry = _as_object(raw_entry, "Tokenizer merge entry")
        merges.append(
            (
                (_require_int(entry, "left"), _require_int(entry, "right")),
                _require_int(entry, "new_id"),
            )
        )

    special_tokens = {
        token: _require_exact_int(token_id, f"Special-token ID for {token!r}")
        for token, token_id in special_tokens_data.items()
    }

    _validate_special_token_ids(vocab, special_tokens)

    manifest = TokenizerManifest(
        name=_require_str(manifest_data, "name"),
        version=_require_str(manifest_data, "version"),
        tokenizer_type=_require_str(manifest_data, "tokenizer_type"),
        encoding=_require_str(manifest_data, "encoding"),
        base_vocab_size=_require_int(manifest_data, "base_vocab_size"),
        target_vocab_size=_require_int(manifest_data, "target_vocab_size"),
        actual_bpe_vocab_size=_require_int(manifest_data, "actual_bpe_vocab_size"),
        special_token_count=_require_int(manifest_data, "special_token_count"),
        min_pair_frequency=_require_int(manifest_data, "min_pair_frequency"),
    )

    return AsterTokenizer(
        model=BPEModel(merges=merges, vocab=vocab),
        special_tokens=special_tokens,
        manifest=manifest,
    )


def _validate_special_token_ids(
    vocab: dict[int, bytes],
    special_tokens: dict[str, int],
) -> None:
    vocab_ids = set(vocab)
    special_ids = list(special_tokens.values())

    if len(special_ids) != len(set(special_ids)):
        raise ValueError("Special-token IDs must be unique")

    if vocab_ids.intersection(special_ids):
        raise ValueError("Special-token IDs must not collide with BPE vocabulary IDs")


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_object(path: Path) -> dict[str, object]:
    return _as_object(_read_json(path), str(path))


def _read_array(path: Path) -> list[object]:
    value = _read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return cast(list[object], value)


def _as_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    if key not in data:
        raise ValueError(f"Missing required integer field: {key}")
    return _require_exact_int(data[key], key)


def _require_exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return cast(int, value)


def _parse_token_id(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit() or (len(value) > 1 and value.startswith("0")):
        raise ValueError(f"{label} must be a canonical non-negative integer string")
    return int(value)
