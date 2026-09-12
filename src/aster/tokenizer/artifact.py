import json
from pathlib import Path

from aster.tokenizer.bpe import BPEModel


def save_bpe(model: BPEModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merges_data = [
        {
            "left": pair[0],
            "right": pair[1],
            "new_id": new_id,
        }
        for pair, new_id in model.merges
    ]

    vocab_data = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in model.vocab.items()
    }

    (output_dir / "merges.json").write_text(
        json.dumps(merges_data, indent=2),
        encoding="utf-8",
    )

    (output_dir / "vocab.json").write_text(
        json.dumps(vocab_data, indent=2),
        encoding="utf-8",
    )

def load_bpe(input_dir: Path) -> BPEModel:
    merges_data = json.loads(
        (input_dir / "merges.json").read_text(
            encoding="utf-8",
        )
    )

    vocab_data = json.loads(
        (input_dir / "vocab.json").read_text(
            encoding="utf-8",
        )
    )

    merges = [
        (
            (item["left"], item["right"]),
            item["new_id"],
        )
        for item in merges_data
    ]

    vocab = {
        int(token_id): bytes.fromhex(token_hex)
        for token_id, token_hex in vocab_data.items()
    }

    return BPEModel(
        merges=merges,
        vocab=vocab,
    )