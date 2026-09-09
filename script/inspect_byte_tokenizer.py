from pathlib import Path

from aster.tokenizer.byte import encode, decode


SAMPLES_DIR = Path("data/samples")


for path in sorted(SAMPLES_DIR.iterdir()):
    if not path.is_file():
        continue

    if path.suffix not in {".txt", ".py", ".jsonl"}:
        continue

    text = path.read_text(encoding="utf-8")

    ids = encode(text)
    restored = decode(ids)

    char_count = len(text)
    token_count = len(ids)

    if char_count == 0:
        tokens_per_char = 0.0
    else:
        tokens_per_char = token_count / char_count

    assert restored == text

    print(f"{path.name}")
    print(f"  characters      : {char_count}")
    print(f"  byte tokens     : {token_count}")
    print(f"  tokens/character: {tokens_per_char:.2f}")
    print(f"  first 20 ids    : {ids[:20]}")
    print()