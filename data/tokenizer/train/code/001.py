from pathlib import Path


def count_words(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    text = path.read_text(encoding="utf-8")
    for word in text.split():
        normalized = word.strip(".,!?()[]{}\"'").lower()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    return counts


def top_items(counts: dict[str, int], limit: int = 5) -> list[tuple[str, int]]:
    items = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return items[:limit]


if __name__ == "__main__":
    source = Path("data/example.txt")

    if source.exists():
        for word, count in top_items(count_words(source)):
            print(f"{word}: {count}")
    else:
        print(f"file not found: {source}")
