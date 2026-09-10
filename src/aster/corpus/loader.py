from pathlib import Path


def load_corpus(directory: Path) -> list[str]:
    texts: list[str] = []

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")
        texts.append(text)

    return texts