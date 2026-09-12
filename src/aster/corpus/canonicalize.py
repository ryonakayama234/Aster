import argparse
import hashlib
import json
from pathlib import Path

from aster.corpus.schema import AsterRecord


def iter_text_files(raw_dir: Path):
    """Yield UTF-8 text source files in deterministic path order."""
    for path in sorted(raw_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def make_record(path: Path, raw_dir: Path) -> AsterRecord:
    """Convert one raw UTF-8 text file into Aster's canonical record."""
    text = path.read_text(encoding="utf-8")
    source = path.relative_to(raw_dir).as_posix()

    identity = f"{source}\0{text}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]

    return AsterRecord(
        id=f"doc_{digest}",
        text=text,
        source=source,
    )


def canonicalize_directory(raw_dir: Path, output_path: Path) -> int:
    """Convert raw .txt files into deterministic JSONL records."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    if not raw_dir.is_dir():
        raise NotADirectoryError(f"raw corpus directory not found: {raw_dir}")

    records = [make_record(path, raw_dir) for path in iter_text_files(raw_dir)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            line = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            )
            file.write(line + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw UTF-8 .txt files into Aster canonical JSONL."
    )
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    count = canonicalize_directory(args.raw_dir, args.output_path)
    print(f"wrote {count} records to {args.output_path}")


if __name__ == "__main__":
    main()
