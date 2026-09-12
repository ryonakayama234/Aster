import json

from aster.corpus.canonicalize import canonicalize_directory


def test_canonicalize_directory_writes_deterministic_jsonl(tmp_path):
    raw_dir = tmp_path / "raw"
    nested_dir = raw_dir / "nested"
    nested_dir.mkdir(parents=True)

    (raw_dir / "b.txt").write_text("こんにちは 🚀", encoding="utf-8")
    (raw_dir / "a.txt").write_text("Aster tokenizer", encoding="utf-8")
    (nested_dir / "c.txt").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (raw_dir / "ignored.md").write_text("not part of v0", encoding="utf-8")

    output_path = tmp_path / "canonical" / "corpus.jsonl"

    count = canonicalize_directory(raw_dir, output_path)
    first_output = output_path.read_text(encoding="utf-8")

    assert count == 3

    records = [json.loads(line) for line in first_output.splitlines()]
    assert [record["source"] for record in records] == [
        "a.txt",
        "b.txt",
        "nested/c.txt",
    ]
    assert records[0]["text"] == "Aster tokenizer"
    assert records[1]["text"] == "こんにちは 🚀"
    assert records[2]["text"] == "def add(a, b):\n    return a + b\n"
    assert all(record["id"].startswith("doc_") for record in records)

    canonicalize_directory(raw_dir, output_path)
    second_output = output_path.read_text(encoding="utf-8")

    assert second_output == first_output


def test_canonicalize_directory_rejects_missing_raw_directory(tmp_path):
    output_path = tmp_path / "canonical" / "corpus.jsonl"

    try:
        canonicalize_directory(tmp_path / "missing", output_path)
    except NotADirectoryError:
        pass
    else:
        raise AssertionError("expected NotADirectoryError")
