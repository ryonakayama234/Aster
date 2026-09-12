import pytest

from aster.tokenizer.artifact import load_tokenizer
from aster.tokenizer.build import build_tokenizer_v0_1, load_canonical_texts


def test_load_canonical_texts_reads_txt_files_in_sorted_order(tmp_path):
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()

    (canonical_dir / "b.txt").write_text("second", encoding="utf-8")
    (canonical_dir / "a.txt").write_text("first", encoding="utf-8")
    (canonical_dir / "ignore.json").write_text("{}", encoding="utf-8")

    texts = load_canonical_texts(canonical_dir)

    assert texts == ["first", "second"]


def test_load_canonical_texts_requires_txt_files(tmp_path):
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()

    with pytest.raises(ValueError):
        load_canonical_texts(canonical_dir)


def test_build_tokenizer_v0_1_writes_loadable_artifact(tmp_path):
    canonical_dir = tmp_path / "canonical"
    artifact_dir = tmp_path / "AsterTokenizer-v0.1"
    canonical_dir.mkdir()

    (canonical_dir / "ja.txt").write_text(
        "こんにちは Aster。こんにちは Aster。",
        encoding="utf-8",
    )
    (canonical_dir / "code.txt").write_text(
        "def hello():\n    return 'hello'\n" * 3,
        encoding="utf-8",
    )

    built = build_tokenizer_v0_1(
        canonical_dir,
        artifact_dir,
        target_vocab_size=300,
    )
    loaded = load_tokenizer(artifact_dir)

    sample = "こんにちは Aster。"

    assert loaded.encode(sample) == built.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample
    assert loaded.manifest.version == "0.1"
