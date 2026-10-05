from pathlib import Path

from aster.tokenizer.artifact import train_aster_tokenizer
from aster.tokenizer.lang_v0 import load_audit_rows, merge_audit, summarize_documents


def test_load_audit_rows_never_reads_test_text(tmp_path):
    view = tmp_path / "view"
    (view / "train").mkdir(parents=True)
    (view / "dev").mkdir()
    (view / "train/train.txt").write_text("こんにちは", encoding="utf-8")
    (view / "dev/dev.txt").write_text("Asterです", encoding="utf-8")

    samples = [
        {
            "record_id": "train",
            "split": "train",
            "domain": "prose",
            "text_path": "train/train.txt",
        },
        {
            "record_id": "dev",
            "split": "dev",
            "domain": "prose",
            "text_path": "dev/dev.txt",
        },
        {
            "record_id": "test",
            "split": "test",
            "domain": "prose",
            "text_path": "test/intentionally-missing.txt",
        },
    ]

    rows, sealed = load_audit_rows(view, samples)

    assert sealed == 1
    assert {row["record_id"] for row in rows} == {"train", "dev"}


def test_summarize_documents_compares_byte_baseline_and_bpe():
    tokenizer = train_aster_tokenizer(
        ["こんにちはこんにちは", "Aster Aster Aster"],
        target_vocab_size=280,
    )
    rows = [
        {"text": "こんにちは", "split": "train", "domain": "prose"},
        {"text": "Aster", "split": "train", "domain": "prose"},
    ]

    summary = summarize_documents(rows, tokenizer, context_length=128)

    expected_bytes = sum(len(row["text"].encode("utf-8")) for row in rows)
    assert summary["documents"] == 2
    assert summary["bytes"] == expected_bytes
    assert summary["byte_baseline"]["tokens"] == expected_bytes
    assert summary["byte_baseline"]["bytes_per_token"] == 1.0
    assert summary["bpe"]["tokens"] <= expected_bytes
    assert summary["roundtrip_failures"] == 0
    assert summary["context_length"] == 128


def test_merge_audit_contains_no_plaintext_field():
    tokenizer = train_aster_tokenizer(
        ["banana banana banana", "こんにちはこんにちは"],
        target_vocab_size=280,
    )

    audit = merge_audit(tokenizer, limit=5)

    assert audit["merge_count"] == len(tokenizer.model.merges)
    for entry in audit["first_merges"] + audit["longest_merges"]:
        assert "token_hex" in entry
        assert "text" not in entry
        assert entry["byte_length"] >= 2
