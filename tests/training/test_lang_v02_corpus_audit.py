"""Synthetic fixtures only. No private corpus, Run or sealed test opened."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def runner():
    spec = importlib.util.spec_from_file_location(
        "lang_v02_corpus_audit", ROOT / "scripts/audit_lang_v02_corpus.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ToyTokenizer:
    bos_id = 256
    eos_id = 257
    def encode(self, value, *, add_bos=False, add_eos=False):
        ids = list(value.encode("utf-8"))
        return ([self.bos_id] if add_bos else []) + ids + (
            [self.eos_id] if add_eos else [])
    def decode(self, ids):
        return bytes(token for token in ids if token < 256).decode("utf-8")


def fixture(tmp_path):
    r = runner()
    view = tmp_path / "view"
    (view / "train").mkdir(parents=True)
    text = ("Leanについて述べる。[3][3] URL https://test.invalid\n"
            "説明文です。" * 4)
    data = text.encode()
    (view / "train/first.txt").write_bytes(data)
    rows = [
        {"split": "train", "record_id": "first", "text_path": "train/first.txt",
         "text_sha256": hashlib.sha256(data).hexdigest(),
         "raw_path": "data/raw/pool/prose/ja/003.txt",
         "domain": "prose", "group_id": "article-lean-trust",
         "leakage_group": "article-lean-trust"},
        {"split": "test", "record_id": "sealed", "text_path": "test/sealed.txt",
         "text_sha256": "e" * 64},
    ]
    manifest = {"files": {"train/first.txt": hashlib.sha256(data).hexdigest(),
                          "test/sealed.txt": "e"*64}}
    return r, view, rows, manifest, text, data


def test_marker_counts_are_literal_not_deletion_policy():
    r = runner()
    m = r.marker_counts("Lean clean [3][3] [24] [9999] https://x/ " +
                        chr(96)*3 + "\x01")
    assert m["lean_case_insensitive_substring"] == 2
    assert m["bracket_3"] == 2
    assert m["numbered_brackets"] == 3
    assert m["code_fences"] == 1
    assert m["url_prefixes"] == 1
    assert m["control_characters"] == 1


def test_train_scan_never_opens_sealed_test(tmp_path, monkeypatch):
    r, view, rows, manifest, text, data = fixture(tmp_path)
    original = r._in_dir
    def guarded(base, name):
        assert not name.startswith("test/"), "sealed test accessed"
        return original(base, name)
    monkeypatch.setattr(r, "_in_dir", guarded)
    result = r.scan_train_view(
        view, rows, manifest, ToyTokenizer(), 128,
        {"prose/ja/003.txt": len(data)+5})
    assert result["train_documents"] == 1
    assert result["totals"]["raw_original_bytes"] == len(data)+5
    assert result["totals"]["extracted_train_bytes"] == len(data)
    assert result["totals"]["bpe_text_tokens"] == len(data)
    assert result["totals"]["bpe_targets_including_eos"] == len(data)+1
    assert result["totals"]["training_windows"] == (len(data)+1+127)//128
    assert result["lean_source_text_token_share"] == 1
    assert result["totals"]["bracket_3"] == 8


def test_mutation_of_train_snapshot_rejected(tmp_path):
    r, view, rows, manifest, text, data = fixture(tmp_path)
    (view / "train/first.txt").write_bytes(b"malicious")
    with pytest.raises(ValueError, match="hash mismatch"):
        r.scan_train_view(view, rows, manifest, ToyTokenizer(), 128,
                          {"prose/ja/003.txt": len(data)})


def test_no_inferred_source_and_context(tmp_path):
    r, view, rows, manifest, text, data = fixture(tmp_path)
    with pytest.raises(ValueError, match="source absent"):
        r.scan_train_view(view, rows, manifest, ToyTokenizer(), 128, {})
    with pytest.raises(ValueError, match="context"):
        r.scan_train_view(view, rows, manifest, ToyTokenizer(), 0, {})


def test_shingle_duplicates_are_candidates_only():
    r = runner()
    duplicated = "明るい空と深い森。" * 25
    other = "山の向こうに雲が流れる。" * 25
    result = r.near_duplicate_candidates([
        ("a", duplicated), ("b", duplicated), ("c", other)])
    assert result == [{"record_ids": ["a", "b"], "jaccard": 1.0}]


def test_runner_requires_exact_reviewed_clean_checkout(tmp_path, monkeypatch):
    r = runner()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    path = tmp_path / "scripts/audit_lang_v02_corpus.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic audit\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args],
                              check=True, capture_output=True).stdout.decode().strip()
    git("init", "-q")
    git("add", ".")
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.com", "commit", "-qm",
                    "synthetic"], check=True, capture_output=True)
    commit = git("rev-parse", "HEAD")
    assert r.verify_checkout(tmp_path, commit) == hashlib.sha256(b"synthetic audit\n").hexdigest()
    with pytest.raises(ValueError, match="Git HEAD"):
        r.verify_checkout(tmp_path, "f"*40)
    with pytest.raises(ValueError, match="40-hex"):
        r.verify_checkout(tmp_path, "HEAD")
    path.write_text("modified")
    with pytest.raises(ValueError, match="not clean"):
        r.verify_checkout(tmp_path, commit)


def test_private_output_contract_uses_exclusive_directory_and_modes(tmp_path):
    r = runner()
    # Simulate the exact exclusive directory / private file boundary without
    # needing any real local corpus data or touching an existing Run.
    target = tmp_path / r.OUTPUT_DIR
    target.mkdir(mode=0o700, exist_ok=False)
    output = target / "private-report.json"
    import os
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        target.mkdir(mode=0o700, exist_ok=False)
