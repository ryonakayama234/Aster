"""Synthetic-only tests: strict UTF-8 decoding without private corpus or retraining."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest
import torch

from aster.inference.generate import generate_ids
from aster.inference.utf8_greedy import (
    Utf8Generation, extend_utf8_prefix, generate_utf8_greedy,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("pending,payload,expected", [
    (b"", b"\xe3", b"\xe3"),
    (b"\xe3", b"\x81", b"\xe3\x81"),
    (b"\xe3\x81", b"\x82", b""),
    (b"", "\u3042A".encode("utf-8"), b""),
    (b"", b"\xe3\x81", b"\xe3\x81"),
    (b"\xe3\x81", b"\x82A\xe3", b"\xe3"),
    (b"", b"\x80", None),              # stray continuation
    (b"\xe3", b"A", None),             # continuation expected
    (b"", b"\xc0\xaf", None),          # overlong
    (b"", b"\xe0\x80\x80", None),      # overlong 3-byte
    (b"", b"\xed\xa0\x80", None),      # surrogate
    (b"", b"\xf4\x90\x80\x80", None),  # > U+10FFFF
    (b"", b"\xf5\x80\x80\x80", None),  # invalid lead
    (b"", b"", None),                  # empty BPE payload
])
def test_prefix_transition(pending, payload, expected):
    assert extend_utf8_prefix(pending, payload) == expected


class FrozenSyntheticModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(context_length=8)
        self.scores = scores
        self.calls = 0

    def forward(self, x):
        row = torch.tensor(self.scores[min(self.calls, len(self.scores)-1)],
                           dtype=torch.float32)
        self.calls += 1
        return row.repeat(x.shape[0], x.shape[1], 1)


def fake_tokenizer():
    return SimpleNamespace(
        model=SimpleNamespace(vocab={
            0: b"\xe3", 1: b"\x81", 2: b"\x82",
            3: b"A", 4: b"\x80", 5: "\u3042Z".encode("utf-8")}),
        bos_id=6, eos_id=7)


def test_mask_rejects_stray_bytes_and_eos_inside_scalar():
    t = fake_tokenizer()
    # Stray continuation dominates at step 0; EOS is highest while a
    # Japanese codepoint is unfinished, but both must be masked out.
    scores = [
        [150, 0, 0, 100, 200, 0, 0, 1],
        [0, 150, 0, 200, 10, 0, 0, 250],
        [0, 0, 150, 200, 10, 0, 0, 250],
        [0, 0, 0, 100, 200, 0, 0, 250],
    ]
    output = generate_utf8_greedy(FrozenSyntheticModel(scores), t, [6], 4)
    assert output == Utf8Generation([0, 1, 2, 7], b"", "eos")
    assert b"".join(t.model.vocab[i] for i in output.token_ids if i in t.model.vocab).decode() == "あ"


def test_original_greedy_policy_remains_unchanged():
    t = fake_tokenizer()
    scores = [[150, 0, 0, 0, 200, 0, 0, 50]]
    assert generate_ids(FrozenSyntheticModel(scores), t, [6], max_new_tokens=1) == [4]
    assert generate_utf8_greedy(FrozenSyntheticModel(scores), t, [6], 1).token_ids == [0]


def test_multi_character_bpe_token_and_budget_pending():
    t = fake_tokenizer()
    scores = [[25, 0, 0, 0, 100, 200, 0, 1]]
    gen = generate_utf8_greedy(FrozenSyntheticModel(scores), t, [6], 1)
    assert gen == Utf8Generation([5], b"", "budget")
    scores = [[200, 0, 0, 0, 100, 0, 0, 1]]
    incomplete = generate_utf8_greedy(FrozenSyntheticModel(scores), t, [6], 1)
    assert incomplete == Utf8Generation([0], b"\xe3", "budget")
    assert not b"\xe3".decode("utf-8", errors="replace").encode() == b"\xe3"


def runner():
    spec = importlib.util.spec_from_file_location(
        "lang_v01_runner", ROOT / "scripts/run_lang_v01_utf8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_original_gate4(run: Path, r, report):
    steps = (0, 200)
    pairs = []
    for step in steps:
        expected = r.EXPECTED_OLD_COUNTS[step]
        results = []
        for i, p in enumerate(r.gate4.PROMPTS):
            results.append({
                "prompt_index": i, "prompt": p,
                "utf8_valid": False,
                "repeated_token_trigram": step == 200,
                "train_exact_32_byte_match": False,
                "ended_with_eos": step == 0 and i == 0,
                "prompt_in_train_window": False,
            })
        assert len(results) == expected["outputs"]
        pairs.append({"step": step,
                      "checkpoint_sha256": report["checkpoint_evaluations"][step // 50]["checkpoint_sha256"],
                      "results": results})
    data = {
        "schema_version": "aster-lang-gate4-private-0",
        "run_id": r.gate4.RUN_ID, "gate3_report_sha256": r.gate4.REPORT_SHA256,
        "policy": r.gate4.POLICY, "max_new_tokens": r.gate4.MAX_NEW_TOKENS,
        "prompts": list(r.gate4.PROMPTS), "checkpoints": pairs,
    }
    target = run / r.gate4.PRIVATE_OUTPUT
    target.write_bytes(json.dumps(data, ensure_ascii=False).encode())
    return target


def test_original_baseline_is_immutable_and_count_locked(tmp_path):
    r = runner()
    report = {"checkpoint_evaluations": [
        {"step": s, "checkpoint_sha256": "a"*64} for s in range(0, 201, 50)]}
    p = write_original_gate4(tmp_path, r, report)
    digest, counts = r.read_immutable_baseline(tmp_path, report)
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    assert counts["200"]["repeated_trigram"] == 6
    data = json.loads(p.read_bytes())
    data["checkpoints"][1]["results"][0]["utf8_valid"] = True
    p.write_text(json.dumps(data, ensure_ascii=False))
    with pytest.raises(ValueError, match="aggregate counters"):
        r.read_immutable_baseline(tmp_path, report)


def test_exact_reviewed_commit_and_source_integrity(tmp_path):
    r = runner()
    for name, contents in [
        ("scripts/run_lang_v01_utf8.py", b"trusted runner\n"),
        ("src/aster/inference/utf8_greedy.py", b"trusted decoder\n"),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True).stdout.decode().strip()
    git("init", "-q")
    git("add", ".")
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.com", "commit", "-qm", "synthetic"],
                   check=True, capture_output=True)
    commit = git("rev-parse", "HEAD")
    checked = r.verify_reviewed_checkout(tmp_path, commit)
    assert checked["scripts/run_lang_v01_utf8.py"] == hashlib.sha256(b"trusted runner\n").hexdigest()
    assert len(checked) == 2
    with pytest.raises(ValueError, match="differs from exact reviewed commit"):
        r.verify_reviewed_checkout(tmp_path, "f"*40)
    with pytest.raises(ValueError, match="exact 40-hex"):
        r.verify_reviewed_checkout(tmp_path, "HEAD")
    (tmp_path / "src/aster/inference/utf8_greedy.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="dirty"):
        r.verify_reviewed_checkout(tmp_path, commit)


def test_read_only_pair_run_no_private_stdout_no_overwrite(tmp_path, monkeypatch, capsys):
    r = runner()
    run = tmp_path / "runs" / r.gate4.RUN_ID
    run.mkdir(parents=True)
    conf = {"threads": 2}
    experiment = {"config": conf, "tokenizer_id": "payload",
                  "tokenizer_artifact_id": "artifact", "code_sha256": {}}
    (run / "experiment.json").write_text(json.dumps(experiment))
    (run / "run.json").write_text(json.dumps({
        "run_id": r.gate4.RUN_ID, "status": "completed",
        "inputs": {"tokenizer_artifact_id": "artifact"}}))
    (run / "training-bundle.json").write_text(json.dumps({
        "run_id": r.gate4.RUN_ID, "status": "completed", "experiment": experiment}))
    records = [{"step": s, "tokens_seen": s,
                "checkpoint_sha256": "a"*64} for s in range(0, 201, 50)]
    report = {"training_view_id": "view", "tokenizer_artifact_id": "artifact",
              "pilot_config": conf, "preflight": {
                  "view_id": "view", "tokenizer_payload_sha256": "payload",
                  "sealed_test_text_read": False}, "checkpoint_evaluations": records}
    original_path = write_original_gate4(run, r, report)
    tok = SimpleNamespace(model=SimpleNamespace(vocab={1: b"A"}))
    tok.encode = lambda _prompt, add_bos=True: [5, 1]
    monkeypatch.setattr(r, "verify_reviewed_checkout",
                        lambda *_args: {"scripts/run_lang_v01_utf8.py": "d"*64,
                                        "src/aster/inference/utf8_greedy.py": "e"*64})
    monkeypatch.setattr(r.gate4, "read_frozen_report", lambda *_args: report)
    monkeypatch.setattr(r.gate4, "preflight", lambda *_args: (
        {"pilot_config": conf, "tokenizer_artifact_id": "artifact"},
        tok, {"train": [SimpleNamespace()]}, report["preflight"]))
    monkeypatch.setattr(r.gate4, "verify_scoring_dependencies", lambda *_args: None)
    monkeypatch.setattr(r.gate4, "verify_run_sources", lambda *_args: None)
    monkeypatch.setattr(r.gate4, "validate_run_observations", lambda *_args: records)
    monkeypatch.setattr(r.gate4, "verified_training_chunks", lambda *_args: (b"training",))
    used = []
    def verified_model(_run, entry, *_args):
        used.append(entry["step"])
        return SimpleNamespace(), tok
    monkeypatch.setattr(r.gate4, "verify_checkpoint", verified_model)
    monkeypatch.setattr(r, "generate_utf8_greedy",
                        lambda *_args: Utf8Generation([1], b"", "budget"))
    summary = r.audit(tmp_path, "a"*40)
    assert summary["training_updates"] == 0
    assert summary["sealed_test_text_read"] is False
    assert summary["paired_prompts"] == 6 and used == [0, 200]
    assert summary["reviewed_source_commit"] == "a"*40
    assert len(summary["verified_source_sha256"]) == 2
    assert summary["utf8_constrained_by_step"]["200"]["valid_utf8"] == 6
    assert "continuation" not in json.dumps(summary)
    assert capsys.readouterr().out == ""
    saved = run / r.PRIVATE_OUTPUT
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert len(json.loads(saved.read_bytes())["checkpoints"]) == 2
    assert json.loads(saved.read_bytes())["reviewed_source_commit"] == "a"*40
    assert original_path.exists()
    before = saved.read_bytes()
    with pytest.raises(FileExistsError):
        r.audit(tmp_path, "a"*40)
    assert saved.read_bytes() == before
