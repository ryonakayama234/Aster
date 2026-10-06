import json
import importlib.util
from pathlib import Path

import pytest

from aster.corpus.pipeline import digest, json_bytes
from aster.tokenizer.artifact import save_tokenizer, train_aster_tokenizer
from aster.training.dataset import load_training_tokenizer


ROOT = Path(__file__).resolve().parents[2]


def make_lang_artifact(tmp_path, view_id="view-123"):
    tokenizer = train_aster_tokenizer(
        ["こんにちは Aster。こんにちは Aster。"],
        target_vocab_size=280,
        version="lang-v0",
    )
    artifact = tmp_path / "artifact"
    save_tokenizer(tokenizer, artifact)

    provenance = {
        "schema_version": "aster-tokenizer-provenance-0",
        "training_view_id": view_id,
        "fit_split": "train",
        "sealed_test_used_for_fit": False,
    }
    (artifact / "provenance.json").write_bytes(json_bytes(provenance))

    (artifact / "audit.json").write_bytes(json_bytes({"test_audit": True}))

    files = {
        path.name: digest(path.read_bytes())
        for path in sorted(artifact.iterdir())
        if path.is_file()
    }
    identity = {
        "schema_version": "aster-tokenizer-artifact-0",
        "training_view_id": view_id,
        "files": files,
    }
    (artifact / "artifact.json").write_bytes(json_bytes(identity))
    destination = tmp_path / digest((artifact / "artifact.json").read_bytes())
    artifact.rename(destination)
    return destination


def test_lang_v0_tokenizer_artifact_loads_for_matching_view(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    tokenizer, payload = load_training_tokenizer(artifact, "view-123")

    assert tokenizer.vocab_size == len(payload["vocab"]) + 2
    probe = "こんにちは Aster。"
    assert tokenizer.decode(tokenizer.encode(probe)) == probe


def test_lang_v0_tokenizer_artifact_rejects_wrong_view(tmp_path):
    artifact = make_lang_artifact(tmp_path)

    with pytest.raises(ValueError, match="training view"):
        load_training_tokenizer(artifact, "other-view")


def test_lang_v0_tokenizer_artifact_detects_tampering(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    vocab = artifact / "vocab.json"
    vocab.write_text(vocab.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file hash mismatch"):
        load_training_tokenizer(artifact, "view-123")


def test_lang_v0_wiring_spec_pins_gate1_outputs_and_overfit_config():
    spec = json.loads((ROOT / "configs/lang-v0-wiring.json").read_text(encoding="utf-8"))

    assert spec["training_view_id"] == (
        "6dd51f52b903829b8548686743acda1dd33ccd6d3a4dfefc8abd14deb6e9ff4d"
    )
    assert spec["tokenizer_artifact_id"] == (
        "37e0b82f8e80f90a84346fd37d00b680669174388cf94f6cd88fa5b76cc13513"
    )
    assert spec["tiny_lm_config"] == "configs/tinylm-overfit-v0.json"


@pytest.mark.parametrize('with_evaluation', [False, True])
def test_missing_identity_is_rejected_even_with_legacy_evaluation(tmp_path, with_evaluation):
    artifact = make_lang_artifact(tmp_path)
    (artifact / 'artifact.json').unlink()
    if with_evaluation:
        (artifact / 'evaluation.json').write_bytes(json_bytes({'experiment': {'view_id': 'view-123'}}))
    with pytest.raises(ValueError, match='requires artifact.json'):
        load_training_tokenizer(artifact, 'view-123')


def test_identity_must_match_directory_and_expected_id(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    with pytest.raises(ValueError, match='expected artifact ID'):
        load_training_tokenizer(artifact, 'view-123', expected_artifact_id='wrong')
    renamed = artifact.rename(tmp_path / 'wrong-directory')
    with pytest.raises(ValueError, match='identity digest'):
        load_training_tokenizer(renamed, 'view-123')


def test_joint_tokenizer_and_identity_rewrite_rejected_under_old_id(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    # Replace with another valid tokenizer and refresh all matching file hashes.
    tokenizer = train_aster_tokenizer(['別の文章。別の文章。'], target_vocab_size=280, version='lang-v0')
    save_tokenizer(tokenizer, artifact)
    identity_path = artifact / 'artifact.json'
    identity = json.loads(identity_path.read_bytes())
    identity['files'] = {name: digest((artifact / name).read_bytes()) for name in identity['files']}
    identity_path.write_bytes(json_bytes(identity))
    with pytest.raises(ValueError, match='identity digest'):
        load_training_tokenizer(artifact, 'view-123', expected_artifact_id=artifact.name)


def test_legacy_evaluation_remains_supported_but_cannot_satisfy_pinned_lang_id(tmp_path):
    artifact = make_lang_artifact(tmp_path)
    (artifact / 'provenance.json').unlink()
    (artifact / 'artifact.json').unlink()
    (artifact / 'evaluation.json').write_bytes(json_bytes({'experiment': {'view_id': 'view-123'}}))
    load_training_tokenizer(artifact, 'view-123')
    with pytest.raises(ValueError, match='requires provenance.json'):
        load_training_tokenizer(artifact, 'view-123', expected_artifact_id=artifact.name)


def make_training_view(root):
    body = b'{"result":42}\n'
    sample = {'record_id': 'sample', 'split': 'train', 'domain': 'tool_json',
              'text_path': 'train/sample.txt', 'text_sha256': digest(body),
              'leakage_group': 'wiring-fixture'}
    files = {'train/sample.txt': body, 'samples.jsonl': (json.dumps(sample) + '\n').encode()}
    manifest = json_bytes({'schema_version': 'aster-training-view-0',
                           'files': {name: digest(data) for name, data in files.items()}})
    view = root / 'data/training' / digest(manifest)
    for name, data in files.items():
        path = view / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (view / 'manifest.json').write_bytes(manifest)
    return view


def test_train_records_distinct_artifacts_with_identical_payload(tmp_path):
    from aster.model.checkpoint import load_checkpoint
    from aster.training.pretrain import TrainConfig, train

    view = make_training_view(tmp_path)
    first = make_lang_artifact(tmp_path, view.name)
    second = tmp_path / 'second'
    import shutil
    shutil.copytree(first, second)
    (second / 'audit.json').write_bytes(json_bytes({'test_audit': 'different'}))
    identity = json.loads((second / 'artifact.json').read_bytes())
    identity['files']['audit.json'] = digest((second / 'audit.json').read_bytes())
    (second / 'artifact.json').write_bytes(json_bytes(identity))
    second = second.rename(tmp_path / digest((second / 'artifact.json').read_bytes()))
    assert first.name != second.name

    experiments = []
    for artifact in (first, second):
        spec_hash = digest(json_bytes({'tokenizer_artifact_id': artifact.name}))
        run = train(tmp_path, view, artifact,
                    TrainConfig(context_length=16, width=16, heads=2, layers=1, steps=1, eval_every=1),
                    tokenizer_artifact_id=artifact.name, wiring_spec_sha256=spec_hash)
        experiment = json.loads((run / 'experiment.json').read_bytes())
        summary = json.loads((run / 'run.json').read_bytes())
        bundle = json.loads((run / 'training-bundle.json').read_bytes())
        _, _, checkpoint = load_checkpoint(run / 'checkpoint-000001.pt')
        for record in (experiment, summary['inputs'], bundle['experiment'], checkpoint['metadata']):
            assert record['tokenizer_artifact_id'] == artifact.name
            assert record['wiring_spec_sha256'] == spec_hash
        assert bundle['reload_exact_match'] is True
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        assert next(e for e in events if e['kind'] == 'prepared')['data']['test_used'] is False
        experiments.append(experiment)
    assert experiments[0]['tokenizer_id'] == experiments[1]['tokenizer_id']


def test_failed_input_validation_preserves_selected_ids(tmp_path):
    from aster.training.pretrain import TrainConfig, train

    artifact = make_lang_artifact(tmp_path)
    (artifact / 'artifact.json').unlink()
    spec_hash = digest(b'wiring spec fixture')
    with pytest.raises(ValueError, match='requires artifact.json'):
        train(tmp_path, tmp_path / 'view-123', artifact, TrainConfig(steps=1),
              tokenizer_artifact_id=artifact.name, wiring_spec_sha256=spec_hash)
    summary, = [json.loads(p.read_bytes()) for p in (tmp_path / 'runs').glob('*/run.json')]
    assert summary['status'] == 'failed'
    assert summary['inputs']['tokenizer_artifact_id'] == artifact.name
    assert summary['inputs']['wiring_spec_sha256'] == spec_hash


def test_runner_passes_digest_of_exact_parsed_spec_bytes(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location('lang_wiring_runner', ROOT / 'scripts/run_lang_v0_wiring.py')
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    spec = json.loads((ROOT / 'configs/lang-v0-wiring.json').read_bytes())
    raw = json.dumps(spec, indent=4).encode() + b'\n\n'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/lang-v0-wiring.json').write_bytes(raw)
    (tmp_path / spec['tiny_lm_config']).write_bytes((ROOT / spec['tiny_lm_config']).read_bytes())
    (tmp_path / 'data/training' / spec['training_view_id']).mkdir(parents=True)
    (tmp_path / 'artifacts/tokenizers/lang-v0' / spec['tokenizer_artifact_id']).mkdir(parents=True)
    received = {}

    def capture_train(root, view, tokenizer, config, **kwargs):
        received.update(kwargs)
        return tmp_path / 'test-run'

    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'train', capture_train)
    runner.main()
    assert received == {'tokenizer_artifact_id': spec['tokenizer_artifact_id'],
                        'wiring_spec_sha256': digest(raw)}


@pytest.mark.parametrize('mutation,message', [
    ('omit_vocab_hash', 'missing required file hashes'),
    ('unsafe_file_name', 'invalid file name'),
    ('test_fitting', 'reports test fitting'),
])
def test_invalid_lang_identity_or_provenance_is_rejected(tmp_path, mutation, message):
    artifact = make_lang_artifact(tmp_path)
    identity = json.loads((artifact / 'artifact.json').read_bytes())
    if mutation == 'omit_vocab_hash':
        del identity['files']['vocab.json']
    elif mutation == 'unsafe_file_name':
        identity['files']['../outside'] = digest(b'outside')
    else:
        provenance = json.loads((artifact / 'provenance.json').read_bytes())
        provenance['sealed_test_used_for_fit'] = True
        (artifact / 'provenance.json').write_bytes(json_bytes(provenance))
        identity['files']['provenance.json'] = digest((artifact / 'provenance.json').read_bytes())
    (artifact / 'artifact.json').write_bytes(json_bytes(identity))
    artifact = artifact.rename(tmp_path / digest((artifact / 'artifact.json').read_bytes()))
    with pytest.raises(ValueError, match=message):
        load_training_tokenizer(artifact, 'view-123')
