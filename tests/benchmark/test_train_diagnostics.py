"""Train diagnostics measure frozen weights and never open test."""
import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from aster.benchmark.metrics import DecisionPrediction
from aster.benchmark.suite import build_calculate_and_store_suite
from aster.training import decision_baseline as baseline
from aster.training.decision import evaluate_decisions
from aster.model.decision_artifact import load_decision_artifact


def _hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def test_train_diagnostics_and_baseline_are_same_frozen_measurement(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    suite = build_calculate_and_store_suite()
    predict = baseline._predict_cases
    seen = []

    def guarded(model, tokenizer, cases):
        assert all(case.split != 'test' for case in cases)
        seen.extend(case.split for case in cases)
        return predict(model, tokenizer, cases)

    monkeypatch.setattr(baseline, '_predict_cases', guarded)
    run = baseline.run_logged_decision_baseline(tmp_path, suite, config=baseline.DecisionBaselineConfig(
        steps=2, width=8, heads=2, layers=1, target_vocab_size=320))
    report = json.loads((run / 'baseline.json').read_text())
    rows = [json.loads(line) for line in (run / 'train-predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len(suite.cases_for('train'))
    assert all(row['split'] == 'train' and row['calibrated_probabilities'] is None for row in rows)
    final = report['train']['final']
    assert final['correct'] == sum(row['correct'] for row in rows)
    assert final['nll'] == pytest.approx(sum(row['nll'] for row in rows) / len(rows))
    for case, row in zip(suite.cases_for('train'), rows, strict=True):
        assert row['target_action'] == case.example.target.to_dict()
        assert row['predicted_action'] == row['candidates'][row['predicted_index']]
        assert row['target_probability'] == row['raw_probabilities'][row['target_index']]
        assert row['target_margin'] == pytest.approx(row['logits'][row['target_index']] - max(
            value for i, value in enumerate(row['logits']) if i != row['target_index']))
    assert (run / 'initial-train-predictions.jsonl').exists()
    artifact = run / 'model-artifact'
    model, tokenizer, _ = load_decision_artifact(artifact)
    reference = evaluate_decisions(model, tokenizer, suite.training_examples())
    assert final['accuracy'] == reference['accuracy']
    assert final['nll'] == pytest.approx(reference['nll'], abs=1e-6)

    def forbidden(*args, **kwargs):
        pytest.fail('diagnostics must not train or calibrate')

    monkeypatch.setattr(baseline, 'train_decision', forbidden)
    monkeypatch.setattr(baseline, 'train_aster_tokenizer', forbidden)
    monkeypatch.setattr(baseline, 'fit_temperature', forbidden)
    before = _hashes(artifact)
    seen.clear()
    diagnostic = baseline.run_logged_decision_train_diagnostics(tmp_path, artifact, suite)
    result = json.loads((diagnostic / 'train-diagnostics.json').read_text())
    assert set(seen) == {'train'}
    assert result['metrics'] == final
    assert result['test']['status'] == 'sealed'
    assert result['initial_train'] is None
    assert result['artifact_id'] == report['artifact_id']
    assert _hashes(artifact) == before
    assert (diagnostic / 'train-predictions.jsonl').read_bytes() == (run / 'train-predictions.jsonl').read_bytes()
    with pytest.raises(ValueError, match='lineage'):
        baseline.run_logged_decision_train_diagnostics(tmp_path, artifact, replace(suite, suite_id='wrong'))


def test_raw_nll_does_not_clip_extreme_errors():
    prediction = DecisionPrediction('x', 'train', 'tiny', 0, 1, (-1000., 0.), (0., 1.))
    assert baseline._raw_nll(prediction) == 1000.


def test_prediction_restores_mode_rng_and_weights():
    from aster.model.decision_head import DecisionModel
    from aster.model.tiny_lm import ModelConfig, TinyLM
    from aster.tokenizer.artifact import train_aster_tokenizer
    torch.set_num_threads(2)
    cases = build_calculate_and_store_suite().cases_for('train')[:1]
    tokenizer = train_aster_tokenizer(baseline._serialized_texts(cases), target_vocab_size=320,
                                     min_pair_frequency=1, name='test', version='0')
    model = DecisionModel(TinyLM(ModelConfig(tokenizer.vocab_size, 2048, 8, 2, 1)))
    model.train()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    baseline._predict_cases(model, tokenizer, cases)
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
