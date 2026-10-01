"""Reject corrupted cached scores before treating logs as evidence."""
import copy
import math
import pytest
pytest.importorskip('torch')
from aster.benchmark.decision_failure import validate_cached_scores
from aster.training.decision_data import anchor_cases


def valid_row():
    example = anchor_cases()[0].example
    logits = [0.0] * len(example.candidates)
    logits[example.target_index] = 2.0
    nll = math.log(math.exp(2) + len(logits) - 1) - 2
    return example, dict(candidates=[a.to_dict() for a in example.candidates],
        target_index=example.target_index, logits=logits, model_action=example.target.to_dict(),
        nll=nll, target_probability=math.exp(-nll), target_margin=2.0)


def test_valid_cached_scores_and_stable_large_logits():
    example, row = valid_row()
    validate_cached_scores(row, example)
    row['logits'] = [v + 10000 for v in row['logits']]
    validate_cached_scores(row, example)


@pytest.mark.parametrize('mutation', ['action', 'order', 'index', 'nan', 'short', 'nll', 'probability', 'margin'])
def test_corrupted_cache_is_rejected(mutation):
    example, source = valid_row()
    row = copy.deepcopy(source)
    if mutation == 'action':
        row['model_action'] = example.candidates[(example.target_index + 1) % len(example.candidates)].to_dict()
    elif mutation == 'order': row['candidates'].reverse()
    elif mutation == 'index': row['target_index'] = -1
    elif mutation == 'nan': row['logits'][0] = float('nan')
    elif mutation == 'short': row['logits'].pop()
    else: row[{'probability':'target_probability', 'margin':'target_margin'}.get(mutation, mutation)] += .1
    with pytest.raises(ValueError): validate_cached_scores(row, example)
