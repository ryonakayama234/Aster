"""重みとその意味を決める設定・Tokenizerを一緒に保存する。再開学習は未対応。"""
from dataclasses import asdict
from pathlib import Path
import torch
from aster.corpus.pipeline import digest, json_bytes
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.training.dataset import tokenizer_from_payload

SCHEMA = 'aster-tinylm-checkpoint-0'


def save_checkpoint(path, model, tokenizer, metadata):
    path = Path(path)
    payload = {'schema_version': SCHEMA, 'model_config': asdict(model.config),
               'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               'tokenizer': tokenizer, 'tokenizer_id': digest(json_bytes(tokenizer)),
               'metadata': metadata}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, expected_tokenizer_id=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema_version'] != SCHEMA:
        raise ValueError('Unsupported checkpoint schema')
    identity = digest(json_bytes(payload['tokenizer']))
    if identity != payload['tokenizer_id'] or (expected_tokenizer_id and identity != expected_tokenizer_id):
        raise ValueError('Checkpoint tokenizer identity mismatch')
    tokenizer = tokenizer_from_payload(payload['tokenizer'])
    config = ModelConfig(**payload['model_config'])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError('Checkpoint vocabulary size mismatch')
    # Loading a model must not change the caller's training random stream.
    with torch.random.fork_rng(devices=[]):
        model = TinyLM(config)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return model, tokenizer, payload
