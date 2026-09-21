"""小さな事前学習ループ。既存view/Tokenizerを固定し、観測とcheckpointを残す。"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
from torch.nn import functional as F
from aster.corpus.pipeline import digest, json_bytes
from aster.inference.generate import describe_ids, generate_ids, inspect_next
from aster.model.checkpoint import load_checkpoint, save_checkpoint
from aster.model.tiny_lm import ModelConfig, TinyLM
from aster.records.runlog import RunLog
from aster.training.dataset import IGNORE_INDEX, collate, load_training_tokenizer, prepare_windows


@dataclass(frozen=True)
class TrainConfig:
    mode: str = 'overfit'
    context_length: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2
    batch_size: int = 8
    steps: int = 200
    eval_every: int = 50
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    seed: int = 42
    overfit_windows: int = 1
    threads: int = 2

    def __post_init__(self):
        if self.mode not in ('overfit', 'pilot'):
            raise ValueError('mode must be overfit or pilot')
        for name in ('batch_size', 'steps', 'eval_every', 'overfit_windows', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0,2**32)')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be nonnegative and finite')
        ModelConfig(256, self.context_length, self.width, self.heads, self.layers)


def mean_loss(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model, windows, eos_id, batch_size):
    if not windows:
        return None
    model.eval()
    totals = defaultdict(lambda: [0.0, 0])
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]
        x, y = collate(batch, eos_id)
        logits = model(x)
        losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=IGNORE_INDEX, reduction='none')
        for i, window in enumerate(batch):
            count = int((y[i] != IGNORE_INDEX).sum())
            value = float(losses[i].sum())
            totals[window.domain][0] += value
            totals[window.domain][1] += count
    count = sum(v[1] for v in totals.values())
    return {'loss': sum(v[0] for v in totals.values()) / count, 'target_tokens': count,
            'windows': len(windows), 'by_domain': {k: {'loss': s/n, 'target_tokens': n}
                                                for k, (s, n) in sorted(totals.items())}}


def train(root, view, tokenizer_dir, config):
    root, view = Path(root).resolve(), Path(view).resolve()
    run = RunLog(root, 'pretrain', {'view_id': view.name, 'config': asdict(config)}, producer='trainer')
    began = time.perf_counter()
    try:
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        chooser = random.Random(config.seed)
        tokenizer, payload = load_training_tokenizer(tokenizer_dir, view.name)
        tokenizer_id = digest(json_bytes(payload))
        splits = prepare_windows(view, tokenizer, config.context_length)
        windows = splits['train']
        if config.mode == 'overfit':
            windows = windows[:config.overfit_windows]
        elif not splits['dev']:
            raise ValueError('pilot requires a nonempty dev split')
        # Store exact selected windows, not just a count: overfit is a named subset.
        selected = [{'record_id': w.record_id, 'offset': w.offset, 'domain': w.domain}
                    for w in windows]
        (run.path / 'training-windows.json').write_bytes(json_bytes(selected))
        model = TinyLM(ModelConfig(tokenizer.vocab_size, config.context_length,
                                    config.width, config.heads, config.layers))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        source = Path(__file__).parents[1]
        code_files = ['training/pretrain.py', 'training/dataset.py', 'training/tokenizer_run.py',
                      'model/tiny_lm.py', 'model/transformer.py', 'model/lm_head.py',
                      'model/checkpoint.py', 'inference/generate.py', 'tokenizer/artifact.py',
                      'tokenizer/bpe.py', 'records/runlog.py']
        provenance = {'view_id': view.name, 'tokenizer_id': tokenizer_id, 'config': asdict(config),
                      'torch_version': str(torch.__version__), 'python_version': sys.version,
                      'device': 'cpu', 'parameter_count': sum(p.numel() for p in model.parameters()),
                      'code_sha256': {p: digest((source / p).read_bytes()) for p in code_files}}
        (run.path / 'experiment.json').write_bytes(json_bytes(provenance))
        run.event('prepared', {**provenance, 'train_windows': len(windows),
                              'dev_windows': len(splits['dev']), 'test_used': False})
        prompt = list(windows[0].ids[:min(8, len(windows[0].ids)-1)])
        observations = []
        tokens_seen = 0
        order, cursor = [], 0
        for step in range(config.steps + 1):
            if step > 0:
                # Shuffle once per pass and include the tail: no dropped short batch.
                if cursor >= len(order):
                    order = list(range(len(windows)))
                    chooser.shuffle(order)
                    cursor = 0
                chosen = order[cursor:cursor + config.batch_size]
                cursor += len(chosen)
                x, y = collate([windows[i] for i in chosen], tokenizer.eos_id)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = mean_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                tokens_seen += int((y != IGNORE_INDEX).sum())
                run.event('step', {'step': step, 'batch_loss': float(loss.detach()),
                                   'gradient_norm': float(norm), 'tokens_seen': tokens_seen})
            if step % config.eval_every == 0 or step == config.steps:
                measured = {'step': step, 'tokens_seen': tokens_seen,
                            'train': evaluate(model, windows, tokenizer.eos_id, config.batch_size),
                            'dev': evaluate(model, splits['dev'], tokenizer.eos_id, config.batch_size)
                                   if config.mode == 'pilot' else None,
                            'prompt': describe_ids(tokenizer, prompt),
                            'raw_next_token_top5': inspect_next(model, tokenizer, prompt),
                            'continuation': describe_ids(tokenizer, generate_ids(model, tokenizer, prompt)),
                            'generation_policy': 'greedy; BOS excluded; EOS stops; sliding context'}
                checkpoint = run.path / f'checkpoint-{step:06d}.pt'
                save_checkpoint(checkpoint, model, payload, {**provenance, 'step': step, 'tokens_seen': tokens_seen})
                measured['checkpoint'] = checkpoint.name
                measured['checkpoint_sha256'] = digest(checkpoint.read_bytes())
                observations.append(measured)
                run.event('evaluated', measured)
                # Future UI can read actual observations; current BPE UI does not accept this schema yet.
                bundle = {'schema_version': 'aster-training-bundle-0', 'run_id': run.id,
                          'status': 'running', 'experiment': provenance, 'observations': observations,
                          'evaluation_scope': 'all selected train windows; all dev windows in pilot; no test',
                          'note': 'overfit measures memorization, not generalization. No Agent trained.'}
                (run.path / 'training-bundle.json').write_bytes(json_bytes(bundle))
                print(f"step={step} train_loss={measured['train']['loss']:.4f} tokens_seen={tokens_seen}", flush=True)
        reloaded, _, _ = load_checkpoint(checkpoint, tokenizer_id)
        x, _ = collate(windows[:config.batch_size], tokenizer.eos_id)
        model.eval()
        with torch.no_grad():
            if not torch.equal(model(x), reloaded(x)):
                raise ValueError('Reloaded checkpoint predictions differ')
        bundle.update(status='completed', reload_exact_match=True)
        (run.path / 'training-bundle.json').write_bytes(json_bytes(bundle))
        run.finish('completed', checkpoint=str(checkpoint), seconds=time.perf_counter()-began,
                   reload_exact_match=True, train_loss=observations[-1]['train']['loss'])
        return run.path
    except BaseException as error:
        run.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error_type=type(error).__name__, error=str(error))
        path = run.path / 'training-bundle.json'
        if path.exists():
            data = json.loads(path.read_text())
            data.update(status=run.summary['status'], error=str(error))
            path.write_bytes(json_bytes(data))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/tinylm-overfit-v0.json'))
    args = parser.parse_args()
    config = TrainConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    print(train(args.root, args.view, args.tokenizer, config))


if __name__ == '__main__':
    main()
