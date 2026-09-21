"""保存済みTinyLMで次token分布を観察し、greedy生成する。重みは更新しない。"""
import argparse
import json
import torch
from aster.model.checkpoint import load_checkpoint


def describe_ids(tokenizer, ids):
    raw = b''.join(tokenizer.model.vocab[i] for i in ids if i in tokenizer.model.vocab)
    try:
        text, valid = raw.decode('utf-8'), True
    except UnicodeDecodeError:
        text, valid = raw.decode('utf-8', errors='replace'), False
    return {'token_ids': ids, 'text': text, 'utf8_valid': valid, 'bytes_hex': raw.hex()}


@torch.no_grad()
def inspect_next(model, tokenizer, prompt_ids, top_k=5):
    if not prompt_ids:
        raise ValueError('Prompt must contain at least BOS')
    model.eval()
    device = next(model.parameters()).device
    logits = model(torch.tensor([prompt_ids[-model.config.context_length:]], device=device))[0, -1]
    probabilities = logits.softmax(-1)
    probs, ids = probabilities.topk(min(top_k, tokenizer.vocab_size))
    special_names = {i: name for name, i in tokenizer.special_tokens.items()}
    return [{'id': i, 'probability': p, 'special': special_names.get(i),
             'bytes_hex': tokenizer.model.vocab[i].hex() if i in tokenizer.model.vocab else None}
            for i, p in zip(ids.tolist(), probs.tolist())]


@torch.no_grad()
def generate_ids(model, tokenizer, prompt_ids, max_new_tokens=32):
    if not prompt_ids or type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 4096:
        raise ValueError('Nonempty prompt and max_new_tokens in [0,4096] required')
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.config.context_length:]], device=device)
        logits = model(x)[0, -1].clone()
        # Raw top-k remains unmodified in inspect_next; only generation excludes BOS.
        logits[tokenizer.bos_id] = -torch.inf
        token = int(logits.argmax())
        generated.append(token)
        ids.append(token)
        if token == tokenizer.eos_id:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', default='')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    ids = tokenizer.encode(args.prompt, add_bos=True)
    print(json.dumps({'prompt': args.prompt, 'step': checkpoint['metadata']['step'],
                      'raw_next_token_top5': inspect_next(model, tokenizer, ids),
                      'generation_policy': 'greedy; BOS excluded; EOS stops; sliding context',
                      'continuation': describe_ids(tokenizer, generate_ids(model, tokenizer, ids, args.max_new_tokens))},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
