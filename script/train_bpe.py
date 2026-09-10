from pathlib import Path

from aster.corpus.loader import load_corpus
from aster.tokenizer.bpe import train_bpe


corpus_dir = Path("data/samples")

texts = load_corpus(corpus_dir)

model = train_bpe(
    texts=texts,
    target_vocab_size=320,
    min_pair_frequency=2,
)

print("documents:", len(texts))
print("merges:", len(model.merges))
print("vocab size:", len(model.vocab))

print()
print("first 20 learned tokens")

for pair, token_id in model.merges[:20]:
    token_bytes = model.vocab[token_id]

    print(
        token_id,
        pair,
        repr(token_bytes),
    )