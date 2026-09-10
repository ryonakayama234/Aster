from dataclasses import dataclass


Pair = tuple[int, int]


@dataclass
class BPEModel:
    merges: list[tuple[Pair, int]]
    vocab: dict[int, bytes]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))

        for pair, new_id in self.merges:
            ids = merge_pair(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8")


def count_pairs(sequences: list[list[int]]) -> dict[Pair, int]:
    counts: dict[Pair, int] = {}

    for ids in sequences:
        for left, right in zip(ids, ids[1:]):
            pair = (left, right)

            if pair not in counts:
                counts[pair] = 0

            counts[pair] += 1

    return counts


def merge_pair(
    ids: list[int],
    pair: Pair,
    new_id: int,
) -> list[int]:
    result: list[int] = []
    i = 0

    while i < len(ids):
        if (
            i + 1 < len(ids)
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    return result


def train_bpe(
    texts: list[str],
    target_vocab_size: int,
    min_pair_frequency: int = 2,
) -> BPEModel:
    if target_vocab_size < 256:
        raise ValueError("target_vocab_size must be at least 256")

    sequences = [
        list(text.encode("utf-8"))
        for text in texts
    ]

    vocab = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    merges: list[tuple[Pair, int]] = []

    next_id = 256

    while next_id < target_vocab_size:
        counts = count_pairs(sequences)

        if not counts:
            break

        best_pair = min(
            counts,
            key=lambda pair: (-counts[pair], pair),
        )

        if counts[best_pair] < min_pair_frequency:
            break

        new_id = next_id

        vocab[new_id] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )

        sequences = [
            merge_pair(ids, best_pair, new_id)
            for ids in sequences
        ]

        merges.append((best_pair, new_id))

        next_id += 1

    return BPEModel(
        merges=merges,
        vocab=vocab,
    )