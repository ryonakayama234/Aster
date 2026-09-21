"""文脈ベクトルを、次tokenの各候補の点数(logits)へ写す。"""
from torch import nn


class LMHead(nn.Module):
    def __init__(self, width: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(width, vocab_size, bias=False)

    def forward(self, hidden):
        # softmaxはここでは使わない。cross_entropyが数値的に安定して処理する。
        return self.projection(hidden)
