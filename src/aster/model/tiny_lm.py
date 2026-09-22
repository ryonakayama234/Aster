"""学習と生成で共用する、小さな因果言語モデル。"""
from dataclasses import dataclass
import torch
from torch import nn
from aster.model.transformer import TransformerBlock
from aster.model.lm_head import LMHead


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self):
        for value in (self.vocab_size, self.context_length, self.width, self.heads, self.layers):
            if type(value) is not int or value <= 0:
                raise ValueError('Model dimensions must be positive integers')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList([TransformerBlock(config.width, config.heads)
                                     for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = LMHead(config.width, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, token_ids):
        """Return the shared contextual representation before any task-specific head."""
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError('Expected [batch, time] within context_length')
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, token_ids):
        return self.lm_head(self.encode(token_ids))
