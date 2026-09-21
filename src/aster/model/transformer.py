"""小さなdecoderの1層。未来のtokenを参照しないself-attention。"""
import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('width must be a positive multiple of heads')
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)
        # [batch, time, width] -> [batch, head, time, width/head]
        q, k, v = [t.reshape(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2) for t in (q, k, v)]
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + self.projection(attended)
        return x + self.mlp(self.norm_mlp(x))
