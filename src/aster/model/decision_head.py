"""Decision head for scoring one structured candidate action at a time."""

import torch
from torch import nn

from aster.model.tiny_lm import TinyLM


class DecisionHead(nn.Module):
    """Map one contextual representation to one scalar candidate score."""

    def __init__(self, width: int):
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        self.projection = nn.Linear(width, 1)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def _pool(self, hidden, lengths=None):
        if hidden.ndim != 3:
            raise ValueError("Expected hidden states with shape [batch, time, width]")
        batch, time, _ = hidden.shape
        if time <= 0:
            raise ValueError("Decision inputs must contain at least one token")

        if lengths is None:
            return hidden[:, -1]

        lengths = torch.as_tensor(lengths, device=hidden.device, dtype=torch.long)
        if lengths.ndim != 1 or lengths.shape[0] != batch:
            raise ValueError("lengths must have shape [batch]")
        if torch.any(lengths <= 0) or torch.any(lengths > time):
            raise ValueError("lengths must point inside the encoded sequence")
        rows = torch.arange(batch, device=hidden.device)
        return hidden[rows, lengths - 1]

    def forward(self, hidden, lengths=None):
        return self.projection(self._pool(hidden, lengths)).squeeze(-1)


class ResidualDecisionHead(DecisionHead):
    """Linear scalar head plus a zero-output-initialized nonlinear residual branch."""

    def __init__(self, width: int, residual_width: int | None = None):
        super().__init__(width)
        if residual_width is None:
            residual_width = width
        if type(residual_width) is not int or residual_width <= 0:
            raise ValueError("residual_width must be a positive integer")
        self.residual_width = residual_width
        self.residual_input = nn.Linear(width, residual_width)
        self.residual_activation = nn.GELU()
        self.residual_output = nn.Linear(residual_width, 1)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    @classmethod
    def from_linear(
        cls,
        head: DecisionHead,
        *,
        residual_width: int | None = None,
    ) -> "ResidualDecisionHead":
        """Copy the exact linear scorer and add a zero-output residual branch."""
        width = head.projection.in_features
        residual = cls(width, residual_width=residual_width)
        residual.projection.load_state_dict(head.projection.state_dict())
        return residual

    def forward(self, hidden, lengths=None):
        pooled = self._pool(hidden, lengths)
        linear = self.projection(pooled)
        residual = self.residual_output(
            self.residual_activation(self.residual_input(pooled))
        )
        return (linear + residual).squeeze(-1)


class DecisionModel(nn.Module):
    """A TinyLM shared backbone plus a scalar candidate-scoring head."""

    def __init__(self, backbone: TinyLM, head: DecisionHead | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head or DecisionHead(backbone.config.width)

    def forward(self, token_ids, lengths=None):
        return self.head(self.backbone.encode(token_ids), lengths)
