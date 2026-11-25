"""Continuous candle embedding: four encoded coordinates -> d_model."""

from __future__ import annotations

import torch
from torch import nn


class CandleEncoder(nn.Module):
    """MLP over (g, b, a_u, a_d). Shared by every resolution and the decoder."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 4:
            raise ValueError(f'expected 4 candle coordinates, got {x.shape[-1]}')
        return self.norm(self.net(x))
