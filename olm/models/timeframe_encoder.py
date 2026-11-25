"""Learned timeframe identity embeddings."""

from __future__ import annotations

import torch
from torch import nn

TIMEFRAMES = {'5min': 0, '15min': 1, '1h': 2, '4h': 3}


class TimeframeEncoder(nn.Module):
    def __init__(self, d_model: int, n_timeframes: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_timeframes, d_model)

    def forward(self, timeframe_id: int, length: int) -> torch.Tensor:
        index = torch.full((length,), timeframe_id, dtype=torch.long)
        return self.embedding(index).unsqueeze(0)
