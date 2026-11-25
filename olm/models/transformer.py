"""Causal pre-LayerNorm Transformer blocks with CPU-friendly attention."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(length: int, d_model: int, device=None) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : pe[:, 1::2].shape[1]]
    return pe


def causal_mask(length: int, device=None, dtype=torch.float32) -> torch.Tensor:
    mask = torch.full((length, length), float('-inf'), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
