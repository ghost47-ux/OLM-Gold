"""Differentiable decoding of model-space candle coordinates (torch)."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from olm.data.encoding import EPS_WICK


def decode_model_torch(x: torch.Tensor, prev_close: torch.Tensor) -> torch.Tensor:
    """Map (..., 4) model coordinates and (...) previous closes to (..., 4) OHLC."""
    if x.shape[-1] != 4:
        raise ValueError('expected four coordinates')
    pc = prev_close.unsqueeze(-1) if prev_close.dim() == x.dim() - 1 else prev_close
    g, b = x[..., 0], x[..., 1]
    upper = torch.clamp(F.softplus(x[..., 2]) - EPS_WICK, min=0.0)
    lower = torch.clamp(F.softplus(x[..., 3]) - EPS_WICK, min=0.0)
    o = pc.squeeze(-1) * torch.exp(g)
    c = o * torch.exp(b)
    top = torch.maximum(o, c)
    bottom = torch.minimum(o, c)
    h = top * torch.exp(upper)
    l = bottom * torch.exp(-lower)
    return torch.stack([o, h, l, c], dim=-1)


def aggregate_children_torch(children: torch.Tensor) -> torch.Tensor:
    """Aggregate (..., 3, 4) children into (..., 4) parents using OHLC semantics."""
    if children.shape[-2] != 3:
        raise ValueError('expected three children')
    o = children[..., 0, 0]
    h = children[..., :, 1].max(dim=-1).values
    l = children[..., :, 2].min(dim=-1).values
    c = children[..., 2, 3]
    return torch.stack([o, h, l, c], dim=-1)
