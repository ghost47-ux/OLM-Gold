"""Learning-rate schedule: linear warmup followed by cosine decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.05):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    min_ratio = min(max(min_ratio, 0.0), 1.0)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, factor)
