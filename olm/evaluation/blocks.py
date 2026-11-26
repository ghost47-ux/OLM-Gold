"""Sequential block evaluation: stability over time, not a retrained backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from olm.evaluation.free_running import evaluate_free_running
from olm.evaluation.metrics import collect_predictions, evaluate_predictions


@dataclass
class BlockResult:
    index: int
    start: int
    end: int
    n: int
    metrics: dict

    def to_dict(self) -> dict:
        return {'index': self.index, 'start': self.start, 'end': self.end, 'n': self.n, 'metrics': self.metrics}


def block_bounds(n: int, folds: int) -> list[tuple[int, int]]:
    if folds < 1:
        raise ValueError('folds must be at least 1')
    if n < folds:
        raise ValueError(f'cannot split {n} samples into {folds} non-empty blocks')
    edges = np.linspace(0, n, folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(folds)]


@torch.no_grad()
def evaluate_sequential_blocks(
    model, dataset, config, device: str, normalizer, folds: int = 4, n_samples: int = 16,
    ar1_params: dict | None = None,
) -> dict:
    if folds > len(dataset):
        raise ValueError('not enough samples for the requested number of folds')
    results: list[BlockResult] = []
    aggregate: list[dict] = []
    for index, (start, end) in enumerate(block_bounds(len(dataset), folds)):
        subset = Subset(dataset, list(range(start, end)))
        loader = DataLoader(subset, batch_size=config.training.eval_batch_size, shuffle=False)
        predictions = collect_predictions(
            model, loader, config, device, normalizer, n_samples=n_samples, ar1_params=ar1_params
        )
        evaluation = evaluate_predictions(predictions)
        free = evaluate_free_running(model, loader, config, device, normalizer)
        results.append(BlockResult(index, start, end, evaluation.n, evaluation.metrics))
        aggregate.append({**evaluation.metrics, 'free_running_child_mae': free['child_mae_free_running']})
    keys = aggregate[0].keys()
    summary = {
        key: float(np.mean([row[key] for row in aggregate])) for key in keys
    }
    return {
        'folds': folds,
        'blocks': [block.to_dict() for block in results],
        'mean': summary,
        'note': 'Blocks are evaluated with one fixed model; this measures temporal stability, not a retrained walk-forward backtest.',
    }
