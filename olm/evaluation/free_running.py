"""Recursive multi-step evaluation, separated from teacher-forced scoring."""

from __future__ import annotations

import torch

from olm.config import Config
from olm.evaluation.metrics import _to_model_coords
from olm.models.decoding import aggregate_children_torch, decode_model_torch


@torch.no_grad()
def evaluate_free_running(model, loader, config: Config, device: str, normalizer) -> dict:
    """Recursive child generation from the predicted parent (no actual children)."""
    model.eval()
    errors = []
    aggregate_errors = []
    coherence = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        state = outputs['state']
        parent_coords = outputs['parent'].mean()
        parent_prices = decode_model_torch(
            _to_model_coords(parent_coords, normalizer), batch['m15_prev_close']
        )
        previous = torch.zeros((state.shape[0], 0, 4), device=device)
        prev_close = parent_prices[:, 0]
        predicted = []
        for _ in range(3):
            distribution = model.child_decoder.forward_step(state, parent_coords, previous)
            coords = distribution.mean()
            prices = decode_model_torch(_to_model_coords(coords, normalizer), prev_close)
            predicted.append(prices)
            previous = torch.cat([previous, coords.unsqueeze(1)], dim=1)
            prev_close = prices[:, 3]
        children = torch.stack(predicted, dim=1)
        aggregated = aggregate_children_torch(children)
        absolute = (children - batch['child_prices']).abs()
        errors.append(absolute.mean(dim=(1, 2)).cpu())
        aggregate_errors.append(
            (torch.log(torch.clamp(aggregated, min=1e-8)) - torch.log(torch.clamp(batch['parent_prices'], min=1e-8)))
            .abs().mean(dim=1).cpu()
        )
        coherence.append((aggregated - parent_prices).abs().mean(dim=1).cpu())
    error = torch.cat(errors)
    aggregate = torch.cat(aggregate_errors)
    coherence_error = torch.cat(coherence)
    return {
        'samples': int(len(error)),
        'child_mae_free_running': float(error.mean()),
        'aggregate_log_error_free_running': float(aggregate.mean()),
        'predicted_parent_aggregate_abs_error': float(coherence_error.mean()),
        'note': 'Children are generated recursively from the predicted parent; this measures multi-step compounding, not a live trading result.',
    }
