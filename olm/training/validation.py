"""Evaluation of the joint objective on a held-out loader."""

from __future__ import annotations

import torch

from olm.config import Config
from olm.training.losses import compute_losses


@torch.no_grad()
def evaluate_losses(model, loader, config: Config, device: str, normalizer) -> dict[str, float]:
    model.eval()
    totals = {'total': 0.0, 'parent': 0.0, 'child': 0.0, 'aggregate': 0.0, 'consistency': 0.0, 'kl': 0.0}
    seen = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        losses = compute_losses(outputs, batch, config, normalizer).detached()
        batch_size = batch['m5'].shape[0]
        for key in totals:
            totals[key] += losses[key] * batch_size
        seen += batch_size
    if not seen:
        return {key: float('nan') for key in totals} | {'samples': 0}
    result = {key: value / seen for key, value in totals.items()}
    result['samples'] = seen
    return result


@torch.no_grad()
def evaluate_validation(
    model, loader, config: Config, device: str, normalizer, n_samples: int = 8, ar1_params: dict | None = None,
) -> dict[str, float]:
    """Losses plus teacher-forced price, calibration and baseline metrics."""
    result = evaluate_losses(model, loader, config, device, normalizer)
    if int(result.get('samples', 0)) == 0:
        return result
    from olm.evaluation.metrics import collect_predictions, evaluate_predictions

    data = collect_predictions(
        model, loader, config, device, normalizer, n_samples=n_samples, ar1_params=ar1_params
    )
    metrics = evaluate_predictions(data).metrics
    result.update(metrics)
    return result
