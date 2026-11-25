"""Price, probabilistic and calibration metrics for OLM forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from olm.config import Config
from olm.models.decoding import aggregate_children_torch, decode_model_torch


COORDS = ('open', 'high', 'low', 'close')


@dataclass
class ForecastEvaluation:
    n: int
    metrics: dict = field(default_factory=dict)
    per_coordinate: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'n': self.n, 'metrics': self.metrics, 'per_coordinate': self.per_coordinate}


@torch.no_grad()
def _to_model_coords(coords: torch.Tensor, normalizer) -> torch.Tensor:
    return coords * torch.as_tensor(normalizer.std, dtype=coords.dtype) + torch.as_tensor(
        normalizer.mean, dtype=coords.dtype
    )


@torch.no_grad()
def collect_predictions(model, loader, config: Config, device: str, normalizer, n_samples: int = 32) -> dict:
    model.eval()
    store: dict[str, list] = {
        'parent_pred': [], 'parent_true': [], 'child_pred': [], 'child_true': [],
        'm15_prev_close': [], 'child_prev_close': [], 'parent_nll': [], 'child_nll': [],
        'parent_samples': [], 'target_ts': [],
    }
    generator = torch.Generator(device='cpu').manual_seed(0)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        parent_pred = decode_model_torch(
            _to_model_coords(outputs['parent'].mean(), normalizer), batch['m15_prev_close']
        )
        children = [
            decode_model_torch(
                _to_model_coords(mixture.mean(), normalizer), batch['children_prev_close'][:, step]
            )
            for step, mixture in enumerate(outputs['children'])
        ]
        child_pred = torch.stack(children, dim=1)
        samples = []
        for _ in range(n_samples):
            coords = outputs['parent'].sample(generator)
            samples.append(
                decode_model_torch(_to_model_coords(coords, normalizer), batch['m15_prev_close'])
            )
        parent_nll = -outputs['parent'].log_prob(batch['parent_enc'])
        child_nll = torch.stack([-m.log_prob(batch['children_enc'][:, step]) for step, m in enumerate(outputs['children'])], 1)
        store['parent_pred'].append(parent_pred.cpu())
        store['parent_true'].append(batch['parent_prices'].cpu())
        store['child_pred'].append(child_pred.cpu())
        store['child_true'].append(batch['child_prices'].cpu())
        store['m15_prev_close'].append(batch['m15_prev_close'].cpu())
        store['child_prev_close'].append(batch['children_prev_close'].cpu())
        store['parent_nll'].append(parent_nll.cpu())
        store['child_nll'].append(child_nll.cpu())
        store['parent_samples'].append(torch.stack(samples, dim=1).cpu())
        store['target_ts'].append(batch['target_ts'].cpu())
    return {key: torch.cat(value, dim=0).numpy() for key, value in store.items()}


def _price_metrics(pred: np.ndarray, true: np.ndarray, prev_close: np.ndarray) -> dict:
    error = np.abs(pred - true)
    normalized = error / np.maximum(prev_close[..., None] if prev_close.ndim else prev_close, 1e-9)
    body_pred = np.abs(pred[..., 3] - pred[..., 0])
    body_true = np.abs(true[..., 3] - true[..., 0])
    upper_pred = pred[..., 1] - np.maximum(pred[..., 0], pred[..., 3])
    upper_true = true[..., 1] - np.maximum(true[..., 0], true[..., 3])
    lower_pred = np.minimum(pred[..., 0], pred[..., 3]) - pred[..., 2]
    lower_true = np.minimum(true[..., 0], true[..., 3]) - true[..., 2]
    range_pred = pred[..., 1] - pred[..., 2]
    range_true = true[..., 1] - true[..., 2]
    return {
        'mae': float(error.mean()),
        'normalized_mae': float(normalized.mean()),
        'body_error': float(np.abs(body_pred - body_true).mean()),
        'wick_error': float((np.abs(upper_pred - upper_true) + np.abs(lower_pred - lower_true)).mean()),
        'range_error': float(np.abs(range_pred - range_true).mean()),
        'per_coordinate_mae': {name: float(error[..., idx].mean()) for idx, name in enumerate(COORDS)},
    }


def _crps(samples: np.ndarray, target: np.ndarray) -> float:
    """Sample-based CRPS averaged over rows (samples shape (N, S))."""
    term1 = np.abs(samples - target[:, None]).mean(axis=1)
    term2 = np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2)) / 2.0
    return float((term1 - term2).mean())


def _coverage(samples: np.ndarray, target: np.ndarray, level: float) -> float:
    lower = (1 - level) / 2
    low = np.quantile(samples, lower, axis=1)
    high = np.quantile(samples, 1 - lower, axis=1)
    return float(((target >= low) & (target <= high)).mean())


def evaluate_predictions(data: dict) -> ForecastEvaluation:
    parent_pred = data['parent_pred']
    parent_true = data['parent_true']
    child_pred = data['child_pred']
    child_true = data['child_true']
    m15_prev = data['m15_prev_close']
    child_prev = data['child_prev_close']
    parent_metrics = _price_metrics(parent_pred, parent_true, m15_prev)
    child_metrics = _price_metrics(child_pred, child_true, child_prev)
    log_pred = np.log(np.clip(child_pred, 1e-8, None))
    agg = np.stack([
        log_pred[:, 0, 0],
        log_pred[:, :, 1].max(axis=1),
        log_pred[:, :, 2].min(axis=1),
        log_pred[:, 2, 3],
    ], axis=1)
    log_true = np.log(np.clip(parent_true, 1e-8, None))
    aggregation_error = float(np.abs(agg - log_true).mean())
    samples = data['parent_samples']  # (N, S, 4)
    crps = {name: _crps(samples[..., idx], parent_true[..., idx]) for idx, name in enumerate(COORDS)}
    coverage = {
        f'coverage_{int(level * 100)}': {name: _coverage(samples[..., idx], parent_true[..., idx], level) for idx, name in enumerate(COORDS)}
        for level in (0.5, 0.9)
    }
    metrics = {
        'parent_mae': parent_metrics['mae'],
        'parent_normalized_mae': parent_metrics['normalized_mae'],
        'child_mae': child_metrics['mae'],
        'child_normalized_mae': child_metrics['normalized_mae'],
        'body_error': child_metrics['body_error'],
        'wick_error': child_metrics['wick_error'],
        'range_error': child_metrics['range_error'],
        'parent_nll_model_space': float(data['parent_nll'].mean()),
        'child_nll_model_space': float(data['child_nll'].mean()),
        'aggregation_log_error': aggregation_error,
        'crps_mean': float(np.mean(list(crps.values()))),
        'coverage_50_mean': float(np.mean(list(coverage['coverage_50'].values()))),
        'coverage_90_mean': float(np.mean(list(coverage['coverage_90'].values()))),
    }
    return ForecastEvaluation(
        n=int(len(parent_true)),
        metrics=metrics,
        per_coordinate={
            'parent_mae': parent_metrics['per_coordinate_mae'],
            'child_mae': child_metrics['per_coordinate_mae'],
            'crps': crps,
            **coverage,
        },
    )
