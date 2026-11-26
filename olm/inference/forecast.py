"""Standalone probabilistic forecasting and coherent scenario generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from olm.config import Config, ModelConfig
from olm.data.aggregation import aggregate_buckets
from olm.data.calendar import TemporalFeatures
from olm.data.datasets import ResolutionArrays, context_batch, prepare_resolution
from olm.data.encoding import decode_model
from olm.models.decoding import aggregate_children_torch, decode_model_torch
from olm.training.checkpointing import load_checkpoint, model_from_checkpoint, normalizer_from_checkpoint


@dataclass
class ForecastResult:
    as_of: str
    parent_distribution: dict
    scenarios: list[dict]
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            'as_of': self.as_of,
            'parent_distribution': self.parent_distribution,
            'scenarios': self.scenarios,
            'diagnostics': self.diagnostics,
        }


def _candle_dict(prices: np.ndarray) -> dict:
    return {
        'open': float(prices[0]),
        'high': float(prices[1]),
        'low': float(prices[2]),
        'close': float(prices[3]),
    }


def project_children_to_parent(children: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """Enforce exact OHLC aggregation with minimal, documented distortion.

    Prices are clamped into the parent range, the first child opens at the
    parent open, the last closes at the parent close, and the extreme wicks are
    moved to the parent high/low. Bodies stay valid. This is a coherence
    projection, not a claim that independent child samples were consistent.
    """
    out = np.clip(children.copy(), parent[:, None, 2:3], parent[:, None, 1:2])
    out[:, 0, 0] = parent[:, 0]
    out[:, 2, 3] = parent[:, 3]
    top = np.maximum(out[..., 0], out[..., 3])
    bottom = np.minimum(out[..., 0], out[..., 3])
    out[..., 1] = np.maximum(out[..., 1], top)
    out[..., 2] = np.minimum(out[..., 2], bottom)
    rows = np.arange(len(out))
    out[rows, out[:, :, 1].argmax(axis=1), 1] = parent[:, 1]
    out[rows, out[:, :, 2].argmin(axis=1), 2] = parent[:, 2]
    return out


def derive_context_frames(m5: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive M15/H1/H4 from validated M5 history using exact block aggregation."""
    frame = m5.copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    if len(frame) > 1:
        step = frame['timestamp'].diff().dropna().dt.total_seconds().div(60).min()
        if not np.isclose(step, 5.0):
            raise ValueError(f'derive_context_frames expects 5-minute candles, got {step:g}-minute spacing')
    frames = {'5min': frame}
    for name, minutes in (('15min', 15), ('1h', 60), ('4h', 240)):
        built = aggregate_buckets(frame, minutes, 5)
        if built.empty:
            raise ValueError(f'not enough complete M5 candles to build {name}')
        frames[name] = built
    return frames


class Forecaster:
    def __init__(self, checkpoint_path: str | Path, config: Config | None = None, device: str = 'cpu') -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = load_checkpoint(self.checkpoint_path, map_location=device)
        self.model = model_from_checkpoint(self.checkpoint, map_location=device)
        self.device = device
        self.config = config or Config()
        model_config = dict(self.checkpoint['model_config'])
        temporal_dim = int(model_config.pop('temporal_dim'))
        self.model_config = ModelConfig(**model_config)
        self.config.model = self.model_config
        self.normalizer = normalizer_from_checkpoint(self.checkpoint)
        self.temporal = TemporalFeatures(self.config.temporal)
        if self.temporal.dim != temporal_dim:
            raise ValueError(
                f'temporal feature width {self.temporal.dim} does not match checkpoint ({temporal_dim}); '
                'the temporal configuration used at training time must be reproduced'
            )
        digest = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()[:12]
        self.model_version = f"olm-{self.checkpoint.get('olm_version', 'unknown')}-{digest}"
        self.model.eval()

    def _arrays(self, frames: dict[str, pd.DataFrame]) -> dict[str, ResolutionArrays]:
        arrays = {}
        for name in ('5min', '15min', '1h', '4h'):
            prepared = prepare_resolution(frames[name], self.temporal)
            prepared.coords = self.normalizer.transform(prepared.coords)
            arrays[name] = prepared
        return arrays

    def _latest_context(self, arrays: dict[str, ResolutionArrays]) -> tuple[int, int, int, int]:
        m5, m15 = arrays['5min'], arrays['15min']
        n = len(m5.ts)
        cfg = self.model_config
        for i in range(n - 1, -1, -1):
            next_aligned = (m5.ts[i].astype('datetime64[m]').astype(np.int64) + 5) % 15 == 0
            if not next_aligned or i - cfg.m5_context + 1 < 0:
                continue
            p15 = int(np.searchsorted(m15.ts, m5.ts[i] + np.timedelta64(5, 'm'), side='left'))
            if p15 < cfg.m15_context:
                continue
            target_start = m5.ts[i] + np.timedelta64(5, 'm')
            j1 = int(np.searchsorted(arrays['1h'].ts + np.timedelta64(60, 'm'), target_start, side='right')) - 1
            j4 = int(np.searchsorted(arrays['4h'].ts + np.timedelta64(240, 'm'), target_start, side='right')) - 1
            if j1 - cfg.h1_context + 1 < 0 or j4 - cfg.h4_context + 1 < 0:
                continue
            return i, p15, j1, j4
        raise ValueError('history is too short or not aligned for a forecast')

    @torch.no_grad()
    def forecast(
        self,
        frames: dict[str, pd.DataFrame],
        scenarios: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        arrays = self._arrays(frames)
        i, p15, j1, j4 = self._latest_context(arrays)
        m15 = arrays['15min']
        m5 = arrays['5min']
        batch = context_batch(arrays, i, p15, j1, j4, self.config)
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in batch.items()}
        state, _ = self.model.encode_state(batch)
        parent_mixture = self.model.parent_head(state)
        n_scenarios = int(scenarios or self.model_config.n_scenarios)
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()
        parent_coords = parent_mixture.sample_many(n_scenarios, generator)[0]
        parent_prev_close = float(m15.prices[p15 - 1, 3])
        parent_model = torch.as_tensor(
            self.normalizer.inverse(parent_coords.cpu().numpy()), dtype=torch.float32
        )
        parent_prices = decode_model(parent_model.numpy(), parent_prev_close)
        state_batch = state.expand(n_scenarios, -1)
        previous_children = torch.zeros((n_scenarios, 0, 4), device=self.device)
        child_coords = []
        child_prices = []
        prev_child_close = np.repeat(np.asarray([parent_prices[:, 0]]), 1, axis=0).reshape(-1)
        for step in range(3):
            distribution = self.model.child_decoder.forward_step(state_batch, parent_coords, previous_children)
            coords = distribution.sample(generator)
            coords_model = torch.as_tensor(
                self.normalizer.inverse(coords.cpu().numpy()), dtype=torch.float32
            )
            prices = decode_model_torch(coords_model, torch.as_tensor(prev_child_close, dtype=torch.float32))
            child_coords.append(coords)
            child_prices.append(prices.cpu().numpy())
            previous_children = torch.cat([previous_children, coords.unsqueeze(1)], dim=1)
            prev_child_close = prices[:, 3].cpu().numpy()
        raw_children = np.stack(child_prices, axis=1)
        projected = project_children_to_parent(raw_children, parent_prices)
        raw_aggregate = np.stack([
            raw_children[:, 0, 0],
            raw_children[:, :, 1].max(axis=1),
            raw_children[:, :, 2].min(axis=1),
            raw_children[:, 2, 3],
        ], axis=1)
        raw_gap = np.abs(np.log(np.clip(raw_aggregate, 1e-9, None)) - np.log(np.clip(parent_prices, 1e-9, None))).mean()
        projected_gap = np.abs(
            np.log(np.clip(np.stack([projected[:, 0, 0], projected[:, :, 1].max(1), projected[:, :, 2].min(1), projected[:, 2, 3]], 1), 1e-9, None))
            - np.log(np.clip(parent_prices, 1e-9, None))
        ).max()
        scenario_rows = []
        for index in range(n_scenarios):
            scenario_rows.append({
                'parent': _candle_dict(parent_prices[index]),
                'children': [_candle_dict(projected[index, step]) for step in range(3)],
                'coherent': bool(projected_gap <= 1e-6),
            })
        mixture_mean_prices = decode_model(
            self.normalizer.inverse(parent_mixture.mean()[0].cpu().numpy()), parent_prev_close
        )
        dominant_prices = decode_model(
            self.normalizer.inverse(parent_mixture.dominant_mean()[0].cpu().numpy()), parent_prev_close
        )
        parent_summary = {
            'expected': _candle_dict(mixture_mean_prices),
            'dominant_component': _candle_dict(dominant_prices),
            'quantiles': {},
            'prev_close': parent_prev_close,
        }
        for idx, name in enumerate(('open', 'high', 'low', 'close')):
            parent_summary['quantiles'][name] = {
                'q05': float(np.quantile(parent_prices[:, idx], 0.05)),
                'q50': float(np.quantile(parent_prices[:, idx], 0.50)),
                'q95': float(np.quantile(parent_prices[:, idx], 0.95)),
            }
        return ForecastResult(
            as_of=str(pd.Timestamp(m5.ts[i] + np.timedelta64(5, 'm')).tz_localize('UTC')),
            parent_distribution=parent_summary,
            scenarios=scenario_rows,
            diagnostics={
                'model_version': self.model_version,
                'scenarios': n_scenarios,
                'raw_aggregation_gap': float(raw_gap),
                'projected_max_gap': float(projected_gap),
                'coherence_projection': 'clamp-to-parent + open/close/extreme-wick alignment',
                'seed': seed,
            },
        )


def forecast_frames(frames: dict[str, pd.DataFrame], checkpoint: str | Path, config: Config,
                    scenarios: int = 32, seed: int | None = None) -> dict:
    forecaster = Forecaster(checkpoint, config)
    return forecaster.forecast(frames, scenarios=scenarios, seed=seed).to_dict()
