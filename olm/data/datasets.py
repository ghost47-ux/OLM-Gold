"""Chronological windows, M15 parent targets and teacher-forced child targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from olm.config import Config
from olm.data.encoding import Normalizer, encode_model
from olm.logging_utils import get_logger

logger = get_logger(__name__)

RESOLUTIONS = ('5min', '15min', '1h', '4h')


@dataclass
class ResolutionArrays:
    ts: np.ndarray
    coords: np.ndarray
    prices: np.ndarray
    prev_close: np.ndarray
    temporal: np.ndarray


@dataclass
class DatasetBundle:
    train: 'OLMDataset'
    val: 'OLMDataset'
    test: 'OLMDataset'
    normalizer: Normalizer
    boundaries: dict
    meta: dict = field(default_factory=dict)


def prepare_resolution(frame: pd.DataFrame, temporal, timestamp_col: str = 'timestamp') -> ResolutionArrays:
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    prev_close = frame['close'].shift(1).to_numpy(dtype=np.float64)
    keep = ~np.isnan(prev_close)
    frame = frame.loc[keep].reset_index(drop=True)
    prev_close = prev_close[keep]
    ts = pd.to_datetime(frame[timestamp_col], utc=True).to_numpy(dtype='datetime64[ns]')
    prices = frame[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64)
    coords = encode_model(prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3], prev_close)
    features = temporal.compute(pd.DatetimeIndex(ts))
    return ResolutionArrays(ts=ts, coords=coords, prices=prices, prev_close=prev_close, temporal=features)


def _index_map(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    return np.searchsorted(source_ts, target_ts, side='right') - 1


def _completed_index(target_ts: np.ndarray, source_ts: np.ndarray, minutes: int) -> np.ndarray:
    """Index of the last source bar whose close is at or before each target timestamp.

    A bar stamped at its open time T covers [T, T + minutes).  Using it while the
    bar is still open would expose the future target candles; this is the core
    point-in-time guard for the H1/H4 context windows.
    """
    closes = source_ts + np.timedelta64(int(minutes), 'm')
    return np.searchsorted(closes, target_ts, side='right') - 1


def _candidate_samples(m5: ResolutionArrays, m15: ResolutionArrays, h1: ResolutionArrays, h4: ResolutionArrays, config: Config):
    n = len(m5.ts)
    horizon = 3
    if n <= horizon:
        raise ValueError('not enough M5 candles to build samples')
    minute_of_day = (m5.ts.astype('datetime64[m]').astype(np.int64) // 15) * 15
    aligned = m5.ts.astype('datetime64[m]').astype(np.int64) % 15 == 0
    p15_map = _index_map(m5.ts, m15.ts)
    h1_map = _completed_index(m5.ts, h1.ts, 60)
    h4_map = _completed_index(m5.ts, h4.ts, 240)
    cand = []
    rejected = {'alignment': 0, 'history': 0, 'aggregation': 0}
    for i in range(n - horizon):
        k = i + 1
        if not aligned[k]:
            rejected['alignment'] += 1
            continue
        p15 = int(p15_map[k])
        j1 = int(h1_map[k])
        j4 = int(h4_map[k])
        if (
            p15 < 0 or p15 >= len(m15.ts) or m15.ts[p15] != m5.ts[k]
            or p15 - config.model.m15_context + 1 < 0
            or j1 - config.model.h1_context + 1 < 0
            or j4 - config.model.h4_context + 1 < 0
            or i - config.model.m5_context + 1 < 0
        ):
            rejected['history'] += 1
            continue
        children = m5.prices[k : k + horizon]
        parent = m15.prices[p15]
        if len(children) != horizon:
            rejected['aggregation'] += 1
            continue
        reconstructed = np.array([
            children[0, 0],
            children[:, 1].max(),
            children[:, 2].min(),
            children[-1, 3],
        ])
        if not np.allclose(reconstructed, parent, rtol=1e-10, atol=1e-10):
            rejected['aggregation'] += 1
            continue
        cand.append((i, p15, j1, j4))
    return np.asarray(cand, dtype=np.int64), rejected


def _split_boundaries(target_ts: np.ndarray, config: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    ordered = np.sort(target_ts)
    n = len(ordered)
    train_end = pd.Timestamp(int(ordered[min(n - 1, int(n * config.data.train_fraction))]), tz='UTC') if n else pd.Timestamp('2100-01-01', tz='UTC')
    val_index = min(n - 1, int(n * (config.data.train_fraction + config.data.val_fraction)))
    val_end = pd.Timestamp(int(ordered[val_index]), tz='UTC') if n else train_end
    return train_end, val_end


def context_batch(arrays: dict[str, ResolutionArrays], i: int, p15: int, j1: int, j4: int, config: Config) -> dict[str, torch.Tensor]:
    """Context tensors only; used by both training samples and live inference."""
    m5, m15, h1, h4 = arrays['5min'], arrays['15min'], arrays['1h'], arrays['4h']
    c5, c15 = config.model.m5_context, config.model.m15_context
    c1h, c4h = config.model.h1_context, config.model.h4_context
    return {
        'm5': torch.from_numpy(m5.coords[i - c5 + 1 : i + 1].copy()).float(),
        'm5_temporal': torch.from_numpy(m5.temporal[i - c5 + 1 : i + 1].copy()).float(),
        'm15': torch.from_numpy(m15.coords[p15 - c15 : p15].copy()).float(),
        'm15_temporal': torch.from_numpy(m15.temporal[p15 - c15 : p15].copy()).float(),
        'h1': torch.from_numpy(h1.coords[j1 - c1h + 1 : j1 + 1].copy()).float(),
        'h1_temporal': torch.from_numpy(h1.temporal[j1 - c1h + 1 : j1 + 1].copy()).float(),
        'h4': torch.from_numpy(h4.coords[j4 - c4h + 1 : j4 + 1].copy()).float(),
        'h4_temporal': torch.from_numpy(h4.temporal[j4 - c4h + 1 : j4 + 1].copy()).float(),
    }


class OLMDataset(Dataset):
    """One chronological split. Each item is one teacher-forced parent/child example."""

    def __init__(
        self,
        arrays: dict[str, ResolutionArrays],
        indices: np.ndarray,
        target_rows: dict[str, np.ndarray],
        config: Config,
        split_name: str,
    ) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64)
        self.targets = target_rows
        self.config = config
        self.split_name = split_name

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[item])
        p15 = int(self.targets['p15'][item])
        j1 = int(self.targets['j1'][item])
        j4 = int(self.targets['j4'][item])
        batch = context_batch(self.arrays, i, p15, j1, j4, self.config)
        batch.update({
            'parent_enc': torch.from_numpy(self.targets['parent_enc'][item].copy()).float(),
            'children_enc': torch.from_numpy(self.targets['children_enc'][item].copy()).float(),
            'parent_prices': torch.from_numpy(self.targets['parent_prices'][item].copy()).float(),
            'child_prices': torch.from_numpy(self.targets['child_prices'][item].copy()).float(),
            'm15_prev_close': torch.tensor(float(self.targets['m15_prev_close'][item]), dtype=torch.float32),
            'children_prev_close': torch.from_numpy(self.targets['children_prev_close'][item].copy()).float(),
            'target_ts': torch.tensor(int(self.targets['target_ts'][item]), dtype=torch.int64),
        })
        return batch


def build_datasets(config: Config, frames: dict[str, pd.DataFrame], temporal) -> DatasetBundle:
    """Build chronological train/validation/test splits with a leakage embargo."""
    arrays = {
        '5min': prepare_resolution(frames['5min'], temporal),
        '15min': prepare_resolution(frames['15min'], temporal),
        '1h': prepare_resolution(frames['1h'], temporal),
        '4h': prepare_resolution(frames['4h'], temporal),
    }
    m5, m15 = arrays['5min'], arrays['15min']
    candidates, rejected = _candidate_samples(m5, m15, arrays['1h'], arrays['4h'], config)
    if not len(candidates):
        raise ValueError('no valid parent/child samples; check aggregation alignment and history')
    target_ts = m5.ts[candidates[:, 0] + 1]
    train_end, val_end = _split_boundaries(target_ts, config)
    train_mask = target_ts < np.datetime64(train_end.tz_convert('UTC').tz_localize(None))
    val_mask = (target_ts >= np.datetime64((train_end + pd.Timedelta(minutes=config.data.embargo_minutes)).tz_convert('UTC').tz_localize(None))) & (
        target_ts < np.datetime64(val_end.tz_convert('UTC').tz_localize(None))
    )
    test_mask = target_ts >= np.datetime64((val_end + pd.Timedelta(minutes=config.data.embargo_minutes)).tz_convert('UTC').tz_localize(None))
    normalizer = Normalizer.fit(m5.coords[m5.ts < np.datetime64(train_end.tz_convert('UTC').tz_localize(None))])
    for name in RESOLUTIONS:
        arrays[name].coords = normalizer.transform(arrays[name].coords)

    parent_rows = candidates[:, 1]
    parent_prices = m15.prices[parent_rows]
    m15_prev_close = m15.prices[parent_rows - 1, 3]
    children = np.stack([m5.prices[candidates[:, 0] + step] for step in (1, 2, 3)], axis=1)
    children_prev = np.empty((len(candidates), 3), dtype=np.float64)
    children_prev[:, 0] = parent_prices[:, 0]
    children_prev[:, 1] = children[:, 0, 3]
    children_prev[:, 2] = children[:, 1, 3]
    parent_enc = encode_model(parent_prices[:, 0], parent_prices[:, 1], parent_prices[:, 2], parent_prices[:, 3], m15_prev_close)
    children_enc = np.stack([
        encode_model(children[:, step, 0], children[:, step, 1], children[:, step, 2], children[:, step, 3], children_prev[:, step])
        for step in range(3)
    ], axis=1)
    parent_enc = normalizer.transform(parent_enc)
    children_enc = normalizer.transform(children_enc.reshape(-1, 4)).reshape(-1, 3, 4)

    def make_split(mask: np.ndarray, name: str) -> OLMDataset:
        rows = np.where(mask)[0]
        targets = {
            'p15': candidates[rows, 1],
            'j1': candidates[rows, 2],
            'j4': candidates[rows, 3],
            'parent_enc': parent_enc[rows],
            'children_enc': children_enc[rows],
            'parent_prices': parent_prices[rows],
            'child_prices': children[rows],
            'm15_prev_close': m15_prev_close[rows],
            'children_prev_close': children_prev[rows],
            'target_ts': target_ts[rows].astype('datetime64[s]').astype(np.int64),
        }
        return OLMDataset(arrays, candidates[rows, 0], targets, config, name)

    bundle = DatasetBundle(
        train=make_split(train_mask, 'train'),
        val=make_split(val_mask, 'val'),
        test=make_split(test_mask, 'test'),
        normalizer=normalizer,
        boundaries={
            'train_end': str(train_end),
            'val_end': str(val_end),
            'embargo_minutes': config.data.embargo_minutes,
        },
        meta={
            'samples_total': int(len(candidates)),
            'rejected': rejected,
            'train': int(train_mask.sum()),
            'val': int(val_mask.sum()),
            'test': int(test_mask.sum()),
            'temporal_dim': temporal.dim,
            'temporal_features': temporal.feature_names,
        },
    )
    return bundle


def leakage_check(bundle: DatasetBundle) -> dict:
    """Assert chronological split separation and point-in-time context windows."""
    def bounds(dataset: OLMDataset) -> tuple[int, int]:
        values = dataset.targets['target_ts']
        return (int(values.min()), int(values.max())) if len(values) else (0, 0)

    def context_check(dataset: OLMDataset) -> dict:
        target_ns = dataset.targets['target_ts'].astype('datetime64[s]').astype('datetime64[ns]')
        checks = {}
        for name, minutes in (('1h', 60), ('4h', 240)):
            index = dataset.targets['j1' if name == '1h' else 'j4']
            closes = dataset.arrays[name].ts[index] + np.timedelta64(minutes, 'm')
            checks[name] = {
                'leaking': int(np.sum(closes > target_ns)),
                'samples': int(len(target_ns)),
            }
        return checks

    train = bounds(bundle.train)
    val = bounds(bundle.val)
    test = bounds(bundle.test)
    contexts = {name: context_check(getattr(bundle, name)) for name in ('train', 'val', 'test')}
    context_ok = all(item['leaking'] == 0 for split in contexts.values() for item in split.values())
    ok = train[1] < val[0] and val[1] < test[0] and context_ok
    return {'ok': ok, 'train': train, 'val': val, 'test': test, 'contexts': contexts}
