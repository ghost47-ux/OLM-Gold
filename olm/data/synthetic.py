"""Deterministic synthetic XAU-like M1 data for tests and offline development.

This data is synthetic. It must never be presented as real market history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _session_volatility(hour: np.ndarray) -> np.ndarray:
    # Broad shape of gold liquidity: quiet Asia, active London, busiest NY overlap.
    table = np.array([0.55, 0.45, 0.4, 0.4, 0.45, 0.55, 0.7, 0.9, 1.1, 1.15, 1.2, 1.1,
                      1.0, 0.95, 1.0, 1.1, 1.25, 1.2, 1.0, 0.85, 0.75, 0.65, 0.6, 0.55])
    return table[hour % 24]


def generate_m1(days: int = 90, seed: int = 7, start: str = '2024-01-01', start_price: float = 2050.0) -> pd.DataFrame:
    """Generate a continuous UTC M1 candle series with volatility clustering."""
    if days < 1:
        raise ValueError('days must be >= 1')
    rng = np.random.default_rng(seed)
    n = int(days) * 1440
    timestamps = pd.date_range(start=pd.Timestamp(start, tz='UTC'), periods=n, freq='1min')
    hour = timestamps.hour.to_numpy()
    base_vol = 0.00016 * _session_volatility(hour)
    regime = np.exp(rng.normal(0.0, 0.35, size=n // 360 + 2))
    regime = np.repeat(regime, 360)[:n]
    shock = np.ones(n)
    shock_idx = rng.choice(n, size=max(1, n // 5000), replace=False)
    shock[shock_idx] = rng.uniform(2.0, 5.0, size=len(shock_idx))
    vol = base_vol * regime * shock
    drift = np.repeat(rng.normal(0.0, 4e-7, size=n // 2000 + 2), 2000)[:n]
    returns = drift + vol * rng.standard_normal(n)
    closes = start_price * np.exp(np.cumsum(returns))
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1]
    body_high = np.maximum(opens, closes)
    body_low = np.minimum(opens, closes)
    wick_scale = vol * rng.uniform(0.3, 1.1, size=n)
    highs = body_high * np.exp(np.abs(rng.normal(0.0, 1.0, n)) * wick_scale)
    lows = body_low * np.exp(-np.abs(rng.normal(0.0, 1.0, n)) * wick_scale)
    volume = rng.lognormal(mean=6.0, sigma=0.6, size=n) * _session_volatility(hour)
    frame = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume,
    })
    frame['symbol'] = 'XAU/USD'
    frame['source'] = 'synthetic'
    return frame


def write_synthetic_dataset(data_root: str = 'data', days: int = 90, seed: int = 7) -> dict:
    """Write raw synthetic M1 plus derived M5/M15/H1/H4 parquet files."""
    from olm.data.aggregation import build_resolutions
    from olm.data.storage import DataPaths, write_parquet_atomic

    paths = DataPaths(data_root).ensure()
    m1 = generate_m1(days=days, seed=seed)
    resolutions = build_resolutions(m1)
    written = {}
    for name, frame in resolutions.items():
        target = paths.raw / f'XAUUSD_synthetic_{name}.parquet'
        write_parquet_atomic(frame, target)
        written[name] = str(target)
    return written
