"""UTC-aligned OHLC aggregation. Child candles must reconstruct parents exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ('timestamp', 'open', 'high', 'low', 'close')


class AggregationError(ValueError):
    pass


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise AggregationError(f'missing columns: {sorted(missing)}')
    out = frame.copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True)
    return out.sort_values('timestamp').reset_index(drop=True)


def aggregate_ohlc(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate candles into fixed UTC buckets aligned to the Unix epoch.

    Empty buckets are dropped (market closures). Call aggregate_blocks when a
    complete grid is required.
    """
    if minutes <= 0:
        raise ValueError('minutes must be positive')
    out = _prepare(frame)
    out['timestamp'] = out['timestamp'].dt.floor(f'{minutes}min')
    grouped = out.groupby('timestamp', sort=True).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        bar_count=('open', 'size'),
    )
    expected = minutes // _infer_minutes(out)
    if (grouped['bar_count'] != expected).any():
        bad = grouped[grouped['bar_count'] != expected]
        raise AggregationError(
            f'{len(bad)} {minutes}-minute buckets have incomplete children; '
            'align or validate the source series first'
        )
    return grouped.reset_index().drop(columns=['bar_count'])


def _infer_minutes(frame: pd.DataFrame) -> int:
    diffs = frame['timestamp'].diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        raise AggregationError('cannot infer source interval from a single candle')
    return int(round(float(diffs.min())))


def aggregate_blocks(frame: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Aggregate consecutive, strictly contiguous groups of `factor` candles.

    This is the exact-parent construction used for M1->M5 and M5->M15. It
    verifies that every block is complete and begins on an aligned boundary.
    """
    if factor < 1:
        raise ValueError('factor must be >= 1')
    if factor == 1:
        return _prepare(frame)
    out = _prepare(frame)
    if len(out) % factor != 0:
        out = out.iloc[: len(out) - (len(out) % factor)]
    minutes = _infer_minutes(out)
    expected = pd.Timedelta(minutes=minutes * factor)
    groups = np.arange(len(out)) // factor
    agg = out.groupby(groups).agg(
        timestamp=('timestamp', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        bar_count=('open', 'size'),
    )
    if (agg['bar_count'] != factor).any():
        raise AggregationError('incomplete block at the end of the series')
    # Verify that each block is contiguous and epoch-aligned.
    block_times = pd.DatetimeIndex(agg['timestamp'])
    if len(block_times) > 1:
        if not bool((block_times.to_series().diff().dropna() == expected).all()):
            raise AggregationError('source candles contain gaps; blocks are not contiguous')
    epoch_seconds = block_times.tz_convert('UTC').tz_localize(None).as_unit('s').astype('int64')
    block_seconds = minutes * 60 * factor
    if not np.all((epoch_seconds // block_seconds) * block_seconds == epoch_seconds):
        raise AggregationError('blocks are not aligned to epoch boundaries')
    return agg.drop(columns=['bar_count']).reset_index(drop=True)


def build_resolutions(m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build M5/M15/H1/H4 candles from M1 by exact block aggregation."""
    m5 = aggregate_blocks(m1, 5)
    m15 = aggregate_blocks(m1, 15)
    h1 = aggregate_blocks(m1, 60)
    h4 = aggregate_blocks(m1, 240)
    return {'1min': _prepare(m1), '5min': m5, '15min': m15, '1h': h1, '4h': h4}


def reconstruct_parent(children: pd.DataFrame) -> dict[str, float]:
    """Return the unique parent candle implied by children (OHLC semantics)."""
    if children.empty:
        raise AggregationError('no children supplied')
    return {
        'open': float(children['open'].iloc[0]),
        'high': float(children['high'].max()),
        'low': float(children['low'].min()),
        'close': float(children['close'].iloc[-1]),
    }
