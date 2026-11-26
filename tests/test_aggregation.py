"""Exact parent/child aggregation identities."""

import numpy as np
import pandas as pd
import pytest

from olm.data.aggregation import (
    AggregationError,
    aggregate_blocks,
    aggregate_buckets,
    aggregate_ohlc,
    build_resolutions,
    reconstruct_parent,
)


def _m1(rows=480, start='2024-01-01'):
    ts = pd.date_range(start, periods=rows, freq='min', tz='UTC')
    rng = np.random.default_rng(5)
    close = 2000 + np.cumsum(rng.normal(0, 0.1, rows))
    open_ = np.r_[2000.0, close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.05, rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.05, rows))
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close})


def test_build_resolutions_reconstructs_exactly():
    m1 = _m1()
    frames = build_resolutions(m1)
    for parent_name, factor in (('5min', 5), ('15min', 15), ('1h', 60), ('4h', 240)):
        parent = frames[parent_name]
        children = frames['1min'].iloc[: len(parent) * factor]
        for index in range(min(len(parent), 20)):
            block = children.iloc[index * factor : (index + 1) * factor]
            implied = reconstruct_parent(block)
            row = parent.iloc[index]
            assert implied['open'] == pytest.approx(row['open'])
            assert implied['high'] == pytest.approx(row['high'])
            assert implied['low'] == pytest.approx(row['low'])
            assert implied['close'] == pytest.approx(row['close'])


def test_aggregate_blocks_rejects_gaps():
    m1 = _m1(60).drop(index=17).reset_index(drop=True)
    with pytest.raises(AggregationError):
        aggregate_blocks(m1, 5)


def test_aggregate_buckets_preserves_closures_and_drops_partial_buckets():
    m1 = _m1(60).drop(index=range(20, 25)).reset_index(drop=True)
    result = aggregate_buckets(m1, 5, 1)
    assert len(result) == 11
    assert pd.Timedelta(minutes=10) in result['timestamp'].diff().dropna().tolist()


def test_aggregate_ohlc_rejects_incomplete_bucket():
    m1 = _m1(12).drop(index=3).reset_index(drop=True)
    with pytest.raises(AggregationError):
        aggregate_ohlc(m1, 5)


def test_untruncated_length_rejected_by_block_builder():
    m1 = _m1(13)
    result = aggregate_blocks(m1, 5)
    assert len(result) == 2


def test_aggregation_is_timezone_stable():
    m1 = _m1(300)
    m1['timestamp'] = m1['timestamp'].dt.tz_convert('America/New_York')
    frames = build_resolutions(m1)
    assert str(frames['5min']['timestamp'].dt.tz) == 'UTC'
