"""Provider payload parsing (no network)."""

import pandas as pd
import pytest

from olm.data.providers.twelvedata import ProviderError, parse_time_series


def test_parse_time_series_sorts_and_types():
    payload = {
        'meta': {'symbol': 'XAU/USD', 'interval': '5min'},
        'values': [
            {'datetime': '2024-01-01 00:10:00', 'open': '2', 'high': '3', 'low': '1', 'close': '2.5', 'volume': '0'},
            {'datetime': '2024-01-01 00:05:00', 'open': '1', 'high': '2', 'low': '0.5', 'close': '1.5'},
        ],
    }
    parsed = parse_time_series(payload)
    assert len(parsed.frame) == 2
    assert parsed.frame['timestamp'].is_monotonic_increasing
    assert parsed.frame['timestamp'].dt.tz is not None
    assert parsed.frame['close'].dtype.kind == 'f'
    assert parsed.meta['symbol'] == 'XAU/USD'


def test_parse_error_payload_raises():
    with pytest.raises(ProviderError):
        parse_time_series({'status': 'error', 'message': 'invalid api key'})


def test_empty_payload_has_columns():
    parsed = parse_time_series({'values': []})
    assert list(parsed.frame.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
