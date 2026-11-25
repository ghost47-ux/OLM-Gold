"""OHLC validation catches structural defects."""

import pandas as pd

from olm.data.validation import validate_frame


def _frame(rows=20):
    ts = pd.date_range('2024-01-01', periods=rows, freq='5min', tz='UTC')
    return pd.DataFrame({'timestamp': ts, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5})


def test_valid_frame_passes():
    report = validate_frame(_frame())
    assert report.ok


def test_duplicate_and_unsorted_detected():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    report = validate_frame(frame)
    assert not report.ok
    assert any('duplicate' in error for error in report.errors)


def test_bad_high_low_detected():
    frame = _frame()
    frame.loc[2, 'high'] = 90.0
    frame.loc[4, 'low'] = 110.0
    report = validate_frame(frame)
    assert not report.ok
    assert report.stats['invalid_high'] == 1
    assert report.stats['invalid_low'] == 1


def test_gap_is_warning_not_error():
    frame = _frame().drop(index=5).reset_index(drop=True)
    report = validate_frame(frame)
    assert report.ok
    assert report.stats['missing_bars'] == 1


def test_nonpositive_prices_rejected():
    frame = _frame()
    frame.loc[1, 'low'] = 0.0
    report = validate_frame(frame)
    assert not report.ok
