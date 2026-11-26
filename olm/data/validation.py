"""OHLC validation, duplicate/gap detection and quarantine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = ['Validation: ' + ('PASS' if self.ok else 'FAIL')]
        for key, value in self.stats.items():
            lines.append(f'  {key}: {value}')
        for message in self.errors:
            lines.append(f'  ERROR: {message}')
        for message in self.warnings:
            lines.append(f'  WARN:  {message}')
        return '\n'.join(lines)


def validate_frame(frame: pd.DataFrame, interval_minutes: int | None = None) -> ValidationReport:
    report = ValidationReport()
    required = {'timestamp', 'open', 'high', 'low', 'close'}
    missing = required - set(frame.columns)
    if missing:
        report.errors.append(f'missing columns: {sorted(missing)}')
        return report
    if frame.empty:
        report.errors.append('frame is empty')
        return report
    report.stats['rows'] = int(len(frame))
    ts = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    if ts.isna().any():
        report.errors.append(f'{int(ts.isna().sum())} timestamps could not be parsed')
    if not ts.is_monotonic_increasing:
        report.errors.append('timestamps are not sorted ascending')
    duplicates = ts.duplicated().sum()
    report.stats['duplicates'] = int(duplicates)
    if duplicates:
        report.errors.append(f'{int(duplicates)} duplicate timestamps')
    prices = frame[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    arr = prices.to_numpy(dtype=np.float64)
    nonfinite = (~np.isfinite(arr)).any(axis=1)
    nonpositive = (arr <= 0).any(axis=1)
    report.stats['nonfinite_rows'] = int(nonfinite.sum())
    report.stats['nonpositive_rows'] = int(nonpositive.sum())
    if nonfinite.any():
        report.errors.append(f'{int(nonfinite.sum())} rows contain NaN/inf prices')
    if nonpositive.any():
        report.errors.append(f'{int(nonpositive.sum())} rows contain non-positive prices')
    top = np.maximum(arr[:, 0], arr[:, 3])
    bottom = np.minimum(arr[:, 0], arr[:, 3])
    bad_high = arr[:, 1] < top - 1e-12 * np.abs(top)
    bad_low = arr[:, 2] > bottom + 1e-12 * np.abs(bottom)
    report.stats['invalid_high'] = int(np.nansum(bad_high))
    report.stats['invalid_low'] = int(np.nansum(bad_low))
    if np.nansum(bad_high):
        report.errors.append(f'{int(np.nansum(bad_high))} rows have high < max(open, close)')
    if np.nansum(bad_low):
        report.errors.append(f'{int(np.nansum(bad_low))} rows have low > min(open, close)')
    sorted_ts = ts.dropna().sort_values().drop_duplicates()
    if interval_minutes and len(sorted_ts):
        misaligned = int((sorted_ts != sorted_ts.dt.floor(f'{interval_minutes}min')).sum())
        if misaligned:
            report.errors.append(
                f'{misaligned} timestamps are not aligned to {interval_minutes}-minute UTC boundaries'
            )
    if len(sorted_ts) > 1:
        diffs = sorted_ts.diff().dropna().dt.total_seconds() / 60.0
        inferred = int(round(float(diffs.min())))
        expected = interval_minutes or inferred
        report.stats['interval_minutes'] = expected
        gaps = diffs[diffs > expected]
        if len(gaps):
            missing_bars = int(((gaps / expected) - 1).sum())
            report.stats['gap_count'] = int(len(gaps))
            report.stats['missing_bars'] = missing_bars
            report.warnings.append(
                f'{int(len(gaps))} gaps totalling {missing_bars} missing bars '
                '(expected for market closures; inspect before treating as defects)'
            )
        extra = diffs[diffs < expected]
        if len(extra):
            report.errors.append(f'{int(len(extra))} intervals are shorter than the expected {expected} minutes')
    return report


def invalid_reasons(frame: pd.DataFrame) -> pd.Series:
    """Per-row reason string, empty for valid candles."""
    reasons = pd.Series([''] * len(frame), index=frame.index, dtype=object)
    prices = frame[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    arr = prices.to_numpy(dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    positive = (arr > 0).all(axis=1)
    top = np.maximum(arr[:, 0], arr[:, 3])
    bottom = np.minimum(arr[:, 0], arr[:, 3])
    bad_high = arr[:, 1] < top
    bad_low = arr[:, 2] > bottom
    for idx, pos in zip(frame.index, range(len(frame))):
        tags = []
        if not finite[pos]:
            tags.append('nonfinite')
        if not positive[pos]:
            tags.append('nonpositive')
        if finite[pos] and positive[pos]:
            if bad_high[pos]:
                tags.append('high_below_body')
            if bad_low[pos]:
                tags.append('low_above_body')
        if tags:
            reasons.iloc[pos] = ';'.join(tags)
    return reasons


def quarantine(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (valid, quarantined). Invalid rows are never repaired."""
    reasons = invalid_reasons(frame)
    keep = reasons == ''
    valid = frame.loc[keep].reset_index(drop=True)
    bad = frame.loc[~keep].copy()
    bad['quarantine_reason'] = reasons[~keep].to_numpy()
    return valid, bad.reset_index(drop=True)


def detect_missing(frame: pd.DataFrame, interval_minutes: int) -> pd.DatetimeIndex:
    """Return timestamps on the expected UTC grid that are absent."""
    ts = pd.to_datetime(frame['timestamp'], utc=True).sort_values()
    if ts.empty:
        return pd.DatetimeIndex([])
    grid = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=f'{interval_minutes}min')
    return grid.difference(pd.DatetimeIndex(ts))
