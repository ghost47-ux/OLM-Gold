"""Point-in-time economic event calendar features.

Only events whose schedule was known at the prediction timestamp are visible.
Rows may carry an explicit `known_at` column; otherwise `known_at` defaults to
`scheduled_at - default_known_lead_hours` (publication lead time). That default
is an explicit assumption and must be reviewed before research claims are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ('CPI', 'NFP', 'FOMC_RATE', 'FOMC_PRESS', 'PCE', 'PPI', 'GDP', 'JOBLESS_CLAIMS')


@dataclass
class EventCalendar:
    events: pd.DataFrame
    default_known_lead_hours: float = 168.0

    @classmethod
    def from_csv(cls, path: str | Path, default_known_lead_hours: float = 168.0) -> 'EventCalendar':
        frame = pd.read_csv(path)
        required = {'timestamp', 'category'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'event calendar missing columns: {sorted(missing)}')
        frame['scheduled_at'] = pd.to_datetime(frame['timestamp'], utc=True)
        frame['category'] = frame['category'].astype(str).str.upper()
        unknown = set(frame['category']) - set(CATEGORIES)
        if unknown:
            raise ValueError(f'unsupported event categories: {sorted(unknown)}')
        if 'known_at' in frame.columns:
            frame['known_at'] = pd.to_datetime(frame['known_at'], utc=True)
        else:
            frame['known_at'] = frame['scheduled_at'] - pd.Timedelta(hours=default_known_lead_hours)
        frame = frame.sort_values('scheduled_at').reset_index(drop=True)
        if (frame['known_at'] > frame['scheduled_at']).any():
            raise ValueError('known_at must not be later than the scheduled release')
        return cls(events=frame, default_known_lead_hours=default_known_lead_hours)

    @classmethod
    def empty(cls) -> 'EventCalendar':
        frame = pd.DataFrame({
            'scheduled_at': pd.to_datetime([], utc=True),
            'known_at': pd.to_datetime([], utc=True),
            'category': pd.Series([], dtype=str),
        })
        return cls(events=frame)

    def features_for(self, timestamps: pd.DatetimeIndex | pd.Series, window_minutes: int = 480) -> np.ndarray:
        """Return (N, 2*len(CATEGORIES)): signed minutes and visibility flags."""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        n = len(ts)
        out = np.zeros((n, 2 * len(CATEGORIES)), dtype=np.float32)
        if n == 0 or self.events.empty:
            return out
        now = ts.asi8.astype(np.int64)
        window_ns = int(window_minutes * 60 * 1e9)
        for slot, category in enumerate(CATEGORIES):
            subset = self.events[self.events['category'] == category]
            if subset.empty:
                continue
            sched = subset['scheduled_at'].to_numpy(dtype='datetime64[ns]').astype(np.int64)
            known = subset['known_at'].to_numpy(dtype='datetime64[ns]').astype(np.int64)
            order = np.argsort(sched)
            sched, known = sched[order], known[order]
            # Only events whose schedule had been published by each timestamp.
            pos = np.searchsorted(sched, now, side='right')
            best_gap = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
            signed = np.zeros(n, dtype=np.float64)
            visible = np.zeros(n, dtype=bool)
            for candidate in (pos - 1, pos):
                valid = (candidate >= 0) & (candidate < len(sched))
                if not valid.any():
                    continue
                idx = np.where(valid, candidate, 0)
                gap = np.abs(now - sched[idx])
                seen = known[idx] <= now
                take = valid & seen & (gap <= window_ns) & (gap < best_gap)
                best_gap[take] = gap[take]
                signed[take] = (now[take] - sched[idx[take]]) / 6e10
                visible[take] = True
            out[:, slot] = np.clip(signed / window_minutes, -1.0, 1.0)
            out[:, len(CATEGORIES) + slot] = visible.astype(np.float32)
        return out
