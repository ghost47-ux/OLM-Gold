"""Timezone-aware clock and trading-session features (DST handled by zoneinfo)."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from olm.config import TemporalConfig
from olm.data.events import CATEGORIES, EventCalendar


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(':')
    return int(hour) * 60 + int(minute)


@dataclass
class TemporalFeatures:
    config: TemporalConfig
    event_calendar: EventCalendar | None = None

    def __post_init__(self) -> None:
        self._sessions = {}
        for name, spec in self.config.sessions.items():
            self._sessions[name] = (
                ZoneInfo(spec['tz']),
                _minutes(spec['open']),
                _minutes(spec['close']),
            )
        names = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        if self.config.include_sessions:
            names += [f'session_{name}' for name in self._sessions]
            names += ['session_overlap']
            for name in self._sessions:
                names += [f'{name}_since_open', f'{name}_until_close']
        if self.config.include_events:
            names += [f'event_{c.lower()}_minutes' for c in CATEGORIES]
            names += [f'event_{c.lower()}_visible' for c in CATEGORIES]
        self.feature_names = names
        self.dim = len(names)

    def compute(self, timestamps: pd.DatetimeIndex | pd.Series) -> np.ndarray:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        n = len(ts)
        columns = []
        hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
        dow = ts.day_of_week.to_numpy()
        columns.append(np.sin(2 * np.pi * hour / 24.0))
        columns.append(np.cos(2 * np.pi * hour / 24.0))
        columns.append(np.sin(2 * np.pi * dow / 7.0))
        columns.append(np.cos(2 * np.pi * dow / 7.0))
        if self.config.include_sessions:
            active = {}
            for name, (tz, open_min, close_min) in self._sessions.items():
                local = ts.tz_convert(tz)
                minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
                duration = close_min - open_min
                if duration <= 0:
                    duration += 24 * 60
                inside = (minute >= open_min) & (minute < close_min)
                active[name] = inside
                columns.append(inside.astype(np.float32))
            columns.append(sum(active.values()).astype(np.int32) >= 2)
            for name, (tz, open_min, close_min) in self._sessions.items():
                local = ts.tz_convert(tz)
                minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
                duration = close_min - open_min
                if duration <= 0:
                    duration += 24 * 60
                since = np.where(active[name], minute - open_min, 0) / duration
                until = np.where(active[name], close_min - minute, 0) / duration
                columns.append(since)
                columns.append(until)
        if self.config.include_events:
            if self.event_calendar is None:
                raise ValueError('event features requested but no event calendar was loaded')
            columns.append(self.event_calendar.features_for(ts, self.config.event_window_minutes))
        matrix = np.column_stack([np.asarray(c, dtype=np.float32).reshape(n, -1) for c in columns])
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def compute_one(self, timestamp) -> np.ndarray:
        return self.compute(pd.DatetimeIndex([pd.Timestamp(timestamp)])).reshape(-1)
