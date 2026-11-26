"""Historical retrieval: chunking, resume, incremental sync and manifests."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from olm.data.credentials import AuthorizationGates
from olm.data.providers.twelvedata import ProviderError, TwelveDataClient
from olm.data.storage import DataPaths, sha256_file, utc_now_iso, write_manifest, write_parquet_atomic
from olm.logging_utils import get_logger

logger = get_logger(__name__)

INTERVAL_MINUTES = {'1min': 1, '5min': 5, '15min': 15, '1h': 60, '4h': 240}
MAX_RECORDS_PER_REQUEST = 5000


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

    def key(self) -> str:
        return f'{self.start.isoformat()}_{self.end.isoformat()}'

    def api_start(self) -> str:
        return self.start.strftime('%Y-%m-%d %H:%M:%S')

    def api_end(self) -> str:
        return self.end.strftime('%Y-%m-%d %H:%M:%S')


def plan_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    max_records: int = MAX_RECORDS_PER_REQUEST,
    safety: float = 0.90,
) -> list[Window]:
    """Partition [start, end) into non-overlapping, request-sized windows."""
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f'unsupported interval {interval!r}')
    minutes = INTERVAL_MINUTES[interval]
    start = pd.Timestamp(start, tz='UTC') if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert('UTC')
    end = pd.Timestamp(end, tz='UTC') if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert('UTC')
    if start >= end:
        return []
    span = pd.Timedelta(minutes=minutes * int(max_records * safety))
    windows = []
    cursor = start.ceil(f'{minutes}min')
    while cursor < end:
        window_end = min(cursor + span, end)
        windows.append(Window(cursor, window_end))
        cursor = window_end
    return windows


def _to_utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')


def _safe_name(symbol: str) -> str:
    return symbol.replace('/', '_').upper()


class IngestionService:
    def __init__(self, client: TwelveDataClient, data_root: str = 'data') -> None:
        self.client = client
        self.paths = DataPaths(Path(data_root)).ensure()
        self._state_lock = threading.Lock()

    def _partition_dir(self, symbol: str, interval: str) -> Path:
        path = self.paths.raw / _safe_name(symbol) / interval
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_path(self, symbol: str, interval: str) -> Path:
        return self.paths.manifests / f'ingest_{_safe_name(symbol)}_{interval}.json'

    def _load_state(self, symbol: str, interval: str) -> dict:
        path = self._state_path(symbol, interval)
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        return {'symbol': symbol, 'interval': interval, 'completed': {}, 'metadata': {}}

    def _save_state(self, symbol: str, interval: str, state: dict) -> None:
        with self._state_lock:
            write_manifest(self._state_path(symbol, interval), state)

    def verify_symbol(self, symbol: str = 'XAU/USD') -> dict:
        candidates = self.client.symbol_search(symbol)
        matches = [c for c in candidates if str(c.get('symbol', '')).upper() == symbol.upper()]
        if not matches:
            raise ProviderError(f'{symbol} not returned by symbol_search')
        return {'requested': symbol, 'matches': matches}

    def probe(self, symbol: str, interval: str) -> dict:
        earliest = self.client.earliest_timestamp(symbol, interval)
        return {'symbol': symbol, 'interval': interval, 'earliest': earliest}

    def latest_staged_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        path = self.paths.staged / f'{_safe_name(symbol)}_{interval}.parquet'
        if not path.exists():
            return None
        frame = pd.read_parquet(path, columns=['timestamp'])
        if frame.empty:
            return None
        return pd.to_datetime(frame['timestamp'], utc=True).max()

    def plan(
        self,
        symbol: str = 'XAU/USD',
        interval: str = '5min',
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        max_windows: int | None = None,
        resume: bool = True,
        default_days: int = 30,
    ) -> dict:
        """Describe the request plan without credentials, network access or a gate check."""
        if interval not in INTERVAL_MINUTES:
            raise ValueError(f'unsupported interval {interval!r}')
        minutes = INTERVAL_MINUTES[interval]
        end = pd.Timestamp.now(tz='UTC').floor(f'{minutes}min') if end is None else _to_utc(end)
        source = 'argument'
        if start is None:
            latest = self.latest_staged_timestamp(symbol, interval)
            if latest is not None:
                start = latest + pd.Timedelta(minutes=minutes)
                source = 'resume'
            else:
                start = end - pd.Timedelta(days=default_days)
                source = f'default_{default_days}d'
        start = _to_utc(start)
        state = self._load_state(symbol, interval) if resume else {
            'symbol': symbol, 'interval': interval, 'completed': {}, 'metadata': {}
        }
        windows = plan_windows(start, end, interval)
        pending = [w for w in windows if not resume or w.key() not in state['completed']]
        if max_windows is not None:
            pending = pending[:max_windows]
        return {
            'status': 'dry-run',
            'symbol': symbol,
            'interval': interval,
            'start': start.isoformat(),
            'end': end.isoformat(),
            'start_source': source,
            'planned_windows': len(windows),
            'pending_windows': len(pending),
            'estimated_requests': len(pending),
            'resume': resume,
            'first_window': pending[0].key() if pending else None,
            'last_window': pending[-1].key() if pending else None,
            'network_calls': 0,
        }

    def download(
        self,
        symbol: str = 'XAU/USD',
        interval: str = '5min',
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        gates: AuthorizationGates | None = None,
        resume: bool = True,
        concurrency: int | None = None,
        max_windows: int | None = None,
    ) -> dict:
        """Download history into immutable raw partitions, then rebuild staged.

        Bulk retention is blocked unless TD_DATA_RETENTION_APPROVED=true.
        """
        gates = gates or AuthorizationGates.from_env()
        gates.require('data_retention', 'historical data retention')
        if interval not in INTERVAL_MINUTES:
            raise ValueError(f'unsupported interval {interval!r}')
        minutes = INTERVAL_MINUTES[interval]
        end = pd.Timestamp.now(tz='UTC').floor(f'{minutes}min') if end is None else _to_utc(end)
        if start is None:
            latest = self.latest_staged_timestamp(symbol, interval)
            if latest is not None:
                start = latest + pd.Timedelta(minutes=minutes)
            else:
                earliest = self.client.earliest_timestamp(symbol, interval)
                if earliest is None:
                    raise ProviderError('provider did not return an earliest timestamp')
                start = earliest
        start = _to_utc(start)
        if start >= end:
            return {'status': 'up-to-date', 'start': start, 'end': end, 'downloaded': 0}
        state = self._load_state(symbol, interval) if resume else {'symbol': symbol, 'interval': interval, 'completed': {}, 'metadata': {}}
        windows = plan_windows(start, end, interval)
        pending = [w for w in windows if not resume or w.key() not in state['completed']]
        if max_windows is not None:
            pending = pending[:max_windows]
        if not pending:
            return {'status': 'already-complete', 'windows': len(windows), 'downloaded': 0}
        concurrency = concurrency or int(os.environ.get('TD_MAX_CONCURRENCY', '4'))
        concurrency = max(1, min(concurrency, 7))
        downloaded = 0
        failures = []
        partition_dir = self._partition_dir(symbol, interval)

        def fetch(window: Window):
            payload = self.client.time_series(
                symbol=symbol,
                interval=interval,
                start_date=window.api_start(),
                # The provider end_date is inclusive. Windows are [start, end),
                # so stop one bar before the window end to avoid an incomplete
                # current candle and avoid duplicating the next window's first bar.
                end_date=(window.end - pd.Timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S'),
                outputsize=MAX_RECORDS_PER_REQUEST,
            )
            return window, payload

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(fetch, window): window for window in pending}
            for future in as_completed(futures):
                window = futures[future]
                try:
                    _, payload = future.result()
                except Exception as exc:
                    failures.append({'window': window.key(), 'error': str(exc)[:200]})
                    logger.warning('window %s failed: %s', window.key(), exc)
                    continue
                frame = payload.frame
                if not frame.empty:
                    partition = partition_dir / f'{window.start.strftime("%Y%m%dT%H%M")}_{window.end.strftime("%Y%m%dT%H%M")}.parquet'
                    frame = frame.copy()
                    frame['symbol'] = symbol
                    frame['interval'] = interval
                    frame['source'] = payload.meta.get('exchange') or 'twelvedata'
                    write_parquet_atomic(frame, partition)
                    downloaded += len(frame)
                state['completed'][window.key()] = {
                    'rows': int(len(frame)),
                    'retrieved_at': utc_now_iso(),
                    'meta': payload.meta,
                }
                state['metadata'].update({k: v for k, v in payload.meta.items() if v is not None})
                state['metadata']['provider'] = 'twelvedata'
                state['metadata']['retrieval_timestamp'] = utc_now_iso()
                self._save_state(symbol, interval, state)
        combined = self.rebuild_staged(symbol, interval, state)
        return {
            'status': 'ok' if not failures else 'partial',
            'windows': len(windows),
            'pending': len(pending),
            'downloaded': downloaded,
            'failures': failures,
            'staged': combined,
        }

    def rebuild_staged(self, symbol: str, interval: str, state: dict | None = None) -> dict:
        partition_dir = self._partition_dir(symbol, interval)
        files = sorted(partition_dir.glob('*.parquet'))
        if not files:
            return {'rows': 0, 'path': None}
        frames = [pd.read_parquet(path) for path in files]
        combined = pd.concat(frames, ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], utc=True)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
        target = self.paths.staged / f'{_safe_name(symbol)}_{interval}.parquet'
        write_parquet_atomic(combined, target)
        state = state or self._load_state(symbol, interval)
        manifest = {
            'dataset': f'{symbol} {interval}',
            'provider': 'twelvedata',
            'symbol': symbol,
            'interval': interval,
            'exchange': state.get('metadata', {}).get('exchange'),
            'timezone': 'UTC',
            'rows': int(len(combined)),
            'earliest': combined['timestamp'].min(),
            'latest': combined['timestamp'].max(),
            'sha256': sha256_file(target),
            'source_metadata': state.get('metadata', {}),
            'built_at': utc_now_iso(),
            'partitions': len(files),
        }
        write_manifest(self.paths.manifests / f'dataset_{_safe_name(symbol)}_{interval}.json', manifest)
        return {'rows': int(len(combined)), 'path': str(target), 'sha256': manifest['sha256']}


def summarize_dataset(path: str | Path) -> dict:
    frame = pd.read_parquet(path)
    ts = pd.to_datetime(frame['timestamp'], utc=True)
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    return {
        'path': str(path),
        'rows': int(len(frame)),
        'columns': list(frame.columns),
        'earliest': str(ts.min()),
        'latest': str(ts.max()),
        'interval_minutes': int(round(float(diffs.min()))) if len(diffs) else None,
        'duplicate_timestamps': int(ts.duplicated().sum()),
        'missing_bars': int(((diffs / diffs.min()) - 1).sum()) if len(diffs) else 0,
    }
