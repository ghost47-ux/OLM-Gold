"""Official Twelve Data HTTP adapter (time_series, api_usage, symbol_search)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from olm.data.credentials import Credential, RequestScheduler
from olm.logging_utils import get_logger, redact

logger = get_logger(__name__)


class ProviderError(RuntimeError):
    pass


class RateLimitError(ProviderError):
    pass


class AuthenticationError(ProviderError):
    pass


@dataclass
class TimeSeriesPayload:
    frame: pd.DataFrame
    meta: dict = field(default_factory=dict)


def parse_time_series(payload: dict) -> TimeSeriesPayload:
    if not isinstance(payload, dict):
        raise ProviderError('time_series payload is not an object')
    if payload.get('status') == 'error' or 'values' not in payload:
        raise ProviderError(f"provider error: {payload.get('message', 'unknown error')}")
    values = payload.get('values') or []
    if not isinstance(values, list):
        raise ProviderError('values must be a list')
    rows = []
    for item in values:
        if not isinstance(item, dict):
            raise ProviderError('malformed value row')
        row = {'timestamp': item.get('datetime')}
        for name in ('open', 'high', 'low', 'close', 'volume'):
            if name in item:
                row[name] = item[name]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    for name in ('open', 'high', 'low', 'close'):
        if name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors='coerce')
    if 'volume' in frame.columns:
        frame['volume'] = pd.to_numeric(frame['volume'], errors='coerce')
    frame = frame.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    source_meta = payload.get('meta') if isinstance(payload.get('meta'), dict) else {}
    meta = {
        key: source_meta.get(key, payload.get(key))
        for key in ('symbol', 'interval', 'currency', 'exchange_timezone', 'exchange', 'type')
    }
    meta['raw_rows'] = len(values)
    return TimeSeriesPayload(frame=frame, meta=meta)


class TwelveDataClient:
    """Blocking client. Every request passes through the central scheduler."""

    def __init__(
        self,
        scheduler: RequestScheduler,
        base_url: str = 'https://api.twelvedata.com',
        timeout: float = 30.0,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        session: requests.Session | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self._local = threading.local()
        self._sessions: list[requests.Session] = []
        self._sessions_lock = threading.Lock()
        if session is not None:
            self._local.session = session
            self._sessions.append(session)

    def _session(self) -> requests.Session:
        """Return a session owned by the current thread.

        requests.Session is not documented as thread-safe; parallel partition
        downloads therefore get one session per worker instead of sharing a
        connection pool and cookie jar.
        """
        current = getattr(self._local, 'session', None)
        if current is None:
            current = requests.Session()
            self._local.session = current
            with self._sessions_lock:
                self._sessions.append(current)
        return current

    def close(self) -> None:
        with self._sessions_lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()
        self._local = threading.local()

    def __enter__(self) -> 'TwelveDataClient':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict, credential: Credential | None = None) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            credential = credential or self.scheduler.acquire()
            request_params = dict(params)
            request_params['apikey'] = credential.key
            try:
                response = self._session().get(
                    f'{self.base_url}{endpoint}', params=request_params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = ProviderError(redact(f'network error: {exc}'))
                self.scheduler.report_failure(credential, str(last_error))
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            if response.status_code == 429:
                retry_after = _retry_after_seconds(response, self.retry_delay * (attempt + 1) * 2)
                self.scheduler.report_rate_limited(credential, retry_after)
                last_error = RateLimitError('provider rate limit (HTTP 429)')
                time.sleep(min(retry_after, 120.0))
                continue
            if response.status_code in {401, 403}:
                message = redact(response.text[:200])
                credential.last_error = message
                raise AuthenticationError(f'credential {credential.label} rejected: {message}')
            if response.status_code >= 500:
                last_error = ProviderError(f'server error {response.status_code}')
                self.scheduler.report_failure(credential, str(last_error))
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError('provider returned non-JSON content') from exc
            if isinstance(payload, dict) and payload.get('status') == 'error':
                code = payload.get('code')
                message = redact(str(payload.get('message', 'unknown error')))
                if code == 429 or 'limit' in message.lower():
                    self.scheduler.report_rate_limited(credential, 60.0)
                    last_error = RateLimitError(message)
                    time.sleep(min(60.0, self.retry_delay * (attempt + 1) * 2))
                    continue
                if code in {401, 403}:
                    raise AuthenticationError(f'credential {credential.label} rejected: {message}')
                raise ProviderError(f'provider error {code}: {message}')
            return payload
        raise last_error or ProviderError('request failed after retries')

    def api_usage(self) -> dict:
        credential = self.scheduler.acquire()
        payload = self._get('/api_usage', {}, credential=credential)
        credential.validated = True
        credential.plan = payload.get('plan_category') or payload.get('plan') or credential.plan
        credential.plan_limit = _as_int(payload.get('plan_limit')) or credential.plan_limit
        credential.daily_limit = _as_int(payload.get('daily_limit')) or credential.daily_limit
        return {
            'account': credential.label,
            'plan': credential.plan,
            'plan_limit': credential.plan_limit,
            'current_usage': _as_int(payload.get('current_usage')),
            'daily_usage': _as_int(payload.get('daily_usage')),
            'daily_limit': credential.daily_limit,
        }

    def symbol_search(self, symbol: str) -> list[dict]:
        payload = self._get('/symbol_search', {'symbol': symbol})
        data = payload.get('data')
        if data is None:
            return []
        if not isinstance(data, list):
            raise ProviderError('symbol_search returned malformed data')
        return data

    def earliest_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        payload = self._get('/earliest_timestamp', {'symbol': symbol, 'interval': interval})
        value = payload.get('datetime')
        return pd.to_datetime(value, utc=True) if value else None

    def time_series(
        self,
        symbol: str,
        interval: str,
        start_date: str | None = None,
        end_date: str | None = None,
        outputsize: int = 5000,
    ) -> TimeSeriesPayload:
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': min(int(outputsize), 5000),
            'timezone': 'UTC',
            'order': 'ASC',
            'format': 'JSON',
        }
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        payload = self._get('/time_series', params)
        return parse_time_series(payload)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
    header = response.headers.get('Retry-After')
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return fallback
