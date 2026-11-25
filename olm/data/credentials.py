"""Credential discovery, usage accounting and conservative request scheduling.

Security rules enforced here:
  * keys are read only from environment variables / .env, never from datasets;
  * key material is never included in logs or manifests;
  * multi-account aggregate pooling is disabled unless explicitly authorized;
  * a 429 pauses that account; rejected requests are not rerouted to another.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from olm.logging_utils import get_logger, redact

logger = get_logger(__name__)

KEY_ENV_PATTERN = 'TWELVE_DATA_API_KEY_{index}'
MAX_KEYS = 7


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass
class Credential:
    label: str
    key: str = field(repr=False)
    plan: str | None = None
    plan_limit: int | None = None
    daily_limit: int | None = None
    daily_used: int = 0
    requests_made: int = 0
    failures: int = 0
    validated: bool = False
    last_error: str | None = None
    cooldown_until: float = 0.0
    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    minute_window: deque = field(default_factory=deque, repr=False)

    def masked(self) -> str:
        return f'{self.key[:4]}...{self.key[-2:]}' if len(self.key) > 8 else '***'

    def roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.daily_used = 0


@dataclass
class AuthorizationGates:
    multi_account_pool: bool = False
    model_training: bool = False
    data_retention: bool = False
    commercial_use: bool = False

    @classmethod
    def from_env(cls) -> 'AuthorizationGates':
        return cls(
            multi_account_pool=_env_bool('TD_MULTI_ACCOUNT_POOL_AUTHORIZED'),
            model_training=_env_bool('TD_MODEL_TRAINING_APPROVED'),
            data_retention=_env_bool('TD_DATA_RETENTION_APPROVED'),
            commercial_use=_env_bool('TD_COMMERCIAL_USE_APPROVED'),
        )

    def require(self, gate: str, operation: str) -> None:
        if not getattr(self, gate):
            env = {
                'model_training': 'TD_MODEL_TRAINING_APPROVED',
                'data_retention': 'TD_DATA_RETENTION_APPROVED',
                'commercial_use': 'TD_COMMERCIAL_USE_APPROVED',
                'multi_account_pool': 'TD_MULTI_ACCOUNT_POOL_AUTHORIZED',
            }[gate]
            raise PermissionError(
                redact(
                    f'{operation} is blocked: authorization gate {env} is false. '
                    'This is an operational safeguard; enable it only after written '
                    'permission from the data provider and account owners.'
                )
            )


class RequestScheduler:
    """Token-bucket style scheduler with per-account minute and daily budgets."""

    def __init__(
        self,
        credentials: list[Credential],
        pooling_enabled: bool = False,
        default_per_minute: int = 8,
        safety_margin: float = 0.20,
        daily_override: int | None = None,
    ) -> None:
        if not credentials:
            raise ValueError('at least one credential is required')
        self.credentials = credentials
        self.pooling_enabled = pooling_enabled
        self.default_per_minute = max(1, int(default_per_minute))
        self.safety_margin = min(max(safety_margin, 0.0), 0.9)
        self.daily_override = daily_override
        self._lock = threading.Lock()
        self._cursor = 0

    def _effective_per_minute(self, credential: Credential) -> int:
        budget = self.default_per_minute
        return max(1, int(budget * (1.0 - self.safety_margin)))

    def _available(self, credential: Credential, now: float) -> bool:
        if now < credential.cooldown_until:
            return False
        credential.roll_day()
        daily_cap = self.daily_override or credential.daily_limit
        if daily_cap is not None and credential.daily_used >= daily_cap:
            return False
        while credential.minute_window and now - credential.minute_window[0] > 60.0:
            credential.minute_window.popleft()
        return len(credential.minute_window) < self._effective_per_minute(credential)

    def _eligible(self) -> list[Credential]:
        if not self.pooling_enabled:
            return [self.credentials[0]]
        return list(self.credentials)

    def acquire(self, timeout: float = 120.0) -> Credential:
        """Block until a request slot is available, or raise TimeoutError."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.time()
                candidates = [c for c in self._eligible() if self._available(c, now)]
                if candidates:
                    credential = candidates[self._cursor % len(candidates)]
                    self._cursor += 1
                    credential.minute_window.append(now)
                    credential.daily_used += 1
                    credential.requests_made += 1
                    logger.debug('request slot granted for %s', credential.label)
                    return credential
            if time.monotonic() >= deadline:
                raise TimeoutError('no API capacity available before timeout; try again later')
            time.sleep(0.25)

    def report_rate_limited(self, credential: Credential, retry_after: float = 60.0) -> None:
        with self._lock:
            credential.cooldown_until = max(credential.cooldown_until, time.time() + max(1.0, retry_after))
            logger.warning('%s rate-limited; account paused for %.0fs', credential.label, retry_after)

    def report_failure(self, credential: Credential, message: str) -> None:
        with self._lock:
            credential.failures += 1
            credential.last_error = redact(message)[:200]

    def usage_report(self) -> list[dict]:
        report = []
        with self._lock:
            now = time.time()
            for credential in self.credentials:
                credential.roll_day()
                while credential.minute_window and now - credential.minute_window[0] > 60.0:
                    credential.minute_window.popleft()
                daily_cap = self.daily_override or credential.daily_limit
                report.append({
                    'label': credential.label,
                    'key': credential.masked(),
                    'validated': credential.validated,
                    'plan': credential.plan,
                    'plan_limit': credential.plan_limit,
                    'daily_limit': daily_cap,
                    'daily_used': credential.daily_used,
                    'remaining': (daily_cap - credential.daily_used) if daily_cap is not None else None,
                    'requests_made': credential.requests_made,
                    'failures': credential.failures,
                    'requests_last_minute': len(credential.minute_window),
                    'per_minute_budget': self._effective_per_minute(credential),
                    'cooldown_seconds': max(0.0, round(credential.cooldown_until - now, 1)),
                })
        return report


def discover_credentials(load_dotenv_file: bool = True) -> list[Credential]:
    """Read TWELVE_DATA_API_KEY_1..7 from the environment (optionally .env)."""
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except Exception:
            pass
    found: list[Credential] = []
    for index in range(1, MAX_KEYS + 1):
        value = os.environ.get(KEY_ENV_PATTERN.format(index=index), '').strip()
        if not value:
            continue
        if len(value) < 16:
            logger.warning('credential %d is shorter than expected; ignoring it', index)
            continue
        found.append(Credential(label=f'account_{index}', key=value))
    return found


def build_scheduler(gates: AuthorizationGates | None = None) -> RequestScheduler:
    gates = gates or AuthorizationGates.from_env()
    credentials = discover_credentials()
    if not credentials:
        raise RuntimeError(
            'no Twelve Data credentials found. Copy .env.example to .env and set '
            'TWELVE_DATA_API_KEY_1, or run with synthetic data only.'
        )
    if gates.multi_account_pool and len(credentials) > 1:
        logger.info('multi-account pooling enabled for %d authorized accounts', len(credentials))
    elif len(credentials) > 1:
        logger.info(
            '%d credentials configured; pooling disabled, using %s only',
            len(credentials),
            credentials[0].label,
        )
    per_minute = int(os.environ.get('TD_MAX_REQUESTS_PER_MINUTE', '8'))
    margin = float(os.environ.get('TD_SAFETY_MARGIN', '0.20'))
    return RequestScheduler(
        credentials,
        pooling_enabled=gates.multi_account_pool,
        default_per_minute=per_minute,
        safety_margin=margin,
    )
