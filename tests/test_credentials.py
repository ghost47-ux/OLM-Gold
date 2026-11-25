"""Credential handling, redaction and request scheduling."""

import os

import pytest

from olm.data.credentials import Credential, RequestScheduler, discover_credentials
from olm.logging_utils import redact


def test_redaction_hides_key_patterns():
    key = 'a' * 32
    text = f'apikey={key} Authorization: Bearer {key} {key}'
    cleaned = redact(text)
    assert key not in cleaned
    assert '***' in cleaned


def test_discover_credentials_reads_numbered_keys(monkeypatch):
    monkeypatch.setenv('TWELVE_DATA_API_KEY_1', 'k' * 32)
    monkeypatch.setenv('TWELVE_DATA_API_KEY_3', 'm' * 32)
    os.environ.pop('TWELVE_DATA_API_KEY_2', None)
    found = discover_credentials(load_dotenv_file=False)
    assert [c.label for c in found] == ['account_1', 'account_3']
    assert found[0].masked() != 'k' * 32


def test_scheduler_uses_first_account_without_pooling():
    credentials = [Credential('a', 'x' * 32), Credential('b', 'y' * 32)]
    scheduler = RequestScheduler(credentials, pooling_enabled=False)
    assert scheduler.acquire().label == 'a'
    assert scheduler.acquire().label == 'a'


def test_pooling_spreads_requests_when_authorized():
    credentials = [Credential('a', 'x' * 32), Credential('b', 'y' * 32)]
    scheduler = RequestScheduler(credentials, pooling_enabled=True)
    labels = {scheduler.acquire().label for _ in range(2)}
    assert labels == {'a', 'b'}


def test_daily_cap_blocks_account():
    credential = Credential('a', 'x' * 32, daily_limit=1)
    scheduler = RequestScheduler([credential], pooling_enabled=False)
    scheduler.acquire()
    with pytest.raises(TimeoutError):
        scheduler.acquire(timeout=0.1)


def test_rate_limit_pauses_account():
    credential = Credential('a', 'x' * 32)
    scheduler = RequestScheduler([credential], pooling_enabled=False)
    scheduler.report_rate_limited(credential, retry_after=60)
    with pytest.raises(TimeoutError):
        scheduler.acquire(timeout=0.1)
    assert scheduler.usage_report()[0]['cooldown_seconds'] > 0


def test_usage_report_never_contains_full_key():
    credential = Credential('a', 'z' * 32)
    report = RequestScheduler([credential]).usage_report()[0]
    assert 'z' * 32 not in str(report)
