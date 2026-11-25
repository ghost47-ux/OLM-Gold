"""Structured logging with mandatory secret redaction."""

from __future__ import annotations

import logging
import os
import re
import sys

_HEX_KEY = re.compile(r'\b[0-9a-fA-F]{32}\b')
_API_PARAM = re.compile(r'(apikey|api_key|token)=([^&\s]+)', re.IGNORECASE)
_BEARER = re.compile(r'Bearer\s+[A-Za-z0-9._\-]+')


def _known_secrets() -> list[str]:
    secrets = []
    for name, value in os.environ.items():
        upper = name.upper()
        if not value or len(value) < 8:
            continue
        if 'API_KEY' in upper or upper.endswith('_TOKEN') or upper.endswith('_SECRET'):
            secrets.append(value)
    return secrets


def redact(text: str) -> str:
    """Remove credential-shaped strings from any text before it is logged or stored."""
    if not text:
        return text
    result = _API_PARAM.sub(r'\1=***', text)
    result = _BEARER.sub('Bearer ***', result)
    for secret in _known_secrets():
        result = result.replace(secret, '***')
    return _HEX_KEY.sub('***', result)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact(message)
        record.args = ()
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(h, '_olm_handler', False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler._olm_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(name)s | %(message)s'))
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
