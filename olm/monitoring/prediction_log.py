"""Immutable, append-only forecast log with retrospective evaluation."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from olm.logging_utils import redact


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionLog:
    """JSONL log. Entries are never rewritten; actuals are appended as new records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, payload: dict) -> dict:
        payload = dict(payload)
        payload['logged_at'] = _now()
        encoded = redact(json.dumps(payload, default=str))
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
        return payload

    def record_forecast(self, forecast: dict, model_version: str, as_of: str) -> dict:
        """Persist the complete forecast payload (scenarios plus diagnostics)."""
        body = json.dumps(forecast, sort_keys=True, default=str).encode('utf-8')
        payload = {
            'type': 'forecast',
            'forecast_id': uuid.uuid4().hex[:24],
            'as_of': as_of,
            'model_version': model_version,
            'forecast_hash': hashlib.sha256(body).hexdigest()[:16],
            'forecast': forecast,
        }
        return self._append(payload)

    def record_actual(self, forecast_id: str, candles: list[dict]) -> dict:
        return self._append({'type': 'actual', 'forecast_id': forecast_id, 'candles': candles})

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def evaluate_forecast(self, forecast_id: str, actual: list[dict]) -> dict:
        entries = self.read_all()
        match = next((e for e in entries if e.get('forecast_id') == forecast_id and e.get('type') == 'forecast'), None)
        if match is None:
            raise KeyError(f'unknown forecast id {forecast_id}')
        scenarios = match['forecast'].get('scenarios', [])
        if not scenarios:
            raise ValueError('forecast record contains no scenarios')
        actual_parent = {
            'open': float(actual[0]['open']),
            'high': max(float(candle['high']) for candle in actual),
            'low': min(float(candle['low']) for candle in actual),
            'close': float(actual[-1]['close']),
        }
        per_key = []
        for key in ('open', 'high', 'low', 'close'):
            values = np.asarray([float(row['parent'][key]) for row in scenarios])
            per_key.append(abs(float(values.mean()) - float(actual_parent[key])))
        errors = dict(zip(('open', 'high', 'low', 'close'), per_key))
        evaluation = {
            'forecast_id': forecast_id,
            'mae': float(np.mean(per_key)),
            'errors': errors,
            'scenarios': len(scenarios),
            'note': 'Point error of the scenario ensemble mean; not a probabilistic score.',
        }
        self._append({'type': 'evaluation', **evaluation})
        return evaluation
