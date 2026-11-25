"""Lightweight local experiment store (JSON, no external services)."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


class ExperimentTracker:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str, params: dict, tags: dict | None = None) -> Path:
        run_id = f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:8]}'
        run_dir = self.root / f'{name}-{run_id}'
        run_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            'run_id': run_id,
            'name': name,
            'created_at': _now(),
            'params': params,
            'tags': tags or {},
            'git_commit': _git_commit(),
            'status': 'running',
        }
        (run_dir / 'run.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return run_dir

    def log_metrics(self, run_dir: str | Path, metrics: dict, step: int | None = None) -> None:
        path = Path(run_dir) / 'metrics.jsonl'
        record = {'logged_at': _now(), 'step': step, 'metrics': metrics}
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def finish_run(self, run_dir: str | Path, status: str = 'completed', summary: dict | None = None) -> None:
        path = Path(run_dir) / 'run.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['status'] = status
        payload['finished_at'] = _now()
        if summary is not None:
            payload['summary'] = summary
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def list_runs(self) -> list[dict]:
        runs = []
        for path in sorted(self.root.glob('*/run.json')):
            try:
                runs.append(json.loads(path.read_text(encoding='utf-8')))
            except json.JSONDecodeError:
                continue
        return runs


class ChampionRegistry:
    """Champion is only replaced when a challenger beats it on frozen criteria."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if not self.path.exists():
            return {'champion': None, 'history': []}
        return json.loads(self.path.read_text(encoding='utf-8'))

    def register_challenger(self, record: dict) -> dict:
        registry = self.load()
        registry.setdefault('history', []).append({'type': 'challenger', 'at': _now(), **record})
        self.path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
        return registry

    def promote(self, challenger: dict, criteria: dict) -> tuple[bool, str]:
        """Promote only when every required metric is present and better."""
        metrics = challenger.get('metrics', {})
        for key in criteria.get('required_metrics', []):
            if key not in metrics or metrics[key] != metrics[key]:
                return False, f'challenger is missing required metric {key}'
        if int(challenger.get('n_test', 0)) < int(criteria.get('min_test_samples', 1)):
            return False, 'challenger evaluation sample is too small'
        registry = self.load()
        champion = registry.get('champion')
        if champion is None:
            accepted, reason = True, 'no current champion'
        else:
            current = champion.get('metrics', {}).get(criteria.get('primary_metric', 'total'))
            candidate = metrics.get(criteria.get('primary_metric', 'total'))
            if current is None or candidate is None:
                return False, 'primary metric missing'
            margin = float(criteria.get('min_improvement', 0.0))
            accepted = candidate < current - margin
            reason = f'challenger {candidate:.6f} vs champion {current:.6f} (margin {margin})'
        record = {'type': 'promotion', 'at': _now(), 'accepted': accepted, 'reason': reason, **challenger}
        registry.setdefault('history', []).append(record)
        if accepted:
            registry['champion'] = {'promoted_at': _now(), 'reason': reason, **challenger}
        self.path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
        return accepted, reason
