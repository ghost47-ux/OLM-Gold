"""One-shot data preparation: validate, split chronologically, persist metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from olm.config import Config
from olm.data.calendar import TemporalFeatures
from olm.data.datasets import DatasetBundle, build_datasets, leakage_check
from olm.data.loader import load_market_frames
from olm.data.storage import DataPaths, hash_arrays
from olm.data.synthetic import write_synthetic_dataset
from olm.data.validation import validate_frame
from olm.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_datasets(
    config: Config,
    source: str = 'auto',
    synthetic_days: int = 60,
    force_synthetic: bool = False,
    validate: bool = True,
) -> tuple[DatasetBundle, dict]:
    paths = DataPaths(Path(config.data.data_root)).ensure()
    if source == 'synthetic':
        expected = [paths.raw / f'XAUUSD_synthetic_{name}.parquet' for name in ('5min', '15min', '1h', '4h')]
        if force_synthetic or not all(path.exists() for path in expected):
            write_synthetic_dataset(paths.root, days=synthetic_days, seed=config.data.synthetic_seed)
    frames, info = load_market_frames(config, source=source)
    reports = {}
    if validate:
        expected_minutes = {'5min': 5, '15min': 15, '1h': 60, '4h': 240}
        for interval, frame in frames.items():
            report = validate_frame(frame, expected_minutes[interval])
            reports[interval] = {'ok': report.ok, 'stats': report.stats, 'errors': report.errors, 'warnings': report.warnings}
            if not report.ok:
                raise ValueError(f'{interval} data failed validation: ' + '; '.join(report.errors))
    temporal = TemporalFeatures(config.temporal)
    bundle = build_datasets(config, frames, temporal)
    leakage = leakage_check(bundle)
    if not leakage['ok']:
        raise ValueError(f'chronological split violation: {leakage}')
    manifest = {
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'source': info.get('source'),
        'provider': info.get('provider'),
        'symbol': config.data.symbol,
        'rows': info.get('rows'),
        'range': info.get('range'),
        'boundaries': bundle.boundaries,
        'meta': bundle.meta,
        'normalizer': bundle.normalizer.to_dict(),
        'normalizer_hash': hash_arrays(mean=bundle.normalizer.mean, std=bundle.normalizer.std),
        'temporal': config.temporal.__dict__,
        'validation': reports,
        'leakage': leakage,
        'synthetic': info.get('source') in ('synthetic', 'synthetic-m1'),
    }
    manifest_path = paths.manifests / f'prepared_{config.data.symbol.replace("/", "_").upper()}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    logger.info('prepared %d train / %d val / %d test samples', len(bundle.train), len(bundle.val), len(bundle.test))
    return bundle, manifest
