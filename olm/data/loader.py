"""Locate and load prepared market frames for training and inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from olm.config import Config
from olm.data.aggregation import build_resolutions
from olm.data.storage import DataPaths, read_manifest

RESOLUTIONS = ('5min', '15min', '1h', '4h')


class DataNotFoundError(FileNotFoundError):
    pass


def _synthetic_path(paths: DataPaths, interval: str) -> Path:
    return paths.raw / f'XAUUSD_synthetic_{interval}.parquet'


def _real_path(paths: DataPaths, symbol: str, interval: str) -> Path:
    return paths.staged / f'{symbol.replace("/", "_").upper()}_{interval}.parquet'


def load_market_frames(config: Config, source: str = 'auto') -> tuple[dict[str, pd.DataFrame], dict]:
    """Return {'5min','15min','1h','4h'} frames plus provenance metadata."""
    paths = DataPaths(Path(config.data.data_root))
    synthetic_available = all(_synthetic_path(paths, interval).exists() for interval in RESOLUTIONS)
    real_available = all(_real_path(paths, config.data.symbol, interval).exists() for interval in RESOLUTIONS)
    if source == 'auto':
        source = 'synthetic' if synthetic_available else ('twelvedata' if real_available else 'none')
    frames = {}
    if source == 'synthetic':
        for interval in RESOLUTIONS:
            path = _synthetic_path(paths, interval)
            if not path.exists():
                raise DataNotFoundError(f'missing synthetic dataset {path}')
            frames[interval] = pd.read_parquet(path)
        info = {'source': 'synthetic', 'provider': 'synthetic-fixture', 'symbol': 'XAU/USD', 'licensed': False}
    elif source == 'twelvedata':
        for interval in RESOLUTIONS:
            path = _real_path(paths, config.data.symbol, interval)
            if not path.exists():
                raise DataNotFoundError(
                    f'missing staged dataset {path}. Run `python -m olm download-history` '
                    'after the data-retention gate is approved.'
                )
            frames[interval] = pd.read_parquet(path)
        manifest_path = paths.manifests / f'dataset_{config.data.symbol.replace("/", "_").upper()}_5min.json'
        manifest = read_manifest(manifest_path) if manifest_path.exists() else {}
        info = {
            'source': 'twelvedata',
            'provider': 'twelvedata',
            'symbol': config.data.symbol,
            'licensed': False,
            'manifest': manifest,
        }
    elif source == 'm1':
        path = paths.raw / 'XAUUSD_synthetic_1min.parquet'
        if not path.exists():
            raise DataNotFoundError(f'missing M1 dataset {path}')
        frames = build_resolutions(pd.read_parquet(path))
        info = {'source': 'synthetic-m1', 'provider': 'synthetic-fixture', 'symbol': 'XAU/USD', 'licensed': False}
    else:
        raise DataNotFoundError(
            'no market data found. Create synthetic fixtures with '
            '`python -m olm prepare-datasets --synthetic`, or download approved history.'
        )
    for interval, frame in frames.items():
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
        frames[interval] = frame.sort_values('timestamp').reset_index(drop=True)
    info['rows'] = {interval: int(len(frame)) for interval, frame in frames.items()}
    info['range'] = {
        interval: [str(frame['timestamp'].min()), str(frame['timestamp'].max())]
        for interval, frame in frames.items()
    }
    return frames, info
