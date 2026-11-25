"""Filesystem layout, atomic Parquet writes, hashing and manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class DataPaths:
    root: Path = Path('data')

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw(self) -> Path:
        return self.root / 'raw'

    @property
    def staged(self) -> Path:
        return self.root / 'staged'

    @property
    def processed(self) -> Path:
        return self.root / 'processed'

    @property
    def features(self) -> Path:
        return self.root / 'features'

    @property
    def events(self) -> Path:
        return self.root / 'events'

    @property
    def manifests(self) -> Path:
        return self.root / 'manifests'

    def ensure(self) -> 'DataPaths':
        for path in (self.raw, self.staged, self.processed, self.features, self.events, self.manifests):
            path.mkdir(parents=True, exist_ok=True)
        return self


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix='.parquet', dir=str(target.parent))
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return target


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(**arrays) -> str:
    """Deterministic content hash over named arrays (dataset versioning)."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        digest.update(name.encode())
        if isinstance(value, pd.DataFrame):
            digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        elif isinstance(value, pd.Series):
            digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()[:16]


def write_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault('created_at', utc_now_iso())
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def query_duckdb(parquet_glob: str, sql: str):
    """Run a local analytical query against Parquet files with DuckDB."""
    import duckdb

    source = "read_parquet('" + parquet_glob + "')"
    statement = sql.replace('{source}', source)
    return duckdb.sql(statement).df()


def dataset_manifest_name(symbol: str, interval: str) -> str:
    safe = symbol.replace('/', '_').upper()
    return f'{safe}_{interval}.json'
