"""Typed configuration objects with YAML loading and strict key validation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    symbol: str = 'XAU/USD'
    provider: str = 'twelvedata'
    exchange: str = 'Commodity Aggregate'
    timezone: str = 'UTC'
    base_interval: str = '5min'
    parent_interval: str = '15min'
    context_intervals: list[str] = field(default_factory=lambda: ['1h', '4h'])
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    embargo_minutes: int = 60
    cache_dir: str = 'cache'
    data_root: str = 'data'
    synthetic_days: int = 90
    synthetic_seed: int = 7

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'split fractions must sum to 1.0, got {total}')
        if min(self.train_fraction, self.val_fraction, self.test_fraction) <= 0:
            raise ValueError('all split fractions must be positive')


@dataclass
class ModelConfig:
    profile: str = 'cpu-research'
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    n_components: int = 4
    min_scale: float = 1e-4
    context_encoder_layers: int = 2
    use_context: bool = True
    use_temporal: bool = True
    use_events: bool = False
    use_xfactor: bool = False
    latent_dim: int = 8
    xfactor_beta: float = 0.01
    m5_context: int = 192
    m15_context: int = 64
    h1_context: int = 32
    h4_context: int = 16
    n_scenarios: int = 32

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if self.n_components < 1:
            raise ValueError('n_components must be >= 1')
        if self.m5_context < 3:
            raise ValueError('m5_context must be at least 3')


@dataclass
class TrainingConfig:
    seed: int = 7
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    max_steps: int = 0
    lr: float = 6e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    scheduler: str = 'cosine'
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05
    patience: int = 6
    min_delta: float = 1e-4
    num_workers: int = 0
    lambda_parent: float = 1.0
    lambda_child: float = 1.0
    lambda_aggregate: float = 0.5
    lambda_consistency: float = 0.25
    checkpoint_dir: str = 'artifacts/checkpoints'
    run_dir: str = 'artifacts/runs'
    log_every: int = 20
    device: str = 'auto'


@dataclass
class SessionConfig:
    tz: str
    open: str
    close: str


@dataclass
class TemporalConfig:
    enabled: bool = True
    include_sessions: bool = True
    include_events: bool = False
    events_path: str = 'data/events/us_events.csv'
    event_window_minutes: int = 480
    sessions: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            'asia': {'tz': 'Asia/Tokyo', 'open': '09:00', 'close': '18:00'},
            'london': {'tz': 'Europe/London', 'open': '08:00', 'close': '16:30'},
            'new_york': {'tz': 'America/New_York', 'open': '09:30', 'close': '16:00'},
        }
    )


@dataclass
class InferenceConfig:
    checkpoint: str = 'artifacts/checkpoints/best.pt'
    scenarios: int = 32
    seed: int | None = None
    prediction_log: str = 'artifacts/predictions/predictions.jsonl'
    host: str = '127.0.0.1'
    port: int = 8000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')


DEFAULT_CONFIG_FILES = [
    'configs/data.yaml',
    'configs/model.yaml',
    'configs/training.yaml',
    'configs/temporal.yaml',
    'configs/inference.yaml',
]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_dataclass(instance: Any, values: dict, prefix: str = '') -> None:
    known = {f.name: f for f in fields(instance)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f'unknown configuration key {prefix}{key}')
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dataclass(current, value, prefix=f'{prefix}{key}.')
        else:
            setattr(instance, key, value)
    instance.__post_init__() if hasattr(instance, '__post_init__') else None


def load_config(paths: list[str | Path] | None = None) -> Config:
    """Load one or more YAML files in order, merging into typed defaults."""
    config = Config()
    raw: dict[str, Any] = {}
    candidates = paths if paths is not None else DEFAULT_CONFIG_FILES
    for path in candidates:
        p = Path(path)
        if not p.exists():
            if paths is None:
                continue
            raise FileNotFoundError(f'configuration file not found: {p}')
        loaded = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'{p} must contain a mapping at the top level')
        raw = _deep_merge(raw, loaded)
    _apply_dataclass(config, raw)
    return config
