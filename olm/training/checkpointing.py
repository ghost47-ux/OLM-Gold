"""Atomic, complete checkpoints (weights, optimizer, RNG, preprocessing, hashes)."""

from __future__ import annotations

import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from olm import __version__
from olm.config import Config, ModelConfig
from olm.data.encoding import Normalizer
from olm.models.olm import OLMModel, build_model


def _rng_state() -> dict:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path: str | Path, *, model: OLMModel, optimizer, scheduler, epoch: int, global_step: int,
                    config: Config, normalizer: Normalizer, dataset_meta: dict, best_val: float, extra: dict | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'olm-checkpoint-v1',
        'olm_version': __version__,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'best_val': best_val,
        'model_config': model.config_dict(),
        'training_config': config.training.__dict__,
        'data_config': config.data.__dict__,
        'normalizer': normalizer.to_dict(),
        'dataset_meta': dataset_meta,
        'rng_state': _rng_state(),
        'torch_version': torch.__version__,
        'extra': extra or {},
    }
    fd, tmp = tempfile.mkstemp(suffix='.pt', dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def load_checkpoint(path: str | Path, map_location: str = 'cpu') -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get('format') != 'olm-checkpoint-v1':
        raise ValueError(f'{path} is not an OLM checkpoint')
    return checkpoint


def model_from_checkpoint(checkpoint: dict, map_location: str = 'cpu') -> OLMModel:
    config_dict = dict(checkpoint['model_config'])
    temporal_dim = int(config_dict.pop('temporal_dim'))
    model = build_model(ModelConfig(**config_dict), temporal_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.to(map_location)
    model.eval()
    return model


def normalizer_from_checkpoint(checkpoint: dict) -> Normalizer:
    return Normalizer.from_dict(checkpoint['normalizer'])
