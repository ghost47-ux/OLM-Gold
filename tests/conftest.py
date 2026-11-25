"""Shared tiny fixtures; no network or provider credentials."""

import numpy as np
import pandas as pd
import pytest

from olm.config import Config, ModelConfig


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0, n_components=2,
        context_encoder_layers=1, m5_context=6, m15_context=3, h1_context=3, h4_context=2,
        use_context=True, use_temporal=True,
    )


@pytest.fixture
def tiny_config(tiny_model_config) -> Config:
    config = Config()
    config.model = tiny_model_config
    config.training.epochs = 1
    config.training.batch_size = 4
    config.training.eval_batch_size = 8
    config.training.log_every = 1
    config.training.run_dir = 'artifacts/test_runs'
    config.training.checkpoint_dir = 'artifacts/test_checkpoints'
    return config


@pytest.fixture
def synthetic_frames():
    from olm.data.aggregation import build_resolutions
    from olm.data.synthetic import generate_m1

    return build_resolutions(generate_m1(days=4, seed=11))


@pytest.fixture
def tiny_batch(tiny_model_config):
    import torch

    rng = np.random.default_rng(0)
    m5 = torch.tensor(rng.normal(size=(2, tiny_model_config.m5_context, 4)), dtype=torch.float32)
    m15 = torch.tensor(rng.normal(size=(2, tiny_model_config.m15_context, 4)), dtype=torch.float32)
    h1 = torch.tensor(rng.normal(size=(2, tiny_model_config.h1_context, 4)), dtype=torch.float32)
    h4 = torch.tensor(rng.normal(size=(2, tiny_model_config.h4_context, 4)), dtype=torch.float32)
    temporal = lambda length: torch.zeros((2, length, 14))
    return {
        'm5': m5, 'm5_temporal': temporal(tiny_model_config.m5_context),
        'm15': m15, 'm15_temporal': temporal(tiny_model_config.m15_context),
        'h1': h1, 'h1_temporal': temporal(tiny_model_config.h1_context),
        'h4': h4, 'h4_temporal': temporal(tiny_model_config.h4_context),
        'parent_enc': torch.tensor(rng.normal(size=(2, 4)), dtype=torch.float32),
        'children_enc': torch.tensor(rng.normal(size=(2, 3, 4)), dtype=torch.float32),
        'parent_prices': torch.tensor([[2000.0, 2002.0, 1998.0, 2001.0], [2001.0, 2003.0, 1999.0, 2002.0]]),
        'child_prices': torch.tensor(
            [[[2000.0, 2001.0, 1999.5, 2000.5], [2000.5, 2002.0, 2000.0, 2001.0], [2001.0, 2001.5, 1998.5, 2001.0]],
             [[2001.0, 2002.0, 2000.5, 2001.5], [2001.5, 2003.0, 2001.0, 2002.0], [2002.0, 2002.5, 1999.5, 2002.0]]]
        ),
        'm15_prev_close': torch.tensor([1999.0, 2000.0]),
        'children_prev_close': torch.tensor([[2000.0, 2000.5, 2001.0], [2001.0, 2001.5, 2002.0]]),
        'target_ts': torch.tensor([1700000000, 1700000900], dtype=torch.int64),
    }
