"""Scenario generation, coherence projection and history derivation."""

import numpy as np
import pandas as pd
import torch

from olm.data.calendar import TemporalFeatures
from olm.data.datasets import build_datasets
from olm.inference.forecast import Forecaster, project_children_to_parent
from olm.models.olm import build_model
from olm.training.checkpointing import save_checkpoint
from olm.training.scheduler import build_scheduler


def test_projection_produces_exact_parent():
    parent = np.array([[2000.0, 2005.0, 1995.0, 2002.0]])
    children = np.tile(np.array([[1998.0, 2001.0, 1997.0, 1999.0]]), (1, 3, 1))
    projected = project_children_to_parent(children, parent)
    assert projected[0, 0, 0] == 2000.0
    assert projected[0, -1, 3] == 2002.0
    assert projected[0, :, 1].max() == 2005.0
    assert projected[0, :, 2].min() == 1995.0
    assert (projected[..., 1] >= projected[..., 0]).all()
    assert (projected[..., 1] >= projected[..., 3]).all()
    assert (projected[..., 2] <= projected[..., 0]).all()


def test_forecast_scenarios_are_coherent(tiny_config, synthetic_frames, tmp_path):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    model = build_model(tiny_config.model, bundle.meta['temporal_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=10, warmup_steps=1)
    checkpoint = tmp_path / 'model.pt'
    save_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, epoch=0, global_step=0,
        config=tiny_config, normalizer=bundle.normalizer, dataset_meta=bundle.meta, best_val=1.0,
    )
    forecaster = Forecaster(checkpoint, config=tiny_config)
    result = forecaster.forecast(synthetic_frames, scenarios=5, seed=3)
    payload = result.to_dict()
    assert len(payload['scenarios']) == 5
    for scenario in payload['scenarios']:
        parent = scenario['parent']
        children = scenario['children']
        assert len(children) == 3
        assert children[0]['open'] == parent['open']
        assert children[-1]['close'] == parent['close']
        assert max(child['high'] for child in children) == parent['high']
        assert min(child['low'] for child in children) == parent['low']
        assert scenario['coherent'] is True
    assert payload['diagnostics']['scenarios'] == 5

    arrays = forecaster._arrays(synthetic_frames)
    i, p15, j1, j4 = forecaster._latest_context(arrays)
    target_start = arrays['5min'].ts[i] + np.timedelta64(5, 'm')
    # p15 may equal len(m15): the next parent has not formed yet, which is a valid forecast origin.
    assert p15 == len(arrays['15min'].ts) or arrays['15min'].ts[p15] == target_start
    assert arrays['1h'].ts[j1] + np.timedelta64(60, 'm') <= target_start
    assert arrays['4h'].ts[j4] + np.timedelta64(240, 'm') <= target_start


def test_derived_context_is_aligned_and_exact(synthetic_frames):
    from olm.inference.forecast import derive_context_frames

    m5 = synthetic_frames['5min']
    derived = derive_context_frames(m5)
    assert set(derived) == {'5min', '15min', '1h', '4h'}
    assert derived['5min']['timestamp'].is_monotonic_increasing
    assert len(derived['5min']) == len(m5)
    # 4h aggregation starts on the first H4 boundary, so lengths are exact multiples
    assert len(derived['4h']) * 48 <= len(derived['5min'])
