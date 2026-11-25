"""FastAPI endpoint tests with a tiny local checkpoint (no network)."""

import os

import numpy as np
import torch
from fastapi.testclient import TestClient

from olm.data.aggregation import aggregate_blocks
from olm.data.calendar import TemporalFeatures
from olm.data.datasets import build_datasets
from olm.models.olm import build_model
from olm.training.checkpointing import save_checkpoint
from olm.training.scheduler import build_scheduler


def _checkpoint(tiny_config, synthetic_frames, tmp_path):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    model = build_model(tiny_config.model, bundle.meta['temporal_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=10, warmup_steps=1)
    path = tmp_path / 'api_model.pt'
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=0, global_step=0,
        config=tiny_config, normalizer=bundle.normalizer, dataset_meta=bundle.meta, best_val=1.0,
    )
    return path


def test_health_model_and_forecast(tiny_config, synthetic_frames, tmp_path, monkeypatch):
    checkpoint = _checkpoint(tiny_config, synthetic_frames, tmp_path)
    monkeypatch.setenv('OLM_CHECKPOINT', str(checkpoint))
    import olm.inference.api as api

    api.state.forecaster = None
    api.state.checkpoint = str(checkpoint)
    api.state.log = api.PredictionLog(tmp_path / 'predictions.jsonl')
    client = TestClient(api.app)
    assert client.get('/health').json()['status'] == 'ok'
    info = client.get('/v1/model').json()
    assert info['parameters'] > 1000
    # Leave the final M15 block unseen so the forecast has a real actual to score.
    history = synthetic_frames['5min'].tail(1500).iloc[:-3].reset_index(drop=True)
    candles = [
        {'timestamp': str(row.timestamp), 'open': float(row.open), 'high': float(row.high),
         'low': float(row.low), 'close': float(row.close)}
        for row in history.itertuples()
    ]
    response = client.post('/v1/forecast', json={'m5': candles, 'scenarios': 2, 'seed': 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body['scenarios']) == 2
    assert body['forecast_id']
    parent = body['scenarios'][0]['parent']
    assert parent['high'] >= parent['low']
    # Record the next M15 parent as actual, then evaluate.
    last = history['timestamp'].max()
    next_rows = synthetic_frames['5min'][synthetic_frames['5min']['timestamp'] > last].head(3)
    assert len(next_rows) == 3
    payload = {
        'forecast_id': body['forecast_id'],
        'candles': [
            {'timestamp': str(row.timestamp), 'open': float(row.open), 'high': float(row.high),
             'low': float(row.low), 'close': float(row.close)}
            for row in next_rows.itertuples()
        ],
    }
    result = client.post('/v1/actual', json=payload)
    assert result.status_code == 200, result.text
    assert 'mae' in result.json()


def test_forecast_rejects_short_history(tiny_config, synthetic_frames, tmp_path, monkeypatch):
    checkpoint = _checkpoint(tiny_config, synthetic_frames, tmp_path)
    monkeypatch.setenv('OLM_CHECKPOINT', str(checkpoint))
    import olm.inference.api as api

    api.state.forecaster = None
    api.state.checkpoint = str(checkpoint)
    client = TestClient(api.app)
    response = client.post('/v1/forecast', json={'m5': [], 'scenarios': 1})
    assert response.status_code == 422
