"""Sequential block evaluation and append-only prediction log."""

from olm.data.calendar import TemporalFeatures
from olm.data.datasets import build_datasets
from olm.evaluation.blocks import block_bounds, evaluate_sequential_blocks
from olm.models.olm import build_model
from olm.monitoring.prediction_log import PredictionLog


def test_block_bounds_cover_every_sample():
    assert block_bounds(10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert block_bounds(7, 7) == [(i, i + 1) for i in range(7)]


def test_sequential_blocks_report_chronological_metrics(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    model = build_model(tiny_config.model, bundle.meta['temporal_dim'])
    report = evaluate_sequential_blocks(model, bundle.test, tiny_config, 'cpu', bundle.normalizer, folds=3, n_samples=3)
    assert len(report['blocks']) == 3
    assert sum(block['n'] for block in report['blocks']) == len(bundle.test)
    assert all(block['start'] < block['end'] for block in report['blocks'])
    assert 'parent_mae' in report['mean']


def test_prediction_log_round_trip(tmp_path):
    log = PredictionLog(tmp_path / 'predictions.jsonl')
    payload = {
        'scenarios': [{'parent': {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5}, 'children': []}],
        'diagnostics': {'model_version': 'test'},
    }
    record = log.record_forecast(payload, 'test', '2024-01-01T00:00:00+00:00')
    assert len(record['forecast_id']) == 24
    stored = log.read_all()
    assert stored[0]['forecast_id'] == record['forecast_id']
    actual = [
        {'open': 100.0, 'high': 100.6, 'low': 99.8, 'close': 100.2},
        {'open': 100.2, 'high': 101.2, 'low': 100.0, 'close': 101.0},
        {'open': 101.0, 'high': 101.1, 'low': 100.4, 'close': 100.7},
    ]
    evaluation = log.evaluate_forecast(record['forecast_id'], actual)
    assert evaluation['mae'] >= 0
    assert log.read_all()[-1]['type'] == 'evaluation'
