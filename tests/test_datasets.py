"""Chronological splitting, hierarchy coherence and leakage checks."""

import numpy as np

from olm.data.calendar import TemporalFeatures
from olm.data.datasets import build_datasets, leakage_check


def test_splits_are_chronological(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    check = leakage_check(bundle)
    assert check['ok'], check
    assert len(bundle.train) > 0 and len(bundle.test) > 0


def test_each_sample_reconstructs_exact_parent(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    for index in range(min(20, len(bundle.train))):
        item = bundle.train[index]
        children = item['child_prices'].numpy()
        parent = item['parent_prices'].numpy()
        assert children[0, 0] == parent[0]
        assert children[:, 1].max() == parent[1]
        assert children[:, 2].min() == parent[2]
        assert children[-1, 3] == parent[3]



def test_each_child_window_is_contiguous_and_parent_aligned(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    for split in (bundle.train, bundle.val, bundle.test):
        m5 = split.arrays['5min']
        m15 = split.arrays['15min']
        target = split.targets['target_ts'].astype('datetime64[s]').astype('datetime64[ns]')
        child_ts = np.stack([m5.ts[split.indices + step] for step in (1, 2, 3)], axis=1)
        expected = np.stack(
            [target + np.timedelta64(step * 5, 'm') for step in (0, 1, 2)], axis=1
        )
        assert np.array_equal(child_ts, expected)
        assert np.array_equal(m15.ts[split.targets['p15']], target)

def test_context_shapes_follow_config(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    item = bundle.train[0]
    assert item['m5'].shape == (tiny_config.model.m5_context, 4)
    assert item['m15'].shape == (tiny_config.model.m15_context, 4)
    assert item['h1'].shape == (tiny_config.model.h1_context, 4)
    assert item['h4'].shape == (tiny_config.model.h4_context, 4)


def test_higher_timeframe_context_bars_are_complete(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    for split in (bundle.train, bundle.val, bundle.test):
        target = split.targets['target_ts'].astype('datetime64[s]').astype('datetime64[ns]')
        for name, minutes, key in (('1h', 60, 'j1'), ('4h', 240, 'j4')):
            closes = split.arrays[name].ts[split.targets[key]] + np.timedelta64(minutes, 'm')
            assert (closes <= target).all(), f'{name} context leaks future candles'


def test_normalizer_fitted_only_on_train(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    assert np.isfinite(bundle.normalizer.mean).all()
    before = bundle.normalizer.mean.copy()
    _ = bundle.val[0]
    assert np.array_equal(before, bundle.normalizer.mean)
