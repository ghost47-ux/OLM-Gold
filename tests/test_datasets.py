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


def test_context_shapes_follow_config(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    item = bundle.train[0]
    assert item['m5'].shape == (tiny_config.model.m5_context, 4)
    assert item['m15'].shape == (tiny_config.model.m15_context, 4)
    assert item['h1'].shape == (tiny_config.model.h1_context, 4)
    assert item['h4'].shape == (tiny_config.model.h4_context, 4)


def test_normalizer_fitted_only_on_train(tiny_config, synthetic_frames):
    bundle = build_datasets(tiny_config, synthetic_frames, TemporalFeatures(tiny_config.temporal))
    assert np.isfinite(bundle.normalizer.mean).all()
    before = bundle.normalizer.mean.copy()
    _ = bundle.val[0]
    assert np.array_equal(before, bundle.normalizer.mean)
