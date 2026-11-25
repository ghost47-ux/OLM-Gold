"""Session/temporal features including DST transitions."""

import numpy as np
import pandas as pd

from olm.config import TemporalConfig
from olm.data.calendar import TemporalFeatures


def test_feature_shape_and_finiteness():
    features = TemporalFeatures(TemporalConfig())
    index = pd.date_range('2024-03-10 00:00', periods=48, freq='h', tz='UTC')
    values = features.compute(index)
    assert values.shape == (48, features.dim)
    assert np.isfinite(values).all()
    assert features.feature_names[:4] == ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']


def test_dst_transition_keeps_valid_sessions():
    features = TemporalFeatures(TemporalConfig())
    index = pd.date_range('2024-03-31 00:00', periods=30, freq='h', tz='UTC')
    values = features.compute(index)
    assert np.isfinite(values).all()
    names = features.feature_names
    for name in ('session_london', 'session_new_york'):
        column = values[:, names.index(name)]
        assert set(np.unique(column)).issubset({0.0, 1.0})


def test_event_features_are_point_in_time_visible():
    config = TemporalConfig(include_events=True)
    features = TemporalFeatures(config)
    assert features.dim > TemporalFeatures(TemporalConfig()).dim
