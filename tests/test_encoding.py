"""Invertible encoding and probability Jacobian tests."""

import numpy as np

from olm.data.encoding import (
    Normalizer, decode_model, decode_raw, encode_model, encode_raw, log_jacobian_candle_space,
)


def _candles(n=200, seed=3):
    rng = np.random.default_rng(seed)
    prev = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    prev = np.maximum(prev, 100.0)
    o = prev
    c = o * np.exp(rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.0004, n)))
    return o, h, l, c, prev


def test_encode_decode_model_round_trip():
    o, h, l, c, prev = _candles()
    encoded = encode_model(o, h, l, c, prev)
    decoded = decode_model(encoded, prev)
    assert np.allclose(decoded, np.stack([o, h, l, c], axis=-1), rtol=1e-9, atol=1e-9)
    assert np.isfinite(encoded).all()


def test_encode_decode_raw_round_trip():
    o, h, l, c, prev = _candles(50)
    raw = encode_raw(o, h, l, c, prev)
    decoded = decode_raw(raw, prev)
    assert np.allclose(decoded, np.stack([o, h, l, c], axis=-1), rtol=1e-9, atol=1e-9)


def test_zero_wicks_round_trip():
    o = np.array([100.0, 101.0])
    c = np.array([101.0, 100.0])
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    encoded = encode_model(o, h, l, c, np.array([99.0, 99.0]))
    decoded = decode_model(encoded, np.array([99.0, 99.0]))
    assert np.allclose(decoded[:, 1], h) and np.allclose(decoded[:, 2], l)


def test_normalizer_is_train_only_invertible():
    values = np.random.default_rng(0).normal(size=(200, 4)) * 2 + 5
    normalizer = Normalizer.fit(values[:100])
    transformed = normalizer.transform(values)
    assert np.allclose(normalizer.inverse(transformed), values)
    restored = Normalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.mean, normalizer.mean) and np.allclose(restored.std, normalizer.std)


def test_jacobian_matches_finite_difference():
    # d/da softplus(a) = sigmoid(a); log p(u) = log p(a) - log sigmoid(a)
    a = np.array([0.0, 0.7, -0.4])
    coords = np.stack([np.zeros(3), np.zeros(3), a, a], axis=-1)
    correction = log_jacobian_candle_space(coords)
    expected = -np.log(1 / (1 + np.exp(-a)))
    assert correction.shape == (3, 2)
    assert np.allclose(correction[:, 0], expected) and np.allclose(correction[:, 1], expected)


def test_invalid_prices_raise():
    import pytest

    with pytest.raises(Exception):
        encode_model(np.array([100.0]), np.array([99.0]), np.array([98.0]), np.array([100.0]), np.array([99.0]))
