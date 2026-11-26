"""Naive and simple autoregressive baselines used as reference points."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from olm.models.olm import OLMModel, build_model


def persistence_forecast(last_close: np.ndarray) -> np.ndarray:
    """Flat candle at the previous close, shape (N, 4)."""
    last_close = np.asarray(last_close, dtype=np.float64).reshape(-1, 1)
    return np.repeat(last_close, 4, axis=1)


def fit_ar1_pairs(x_prev: np.ndarray, x_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares AR(1) from explicit, aligned predecessor/successor pairs."""
    x_prev = np.asarray(x_prev, dtype=np.float64)
    x_next = np.asarray(x_next, dtype=np.float64)
    if x_prev.ndim != 2 or x_next.shape != x_prev.shape or len(x_prev) < 2:
        raise ValueError('AR1 needs matching (N, D) arrays with N >= 2')
    design = np.concatenate([np.ones((len(x_prev), 1)), x_prev], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, x_next, rcond=None)
    return coefficients[0], coefficients[1:]


def fit_ar1(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares AR(1) on encoded coordinates: x_t = c + A x_{t-1}."""
    return fit_ar1_pairs(encoded[:-1], encoded[1:])


def fit_ar1_bundle(bundle) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit parent and child AR(1) only on contiguous training observations."""
    import pandas as pd

    train_end = np.datetime64(
        pd.Timestamp(bundle.boundaries['train_end']).tz_convert('UTC').tz_localize(None)
    )
    fitted = {}
    for name, minutes in (('15min', 15), ('5min', 5)):
        arrays = bundle.train.arrays[name]
        ts = arrays.ts
        if len(ts) < 2:
            continue
        contiguous = (ts[1:] - ts[:-1]) == np.timedelta64(minutes, 'm')
        mask = contiguous & (ts[1:] < train_end)
        if int(mask.sum()) < 2:
            continue
        fitted[name] = fit_ar1_pairs(arrays.coords[:-1][mask], arrays.coords[1:][mask])
    return fitted


def predict_ar1(model_params: tuple[np.ndarray, np.ndarray], encoded: np.ndarray) -> np.ndarray:
    intercept, transition = model_params
    return intercept + encoded @ transition


def build_m5_only_model(config, temporal_dim: int) -> OLMModel:
    """Build the genuine M5-only ablation: no M15/H1/H4 context, same heads.

    This must be trained separately with `python -m olm train --no-context`;
    wrapping an already-trained context model would not be a valid ablation.
    """
    return build_model(replace(config, use_context=False), temporal_dim)
