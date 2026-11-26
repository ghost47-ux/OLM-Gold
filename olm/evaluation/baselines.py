"""Naive and simple autoregressive baselines used as reference points."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from olm.models.olm import OLMModel, build_model


def persistence_forecast(last_close: np.ndarray) -> np.ndarray:
    """Flat candle at the previous close, shape (N, 4)."""
    last_close = np.asarray(last_close, dtype=np.float64).reshape(-1, 1)
    return np.repeat(last_close, 4, axis=1)


def fit_ar1(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares AR(1) on encoded coordinates: x_t = c + A x_{t-1}."""
    x_prev = encoded[:-1]
    x_next = encoded[1:]
    design = np.concatenate([np.ones((len(x_prev), 1)), x_prev], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, x_next, rcond=None)
    intercept = coefficients[0]
    transition = coefficients[1:]
    return intercept, transition


def predict_ar1(model_params: tuple[np.ndarray, np.ndarray], encoded: np.ndarray) -> np.ndarray:
    intercept, transition = model_params
    return intercept + encoded @ transition


def build_m5_only_model(config, temporal_dim: int) -> OLMModel:
    """Build the genuine M5-only ablation: no M15/H1/H4 context, same heads.

    This must be trained separately with `python -m olm train --no-context`;
    wrapping an already-trained context model would not be a valid ablation.
    """
    return build_model(replace(config, use_context=False), temporal_dim)
