"""Naive and simple autoregressive baselines used as reference points."""

from __future__ import annotations

import numpy as np
import torch

from olm.models.decoding import aggregate_children_torch


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


class M5OnlyBaseline(torch.nn.Module):
    """Ablation: the same backbone without M15/H1/H4 context or hierarchy."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> dict:
        stripped = dict(batch)
        return self.model(stripped)
