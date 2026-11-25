"""Invertible OHLC encoding (eq. 4 of the OLM specification).

Raw coordinates for a candle with previous close PC:

    g = ln(O / PC)          opening displacement
    b = ln(C / O)           body movement
    u = ln(H / max(O, C))   upper wick      (>= 0)
    d = ln(min(O, C) / L)   lower wick      (>= 0)

u and d are non-negative, so the model never predicts them directly. It predicts
unconstrained a_u, a_d with u = softplus(a_u) - EPS_WICK. EPS_WICK is a 1e-8
numerical floor that keeps the inverse softplus finite for zero-wick candles.
Densities in candle space require the Jacobian correction

    log p(u) = log p(a_u) - log sigmoid(a_u)

and likewise for d (see probability_head.mixture_log_prob_candle_space).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS_WICK = 1e-8
COORD_NAMES = ('g', 'b', 'a_u', 'a_d')
N_COORDS = 4


class CandleError(ValueError):
    """Raised when a candle cannot be encoded safely."""


def inv_softplus_np(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0):
        raise CandleError('inverse softplus requires strictly positive input')
    out = np.empty_like(y)
    small = y < 1e-6
    out[small] = np.log(y[small])
    out[~small] = np.log(np.expm1(y[~small]))
    return out


def softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _as_float(*arrays: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(a, dtype=np.float64) for a in arrays]


def validate_candle_arrays(o, h, l, c, prev_close) -> None:
    o, h, l, c, pc = _as_float(o, h, l, c, prev_close)
    if not (o.shape == h.shape == l.shape == c.shape == pc.shape):
        raise CandleError('OHLC arrays must share a shape')
    for name, arr in (('open', o), ('high', h), ('low', l), ('close', c), ('prev_close', pc)):
        if not np.all(np.isfinite(arr)):
            raise CandleError(f'{name} contains NaN or infinity')
        if np.any(arr <= 0):
            raise CandleError(f'{name} contains non-positive prices')
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    if np.any(h < top - 1e-12 * top):
        raise CandleError('high is below max(open, close)')
    if np.any(l > bottom + 1e-12 * bottom):
        raise CandleError('low is above min(open, close)')


def encode_raw(o, h, l, c, prev_close) -> np.ndarray:
    """Return raw (g, b, u, d) coordinates. Purely invertible, no smoothing."""
    validate_candle_arrays(o, h, l, c, prev_close)
    o, h, l, c, pc = _as_float(o, h, l, c, prev_close)
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.log(o / pc)
        b = np.log(c / o)
        u = np.log(h / top)
        d = np.log(bottom / l)
    raw = np.stack([g, b, u, d], axis=-1)
    if not np.all(np.isfinite(raw)):
        raise CandleError('encoding produced non-finite values')
    return raw


def encode_model(o, h, l, c, prev_close) -> np.ndarray:
    """Return model-space (g, b, a_u, a_d) targets."""
    raw = encode_raw(o, h, l, c, prev_close)
    out = raw.copy()
    out[..., 2] = inv_softplus_np(raw[..., 2] + EPS_WICK)
    out[..., 3] = inv_softplus_np(raw[..., 3] + EPS_WICK)
    return out


def decode_model(x: np.ndarray, prev_close) -> np.ndarray:
    """Decode model-space coordinates to (O, H, L, C) prices."""
    x = np.asarray(x, dtype=np.float64)
    pc = np.asarray(prev_close, dtype=np.float64)
    g, b = x[..., 0], x[..., 1]
    u = np.maximum(softplus_np(x[..., 2]) - EPS_WICK, 0.0)
    d = np.maximum(softplus_np(x[..., 3]) - EPS_WICK, 0.0)
    o = pc * np.exp(g)
    c = o * np.exp(b)
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    h = top * np.exp(u)
    l = bottom * np.exp(-d)
    return np.stack([o, h, l, c], axis=-1)


def decode_raw(raw: np.ndarray, prev_close) -> np.ndarray:
    """Decode raw coordinates (g, b, u, d) to (O, H, L, C) prices."""
    raw = np.asarray(raw, dtype=np.float64)
    pc = np.asarray(prev_close, dtype=np.float64)
    o = pc * np.exp(raw[..., 0])
    c = o * np.exp(raw[..., 1])
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    h = top * np.exp(raw[..., 2])
    l = bottom * np.exp(-raw[..., 3])
    return np.stack([o, h, l, c], axis=-1)


def encode_frame(frame: pd.DataFrame, timestamp_col: str = 'timestamp', model_space: bool = True) -> pd.DataFrame:
    """Encode a candle frame using its own previous close; the first row is dropped."""
    missing = {'timestamp', 'open', 'high', 'low', 'close'} - set(frame.columns)
    if missing:
        raise ValueError(f'missing OHLC columns: {sorted(missing)}')
    ordered = frame.sort_values(timestamp_col).reset_index(drop=True)
    prev_close = ordered['close'].shift(1).to_numpy()
    valid = ~np.isnan(prev_close)
    encoder = encode_model if model_space else encode_raw
    encoded = encoder(
        ordered['open'].to_numpy()[valid],
        ordered['high'].to_numpy()[valid],
        ordered['low'].to_numpy()[valid],
        ordered['close'].to_numpy()[valid],
        prev_close[valid],
    )
    result = pd.DataFrame(encoded, columns=list(COORD_NAMES))
    result.insert(0, timestamp_col, ordered[timestamp_col].to_numpy()[valid])
    result['prev_close'] = prev_close[valid]
    return result


@dataclass
class Normalizer:
    """Per-coordinate standardization fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if self.mean.shape != (N_COORDS,) or self.std.shape != (N_COORDS,):
            raise ValueError('normalizer must have four coordinates')
        if np.any(self.std <= 0) or not np.all(np.isfinite(self.std)):
            raise ValueError('normalizer std must be finite and positive')

    @classmethod
    def fit(cls, values: np.ndarray) -> 'Normalizer':
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != N_COORDS:
            raise ValueError('expected an (N, 4) array')
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> 'Normalizer':
        return cls(mean=np.asarray(data['mean'], dtype=np.float64), std=np.asarray(data['std'], dtype=np.float64))


def log_jacobian_candle_space(model_coords: np.ndarray) -> np.ndarray:
    """log|du/da| for the two wick coordinates, shape (..., 2)."""
    a = np.asarray(model_coords, dtype=np.float64)[..., 2:4]
    return -np.log(np.maximum(1.0 / (1.0 + np.exp(-a)), 1e-30))
