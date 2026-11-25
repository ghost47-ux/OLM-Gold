"""Feature and prediction drift monitoring with explicit alert thresholds."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float('nan')
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_share = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_share = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.sort(np.asarray(expected, dtype=np.float64).ravel())
    actual = np.sort(np.asarray(actual, dtype=np.float64).ravel())
    if not len(expected) or not len(actual):
        return float('nan')
    grid = np.unique(np.concatenate([expected, actual]))
    cdf_e = np.searchsorted(expected, grid, side='right') / len(expected)
    cdf_a = np.searchsorted(actual, grid, side='right') / len(actual)
    return float(np.max(np.abs(cdf_e - cdf_a)))


def drift_report(reference: dict[str, np.ndarray], current: dict[str, np.ndarray], psi_alert: float = 0.2) -> dict:
    features = {}
    alerts = []
    for name in sorted(set(reference) & set(current)):
        psi = population_stability_index(reference[name], current[name])
        ks = ks_statistic(reference[name], current[name])
        features[name] = {'psi': psi, 'ks': ks}
        if np.isfinite(psi) and psi >= psi_alert:
            alerts.append(f'{name}: PSI {psi:.3f} >= {psi_alert}')
    return {
        'features': features,
        'alerts': alerts,
        'status': 'alert' if alerts else 'ok',
        'psi_alert_threshold': psi_alert,
        'note': 'Drift indicates changing inputs, not proven model degradation. Validate with out-of-sample metrics.',
    }
