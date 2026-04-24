"""Forecasting metrics used by the paper and by baseline experiments."""

from __future__ import annotations

import numpy as np


def _to_numpy(array: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(array, dtype=float)


def _get_mask(y_true: np.ndarray, null_val: float | None) -> np.ndarray | None:
    """Mask handles zero-inflation typically present in spatial demand grids."""
    if null_val is None:
        return None
    return y_true > null_val


def mae(
    y_true: np.ndarray | list[float], 
    y_pred: np.ndarray | list[float],
    *,
    null_val: float | None = 0.0,
) -> float:
    """Compute mean absolute error with optional zero-masking."""
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    mask = _get_mask(truth, null_val)
    if mask is not None:
        if not np.any(mask):
            return 0.0
        truth = truth[mask]
        pred = pred[mask]
    return float(np.mean(np.abs(truth - pred)))


def rmse(
    y_true: np.ndarray | list[float], 
    y_pred: np.ndarray | list[float],
    *,
    null_val: float | None = 0.0,
) -> float:
    """Compute root mean squared error with optional zero-masking."""
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    mask = _get_mask(truth, null_val)
    if mask is not None:
        if not np.any(mask):
            return 0.0
        truth = truth[mask]
        pred = pred[mask]
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def mape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    *,
    epsilon: float = 1.0,
    null_val: float | None = 0.0,
) -> float:
    """Compute mean absolute percentage error with zero-masking boundary.

    Returns MAPE as a percentage (e.g. 31.82 means 31.82%).
    Uses epsilon=1.0 by default to prevent near-zero pickup counts from
    inflating the metric, following the convention in DCRNN, STGCN, GWNet,
    and other major traffic forecasting papers.
    """
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    mask = _get_mask(truth, null_val)
    if mask is not None:
        if not np.any(mask):
            return 0.0
        truth = truth[mask]
        pred = pred[mask]
    denominator = np.clip(np.abs(truth), epsilon, None)
    return float(np.mean(np.abs(truth - pred) / denominator) * 100)
