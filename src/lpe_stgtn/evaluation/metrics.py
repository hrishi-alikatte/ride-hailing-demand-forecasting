"""Forecasting metrics used by the paper and by baseline experiments."""

from __future__ import annotations

import numpy as np


def _to_numpy(array: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(array, dtype=float)


def mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Compute mean absolute error."""
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Compute root mean squared error."""
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def mape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    *,
    epsilon: float = 1e-6,
) -> float:
    """Compute mean absolute percentage error with explicit zero protection."""
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    denominator = np.clip(np.abs(truth), epsilon, None)
    return float(np.mean(np.abs(truth - pred) / denominator))
