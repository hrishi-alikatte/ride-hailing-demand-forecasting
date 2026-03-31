"""Adjacency normalization helpers."""

from __future__ import annotations

import numpy as np


def normalize_adjacency_with_self_loop(adjacency: np.ndarray) -> np.ndarray:
    """Return `I + D^{-1/2} A D^{-1/2}` for a weighted adjacency matrix."""
    matrix = np.asarray(adjacency, dtype=np.float64)
    degrees = matrix.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    nonzero_mask = degrees > 0
    inv_sqrt[nonzero_mask] = np.power(degrees[nonzero_mask], -0.5)
    normalized = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    return np.eye(matrix.shape[0], dtype=np.float64) + normalized
