"""Helpers to reduce accidental run-to-run variation."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
