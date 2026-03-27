"""Persistence baseline for multistep taxi-demand forecasting."""

from __future__ import annotations

import torch


class PersistenceBaseline:
    """Repeat the most recent observed timestep across the full forecast horizon."""

    def __init__(self, forecast_steps: int) -> None:
        self.forecast_steps = forecast_steps

    def predict(self, history: torch.Tensor) -> torch.Tensor:
        """Generate a persistence forecast from history of shape `(B, T, N)`."""
        last_step = history[:, -1:, :]
        return last_step.repeat(1, self.forecast_steps, 1)
