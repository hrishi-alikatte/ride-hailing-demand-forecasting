"""Simple LSTM baseline for multistep taxi-demand forecasting."""

from __future__ import annotations

import torch
from torch import nn


class LSTMBaseline(nn.Module):
    """Encode the history window with an LSTM and predict the full horizon at once."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        forecast_steps: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.forecast_steps = forecast_steps
        self.input_dim = input_dim
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_dim, forecast_steps * input_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict a full multistep horizon from history of shape `(B, T, N)`."""
        _, (hidden_state, _) = self.encoder(history)
        encoded = hidden_state[-1]
        forecast = self.head(encoded)
        return forecast.view(history.shape[0], self.forecast_steps, self.input_dim)
