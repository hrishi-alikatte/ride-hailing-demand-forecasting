"""Embedding layer for mapping demand and temporal metadata into node features."""

from __future__ import annotations

import torch
from torch import nn


class SpatioTemporalEmbedding(nn.Module):
    """Embeds demand, time-of-day, and day-of-week into a dense hidden state.

    The paper concatenates historical demand with time-of-day and day-of-week
    features before projecting them into a higher-dimensional representation.

    Input shapes:
    - demand: `(B, T, N, 1)`
    - time_of_day: `(B, T)` (values: 0-95)
    - day_of_week: `(B, T)` (values: 0-6)

    Output shape:
    - `(B, T, N, hidden_dim)`
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        time_of_day_dim: int = 12,
        day_of_week_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 15-minute intervals -> 24 * 4 = 96
        self.tod_emb = nn.Embedding(96, time_of_day_dim)
        # 7 days in a week
        self.dow_emb = nn.Embedding(7, day_of_week_dim)
        
        self.linear = nn.Linear(1 + time_of_day_dim + day_of_week_dim, hidden_dim)

    def forward(
        self,
        demand: torch.Tensor,
        time_of_day: torch.Tensor,
        day_of_week: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_steps, num_zones, _ = demand.shape

        # `(B, T, Emb)`
        tod_features = self.tod_emb(time_of_day)
        dow_features = self.dow_emb(day_of_week)

        # Broadcast temporal features across all zones: `(B, T, N, Emb)`
        tod_features = tod_features.unsqueeze(2).expand(batch_size, num_steps, num_zones, -1)
        dow_features = dow_features.unsqueeze(2).expand(batch_size, num_steps, num_zones, -1)

        # Concatenate: `(B, T, N, 1 + tod_dim + dow_dim)`
        combined = torch.cat([demand, tod_features, dow_features], dim=-1)

        # Project: `(B, T, N, hidden_dim)`
        return self.linear(combined)
