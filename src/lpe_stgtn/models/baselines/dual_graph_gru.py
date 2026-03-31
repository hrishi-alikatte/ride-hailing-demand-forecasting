"""Stage-4 graph-based baseline built from the paper's global-module ideas."""

from __future__ import annotations

import torch
from torch import nn

from lpe_stgtn.models.components.attention import SemanticFusionAttention
from lpe_stgtn.models.components.gcn import GraphConvolution


class DualGraphGRUBaseline(nn.Module):
    """Distance-GCN + OD-GCN + semantic attention + GRU forecasting baseline.

    This is intentionally simpler than the full paper model. It reuses the
    paper's global-module ingredients with a lightweight temporal head:
    - distance graph convolution
    - OD-flow graph convolution
    - multi-head semantic fusion attention
    - per-zone GRU encoder over the historical sequence
    - residual forecasting on top of the persistence baseline

    Input:
    - history: `(B, T, N)` normalized demand

    Output:
    - forecast: `(B, P, N)` normalized demand
    """

    def __init__(
        self,
        *,
        num_zones: int,
        forecast_steps: int,
        distance_adjacency: torch.Tensor,
        od_flow_adjacency: torch.Tensor,
        graph_hidden_dim: int,
        temporal_hidden_dim: int,
        attention_heads: int,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if gru_layers > 1 else 0.0
        self.num_zones = num_zones
        self.forecast_steps = forecast_steps
        self.graph_hidden_dim = graph_hidden_dim

        self.register_buffer("distance_adjacency", distance_adjacency.to(torch.float32))
        self.register_buffer("od_flow_adjacency", od_flow_adjacency.to(torch.float32))

        self.distance_gcn = GraphConvolution(1, graph_hidden_dim)
        self.od_flow_gcn = GraphConvolution(1, graph_hidden_dim)
        self.semantic_fusion = SemanticFusionAttention(
            graph_hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.temporal_encoder = nn.GRU(
            input_size=graph_hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.readout = nn.Linear(temporal_hidden_dim, forecast_steps)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, num_zones = history.shape
        if num_zones != self.num_zones:
            raise ValueError(
                f"Expected {self.num_zones} zones in the input history, got {num_zones}."
            )

        node_inputs = history.reshape(batch_size * history_steps, num_zones, 1)
        distance_features = torch.relu(
            self.distance_gcn(node_inputs, self.distance_adjacency)
        )
        od_features = torch.relu(self.od_flow_gcn(node_inputs, self.od_flow_adjacency))
        fused = self.semantic_fusion(distance_features, od_features)

        # Reorder to per-zone temporal sequences: `(B*N, T, D)`.
        fused = fused.view(batch_size, history_steps, num_zones, self.graph_hidden_dim)
        fused = fused.permute(0, 2, 1, 3).reshape(
            batch_size * num_zones,
            history_steps,
            self.graph_hidden_dim,
        )
        _, hidden_state = self.temporal_encoder(fused)
        encoded = hidden_state[-1]
        forecast_delta = self.readout(encoded)
        forecast_delta = forecast_delta.view(batch_size, num_zones, self.forecast_steps).permute(
            0, 2, 1
        )
        persistence = history[:, -1:, :].repeat(1, self.forecast_steps, 1)
        return persistence + forecast_delta
