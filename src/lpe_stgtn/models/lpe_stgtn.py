"""Full LPE-STGTN architecture."""

from __future__ import annotations

import torch
from torch import nn

from lpe_stgtn.models.components.aft_local import AttentionFreeTransformerLocal
from lpe_stgtn.models.components.attention import GlobalMultiHeadAttention
from lpe_stgtn.models.components.dynamic_graph import DynamicGraphGenerator
from lpe_stgtn.models.components.embedding import SpatioTemporalEmbedding
from lpe_stgtn.models.components.gcn import GraphConvolution


class LPE_STGTN(nn.Module):
    """Local Perception-Enhanced Spatial-Temporal Evolving Graph Transformer Network.

    Combines the local perception stream (dynamic graphs, AFT-local) with the
    global stream (pre-computed distance/OD graphs, multi-head attention),
    and aggregates across the historical window to compute the forecast.
    """

    def __init__(
        self,
        *,
        num_zones: int,
        forecast_steps: int,
        distance_adjacency: torch.Tensor,
        od_flow_adjacency: torch.Tensor,
        hidden_dim: int = 64,
        attention_heads: int = 4,
        aft_window_size: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.0, #change to 0.2
    ) -> None:
        super().__init__()
        self.num_zones = num_zones
        self.forecast_steps = forecast_steps
        self.hidden_dim = hidden_dim

        self.register_buffer("distance_adjacency", distance_adjacency.to(torch.float32))
        self.register_buffer("od_flow_adjacency", od_flow_adjacency.to(torch.float32))

        # 1. Embedding
        self.embedding = SpatioTemporalEmbedding(hidden_dim)

        # 2. Local Module
        self.dynamic_graph_generator = DynamicGraphGenerator(hidden_dim, embed_dim=32, num_nodes=num_zones)
        self.aft_local = AttentionFreeTransformerLocal(hidden_dim, window_size=aft_window_size, dropout=dropout)
        
        # Local GCN uses dynamic graph. We just map node features directly using a linear layer
        # before the dynamic spatial correlation in forward.
        self.local_gcn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.local_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 3. Global Module
        self.distance_gcn = GraphConvolution(hidden_dim, hidden_dim)
        self.od_flow_gcn = GraphConvolution(hidden_dim, hidden_dim)
        self.global_attention = GlobalMultiHeadAttention(
            hidden_dim, num_heads=attention_heads, dropout=dropout
        )

        self.global_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 4. Fusion and Prediction
        # Combining Local and Global hidden states
        self.fusion_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.readout = nn.Linear(hidden_dim, forecast_steps)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        demand: torch.Tensor,
        time_of_day: torch.Tensor,
        day_of_week: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
        - demand: `(B, T, N)`
        - time_of_day: `(B, T)`
        - day_of_week: `(B, T)`
        
        Output: `(B, P, N)`
        """
        batch_size, history_steps, num_zones = demand.shape
        
        # Add feature dim to demand `(B, T, N, 1)`
        demand_feat = demand.unsqueeze(-1)

        # `(B, T, N, D)`
        embedded = self.embedding(demand_feat, time_of_day, day_of_week)

        # === Local Stream ===
        # 1. Dynamic Graph: `(B, T, N, N)`
        dynamic_adj = self.dynamic_graph_generator(embedded)
        
        # 2. AFT Local: `(B, N, T, D)`
        # AFT expects `(B, N, T, D)`
        embedded_bntd = embedded.permute(0, 2, 1, 3)
        aft_out = self.aft_local(embedded_bntd)
        
        # Return to `(B, T, N, D)`
        aft_out = aft_out.permute(0, 2, 1, 3)
        
        # 3. Dynamic GCN
        local_transformed = self.local_gcn_proj(aft_out)
        # Batched matmul over nodes per timestep
        local_gcn_out = torch.einsum("btnm,btmd->btnd", dynamic_adj, local_transformed)
        local_gcn_out = torch.relu(local_gcn_out)

        # 4. Local Temporal GRU
        local_gru_in = local_gcn_out.permute(0, 2, 1, 3).reshape(batch_size * num_zones, history_steps, self.hidden_dim)
        _, local_hidden = self.local_gru(local_gru_in)
        local_encoded = local_hidden[-1]  # `(B * N, D)`


        # === Global Stream ===
        # Reshape for static GCNs: `(B*T, N, D)`
        embedded_bt = embedded.view(batch_size * history_steps, num_zones, self.hidden_dim)
        
        distance_features = torch.relu(self.distance_gcn(embedded_bt, self.distance_adjacency))
        od_features = torch.relu(self.od_flow_gcn(embedded_bt, self.od_flow_adjacency))
        
        distance_features = distance_features.view(batch_size, history_steps, num_zones, self.hidden_dim)
        od_features = od_features.view(batch_size, history_steps, num_zones, self.hidden_dim)

        # Semantic fusion: `(B, T, N, D)`
        global_fused = self.global_attention(distance_features, od_features)

        # Temporal GRU
        global_gru_in = global_fused.permute(0, 2, 1, 3).reshape(batch_size * num_zones, history_steps, self.hidden_dim)
        _, global_hidden = self.global_gru(global_gru_in)
        global_encoded = global_hidden[-1]  # `(B * N, D)`


        # === Fusion & Readout ===
        # `(B * N, 2D)`
        combined = torch.cat([local_encoded, global_encoded], dim=-1)
        fused_state = torch.relu(self.fusion_linear(combined))

        # `(B * N, P)`
        forecast_delta = self.readout(fused_state)
        forecast_delta = forecast_delta.view(batch_size, num_zones, self.forecast_steps).permute(0, 2, 1)

        persistence = demand[:, -1:, :].repeat(1, self.forecast_steps, 1)
        return persistence + forecast_delta
