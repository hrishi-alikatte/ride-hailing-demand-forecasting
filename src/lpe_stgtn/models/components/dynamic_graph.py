"""Dynamic spatio-temporal evolving graph generator."""

from __future__ import annotations

import math

import torch
from torch import nn


class DynamicGraphGenerator(nn.Module):
    """Generates period-varying graphs from node embeddings.

    As the paper's exact 'shared pattern pool' formulation is under-specified,
    this implements a standard dynamic scaled dot-product spatial correlation
    graph. For each timestep, the spatial adjacency is derived dynamically from
    the hidden node states.

    Input shape:
    - hidden_states: `(B, T, N, D)`

    Output shape:
    - dynamic_adjacency: `(B, T, N, N)`
    """

    def __init__(self, hidden_dim: int, embed_dim: int = 32, num_nodes: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(hidden_dim, embed_dim)
        self.key_proj = nn.Linear(hidden_dim, embed_dim)
        
        # Optional static node embedding to blend in geographic priors
        if num_nodes is not None:
            self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        else:
            self.register_parameter("node_emb", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `(B, T, N, D)`
        queries = self.query_proj(hidden_states)
        keys = self.key_proj(hidden_states)

        if self.node_emb is not None:
            # Broadcast `(N, D)` to `(B, T, N, D)`
            queries = queries + self.node_emb.unsqueeze(0).unsqueeze(0)
            keys = keys + self.node_emb.unsqueeze(0).unsqueeze(0)

        # Matmul over nodes: `(B, T, N, D) @ (B, T, D, N) -> (B, T, N, N)`
        scores = torch.einsum("b t n d, b t m d -> b t n m", queries, keys)
        scores = scores / math.sqrt(self.embed_dim)

        # Softmax adjacency over the geographic dimension
        return torch.softmax(scores, dim=-1)
