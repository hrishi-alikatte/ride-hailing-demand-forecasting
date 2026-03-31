"""Attention helpers for semantic graph fusion."""

from __future__ import annotations

import torch
from torch import nn


class SemanticFusionAttention(nn.Module):
    """Fuse distance and OD graph features with lightweight head-wise attention.

    This module attends across the two semantic graph views for each node
    independently rather than across all zone tokens. That keeps stage-4
    training practical on CPU while still allowing the model to weigh
    distance-derived and OD-derived features adaptively.

    Inputs:
    - distance_features: `(B, N, D)`
    - od_features: `(B, N, D)`

    Output:
    - `(B, N, D)`
    """

    def __init__(self, embed_dim: int, *, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"SemanticFusionAttention requires embed_dim divisible by num_heads, got "
                f"{embed_dim} and {num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.semantic_queries = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        distance_features: torch.Tensor,
        od_features: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([distance_features, od_features], dim=2)
        batch_size, num_nodes, _, _ = stacked.shape
        head_view = stacked.view(batch_size, num_nodes, 2, self.num_heads, self.head_dim)
        scores = torch.einsum(
            "bnshd,hd->bnsh",
            torch.tanh(head_view),
            self.semantic_queries,
        )
        weights = torch.softmax(scores, dim=2)
        fused_heads = torch.sum(weights.unsqueeze(-1) * head_view, dim=2)
        fused = fused_heads.reshape(batch_size, num_nodes, self.embed_dim)
        return self.layer_norm(self.dropout(fused) + distance_features)


class GlobalMultiHeadAttention(nn.Module):
    """Full semantic fusion via multi-head self-attention.

    The paper fuses the semantic representations from the distance graph
    and the OD-flow graph using standard multi-head self-attention, where
    each node acts as a token.

    Inputs:
    - distance_features: `(B, T, N, D)`
    - od_features: `(B, T, N, D)`

    Output:
    - fused: `(B, T, N, D)`
    """

    def __init__(self, embed_dim: int, *, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"GlobalMultiHeadAttention requires embed_dim divisible by num_heads, "
                f"got {embed_dim} and {num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # We concatenate the semantic features and compress back before MHA,
        # or we treat one as Q and one as K/V? The paper says: "fusion attention".
        # A common fusion is adding/concatenating them, then standard self-attention,
        # or cross-attention. We will sum them and self-attend.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        distance_features: torch.Tensor,
        od_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_steps, num_nodes, dim = distance_features.shape

        # Element-wise addition acts as the initial semantic fusion
        x = distance_features + od_features  # `(B, T, N, D)`
        
        # Batch over Time: `(B * T, N, D)`
        x_bt = x.view(batch_size * num_steps, num_nodes, dim)
        
        qkv = self.qkv_proj(x_bt)  # `(B*T, N, 3D)`
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = self.embed_dim // self.num_heads
        q = q.view(-1, num_nodes, self.num_heads, head_dim).transpose(1, 2)  # `(B*T, H, N, d)`
        k = k.view(-1, num_nodes, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(-1, num_nodes, self.num_heads, head_dim).transpose(1, 2)

        # Standard dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # `(B*T, H, N, d) -> (B*T, N, H, d) -> (B*T, N, D)`
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(-1, num_nodes, self.embed_dim)
        
        attended = self.out_proj(attended)
        
        # Residual and LayerNorm
        fused = self.layer_norm(x_bt + self.dropout(attended))
        
        return fused.view(batch_size, num_steps, num_nodes, dim)
