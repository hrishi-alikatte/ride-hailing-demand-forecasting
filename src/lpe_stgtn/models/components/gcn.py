"""Graph convolution layers used by stage-4 graph baselines."""

from __future__ import annotations

import torch
from torch import nn


class GraphConvolution(nn.Module):
    """Apply one adjacency-weighted linear graph convolution.

    Input shape:
    - node_features: `(B, N, F_in)`
    - adjacency: `(N, N)`

    Output shape:
    - `(B, N, F_out)`
    """

    def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        transformed = self.linear(node_features)
        return torch.einsum("nm,bmf->bnf", adjacency, transformed)
