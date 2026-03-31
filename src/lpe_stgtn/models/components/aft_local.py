"""AFT-Local (Attention Free Transformer) block for chronological learning."""

from __future__ import annotations

import torch
from torch import nn


class AttentionFreeTransformerLocal(nn.Module):
    """Local spatial-temporal representation through an Attention-Free Transformer.

    AFT-local restrains the attention context to a small structural local
    window size, S, which reduces the quadratic parameter scaling of standard
    multi-head temporal attention to O(T x S) complexity.
    """

    def __init__(self, hidden_dim: int, window_size: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relative positional bias over the window size (symmetric)
        self.pos_bias = nn.Parameter(torch.zeros(window_size + 1, hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input: `(B, N, T, D)`
        Output: `(B, N, T, D)`
        """
        batch_size, num_nodes, num_steps, hidden_dim = hidden_states.shape

        q = self.query_proj(hidden_states)  # `(B, N, T, D)`
        k = self.key_proj(hidden_states)
        v = self.value_proj(hidden_states)

        # Pad to handle the local window easily
        pad_size = self.window_size // 2
        
        # `(B, N, D, T)` for F.pad over the T dimension
        v_padded = torch.nn.functional.pad(v.permute(0, 1, 3, 2), (pad_size, pad_size)).permute(0, 1, 3, 2)
        k_padded = torch.nn.functional.pad(k.permute(0, 1, 3, 2), (pad_size, pad_size)).permute(0, 1, 3, 2)
        
        out_slots = []
        for t in range(num_steps):
            # Window slice: `(B, N, S, D)`
            # In AFT, the context is `sum( exp(K + Pos) * V ) / sum( exp(K + Pos) )`
            # For causal or symmetric, we take a slide of `self.window_size + 1` length
            slice_len = pad_size * 2 + 1
            
            k_window = k_padded[:, :, t : t + slice_len, :]
            v_window = v_padded[:, :, t : t + slice_len, :]
            
            # `(B, N, S, D)`
            exp_k = torch.exp(k_window + self.pos_bias.unsqueeze(0).unsqueeze(0))
            
            numerator = torch.sum(exp_k * v_window, dim=2)
            denominator = torch.sum(exp_k, dim=2) + 1e-6
            
            context = numerator / denominator
            out_slots.append(context)
            
        # `(B, N, T, D)`
        attention_context = torch.stack(out_slots, dim=2)
        
        # Multiply with sigmoid(Q)
        y = torch.sigmoid(q) * attention_context
        y = self.output_proj(y)
        
        return self.layer_norm(hidden_states + self.dropout(y))
