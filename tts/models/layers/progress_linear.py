from typing import override

import torch
import torch.nn as nn


class ProgressLinear(nn.Module):
    """
    A non-causal encoder that extracts features from the Mel-spectrogram for the Aligner.
    Can use future context to provide more stable alignment features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.hidden_dim = out_dim
        self.input_proj = (
            nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        )

        # Fractional Position Encoder
        self.pos_proj = nn.Linear(1, out_dim)
        self.pos_mlp = nn.Sequential(
            nn.Conv1d(out_dim * 2, out_dim, 1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
    ):
        B, C, T = x.shape  # pyright: ignore
        device = x.device

        x = self.input_proj(x)

        t_idx = torch.arange(T, device=device).float()
        if T > 1:
            progress = (t_idx / (T - 1)).view(1, T, 1)  # (1, T, 1)
        else:
            progress = torch.zeros(1, T, 1, device=device)

        pos_emb = self.pos_proj(progress).transpose(1, 2)  # (1, hidden_dim, T)
        pos_emb = pos_emb.expand(B, -1, -1)

        # Concat and MLP
        out = torch.cat([x, pos_emb], dim=1)
        out = self.pos_mlp(out)
        return out
