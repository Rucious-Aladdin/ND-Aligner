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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, C, T = x.shape  # pyright: ignore
        device = x.device

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        x = self.input_proj(x)
        lengths = mask.sum(dim=-1).squeeze(1)  # (B,)

        t_idx = torch.arange(T, device=device).float().unsqueeze(0)  # (1, T)

        max_lens = torch.clamp(lengths - 1, min=1.0).unsqueeze(1)  # (B, 1)
        progress = t_idx / max_lens  # (B, T)

        progress = progress * mask.squeeze(1)  # (B, T)
        progress = progress.unsqueeze(-1)  # (B, T, 1)

        pos_emb = self.pos_proj(progress).transpose(1, 2)  # (B, hidden_dim, T)

        out = torch.cat([x, pos_emb], dim=1)
        out = self.pos_mlp(out)

        return out * mask

    def forward_sawtooth(
        self,
        x: torch.Tensor,  # (B, C_text, T_mel) - aligned_feats
        mask: torch.Tensor,  # (B, 1, T_mel) - spec_mask
        hard_gamma: torch.Tensor,  # (B, T_mel, T_text) - Viterbi hard alignment
    ) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        x = self.input_proj(x)

        cumsum_gamma = torch.cumsum(hard_gamma, dim=1)  # (B, T_mel, T_text)
        current_idx = torch.sum(cumsum_gamma * hard_gamma, dim=-1)  # (B, T_mel)

        hard_dur = hard_gamma.sum(dim=1, keepdim=True)  # (B, 1, T_text)

        dur_per_frame = torch.sum(hard_dur * hard_gamma, dim=-1)  # (B, T_mel)

        max_lens = torch.clamp(dur_per_frame, min=1.0)
        progress = current_idx / max_lens  # (B, T_mel)

        progress = progress * mask.squeeze(1)  # (B, T_mel)
        progress = progress.unsqueeze(-1)  # (B, T_mel, 1)

        pos_emb = self.pos_proj(progress).transpose(1, 2)  # (B, hidden_dim, T_mel)
        out = torch.cat([x, pos_emb], dim=1)
        out = self.pos_mlp(out)

        return out * mask
