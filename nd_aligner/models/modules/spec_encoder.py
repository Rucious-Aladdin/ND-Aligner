from typing import override

import torch
import torch.nn as nn

from nd_aligner.models.modules.layers.film_blocks import FiLMResidualBlock


class SpecEncoder(nn.Module):
    """
    Speaker-conditioned Conv1d encoder for Mel-spectrograms.

    Input:
        x    : (B, in_dim, T_s)
        mask : (B, T_s) or (B, 1, T_s)
        cond : (B, cond_dim) or (B, cond_dim, 1)

    Output:
        h_spec: (B, out_dim, T_s)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        dilation_sizes: list[int] | None = None,
    ):
        super().__init__()

        if dilation_sizes is None:
            dilation_sizes = [1, 1, 1, 1]

        if not dilation_sizes:
            raise ValueError("dilation_sizes must contain at least one dilation value.")

        self.cond_dim = cond_dim

        self.input_proj = nn.Conv1d(
            in_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.layers = nn.ModuleList(
            [
                FiLMResidualBlock(
                    channels=hidden_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout_p,
                )
                for dilation in dilation_sizes
            ]
        )

        self.out_proj = nn.Conv1d(
            hidden_dim,
            out_dim,
            kernel_size=1,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        if cond.dim() == 3:
            cond = cond.squeeze(-1)

        mask = mask.to(
            device=x.device,
            dtype=x.dtype,
        )
        cond = cond.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = self.input_proj(x)
        out = out * mask

        for layer in self.layers:
            out = layer(out, cond)
            out = out * mask

        out = self.out_proj(out)
        out = out * mask

        return out
