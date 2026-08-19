from typing import override

import torch
import torch.nn as nn


class FiLMLayer2D(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for 2D feature maps.

    Args:
        channels: Number of feature channels.
        cond_dim: Dimension of conditioning vector.

    Input:
        x:    (B, C, H, W)
        cond: (B, D_cond)

    Output:
        (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
    ):
        super().__init__()

        self.channels = channels
        self.cond_dim = cond_dim

        self.cond_proj = nn.Linear(
            cond_dim,
            channels * 2,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        gamma_beta = self.cond_proj(cond)
        gamma, beta = torch.chunk(
            gamma_beta,
            chunks=2,
            dim=1,
        )

        # (B, C) -> (B, C, 1, 1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return x * gamma + beta


class FiLMResidualConv2D(nn.Module):
    """
    Residual Conv2D block with FiLM conditioning.

    Structure:
        Conv2D
        -> FiLM
        -> GELU
        -> Dropout
        -> Residual Add
        -> LayerNorm

    Input:
        x:    (B, C, T, N)
        cond: (B, D_cond)

    Output:
        (B, C, T, N)
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.film = FiLMLayer2D(
            channels=channels,
            cond_dim=cond_dim,
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Feature grid of shape (B, C, T, N).

            cond:
                Conditioning vector of shape (B, D_cond).

        Returns:
            Feature grid of shape (B, C, T, N).
        """
        residual = x

        x = self.conv(x)
        x = self.film(x, cond)
        x = self.act(x)
        x = self.drop(x)

        # LayerNorm over channel dimension:
        # (B, C, T, N)
        # -> (B, T, N, C)
        x = residual + x

        x = self.norm(x.permute(0, 2, 3, 1))

        # (B, T, N, C)
        # -> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)

        return x
