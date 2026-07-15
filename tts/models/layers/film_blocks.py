from typing import override

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Modulates a feature map `x` with a conditioning vector `cond`.
    The conditioning vector is projected to produce a scale (gamma) and shift (beta)
    that are applied to the feature map.
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        # Project cond_dim to produce scale and shift for the given channels
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    @override
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map (B, C, T) or (B, C, H, W).
            cond (torch.Tensor): Conditioning vector (B, D_cond).
        Returns:
            torch.Tensor: Modulated feature map.
        """
        # Project cond to get gamma (scale) and beta (shift)
        gamma_beta = self.cond_proj(cond)

        # Reshape to match the input feature map `x`
        # Split into scale and shift, keeping channel dimension
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        # Add dimensions for broadcasting, e.g., (B, C) -> (B, C, 1) for Conv1d
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return x * gamma + beta


class FiLMResidualBlock(nn.Module):
    """
    A single block for the AuxiliaryDecoder, combining a dilated Conv1d,
    FiLM for conditioning, and activation.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.film = FiLMLayer(channels, cond_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    @override
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, T)
            cond: Conditioning vector (B, D_cond)
        """
        residual = x
        x = self.conv(x)

        # Ensure sequence length is maintained
        if x.size(2) > residual.size(2):
            x = x[:, :, : residual.size(2)]
        elif x.size(2) < residual.size(2):
            x = nn.functional.pad(x, (0, residual.size(2) - x.size(2)))

        x = self.film(x, cond)
        x = self.act(x)
        x = self.drop(x)

        return self.norm((residual + x).transpose(1, 2)).transpose(1, 2)
