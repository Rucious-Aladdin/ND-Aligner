import math
from typing import override

import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection layer for encoding scalar values, such as time steps or ratios.
    Maps a scalar to a high-dimensional vector using a set of fixed-frequency sinusoids.
    """

    def __init__(self, embedding_dim: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample Fourier frequencies from a Gaussian distribution
        # These are fixed during training
        w = torch.randn(embedding_dim // 2) * scale
        self.register_buffer("W", w)
        self.W: torch.Tensor

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of scalar values, shape (B,).

        Returns:
            torch.Tensor: A tensor of Fourier features, shape (B, embedding_dim).
        """
        # Ensure x is (B, 1) for broadcasting
        if x.dim() == 1:
            x = x.unsqueeze(1)

        # Project x onto the Fourier frequencies
        # x_proj shape: (B, embedding_dim / 2)
        x_proj = x * self.W

        # Concatenate sin and cos components
        # Result shape: (B, embedding_dim)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
