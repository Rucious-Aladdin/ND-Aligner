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
