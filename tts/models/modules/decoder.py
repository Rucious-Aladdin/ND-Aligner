from typing import override

import torch
import torch.nn as nn

from ..layers.film_blocks import FiLMResidualBlock


class Decoder(nn.Module):
    """
    Shallow auxiliary decoder with FiLM conditioning and dilated convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        cond_dim: int,
        kernel_sizes: list[int] | None = None,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.layers = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            dilation = dilation_base**i
            self.layers.append(
                FiLMResidualBlock(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Aligned features (B, T_mel, C)
            cond: Conditioning vector (B, D_cond)
            mask: Spec mask (B, T_mel)

        Returns:
            Predicted mel-spectrogram (B, T_mel, n_mels)
        """
        x = x.transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask_conv = mask.unsqueeze(1)
        else:
            mask_conv = mask

        x = self.input_proj(x)
        if mask_conv is not None:
            x = x * mask_conv

        for layer in self.layers:
            x = layer(x, cond)
            if mask_conv is not None:
                x = x * mask_conv

        x = self.output_proj(x)
        if mask_conv is not None:
            x = x * mask_conv

        return x
