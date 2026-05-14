from typing import override

import torch
import torch.nn as nn

from ..layers.film_layer import FiLMLayer


class SpecDecoderBlock(nn.Module):
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


class SpecDecoder(nn.Module):
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
                SpecDecoderBlock(
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

        return x.transpose(1, 2)
