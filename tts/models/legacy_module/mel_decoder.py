from typing import override

import torch
import torch.nn as nn

from ..layers.linear import Linear
from .conformer import ConformerEncoder


class MelDecoder(nn.Module):
    """
    Conformer-based Acoustic Decoder for TTS.

    It takes the length-regulated (aligned) text features as input and
    processes them through Conformer blocks to capture both local and global
    contexts, finally projecting them to acoustic features (Mel-spectrogram).

    Args:
        num_mels (int): Number of mel-spectrogram channels (usually 80 or 100).
        input_dim (int, optional): Dimension of the input feature vector.
        encoder_dim (int, optional): Hidden dimension of the conformer encoder.
        num_encoder_layers (int, optional): Number of conformer blocks.
            (Note: 4~6 layers are generally sufficient for a TTS decoder,
            unlike ASR which often requires deeper networks like 17).
        num_attention_heads (int, optional): Number of attention heads.
        feed_forward_expansion_factor (int, optional): Expansion factor for FFN.
        conv_expansion_factor (int, optional): Expansion factor for the convolution module.
        input_dropout_p (float, optional): Dropout probability for the input.
        feed_forward_dropout_p (float, optional): Dropout probability for the FFN.
        attention_dropout_p (float, optional): Dropout probability for the attention module.
        conv_dropout_p (float, optional): Dropout probability for the convolution module.
        conv_kernel_size (int, optional): Kernel size of the 1D convolution (should be odd).
        half_step_residual (bool): Whether to use Macaron-net style half-step residuals.
        cond_in_channels (int): Dimension of the condition vector (e.g., speaker embedding).

    Inputs: inputs, input_lengths, cond, mask
        - **inputs** (batch, time, dim): Tensor containing the length-regulated features.
        - **input_lengths** (batch): Tensor containing the valid lengths of each sequence.
        - **cond** (batch, cond_in_channels): Tensor containing the condition vectors.
        - **mask** (batch, time): Tensor indicating valid regions (= True or 1.0)

    Returns: mel_outputs, output_lengths
        - **mel_outputs** (batch, time, num_mels): Predicted mel-spectrogram frames.
        - **output_lengths** (batch): List or tensor of sequence output lengths.
    """

    def __init__(
        self,
        num_mels: int = 80,
        input_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        cond_in_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            cond_in_channels=cond_in_channels,
        )

        self.mel_proj = Linear(hidden_dim, num_mels, bias=True)

    def count_parameters(self) -> int:
        """Count total parameters of the decoder."""
        return self.encoder.count_parameters() + sum(p.numel() for p in self.mel_proj.parameters())

    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of the conformer encoder."""
        self.encoder.update_dropout(dropout_p)

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        encoder_outputs, encoder_output_lengths = self.encoder(
            inputs=inputs,
            input_lengths=input_lengths,
            cond=cond,
            mask=mask,
        )

        mel_outputs = self.mel_proj(encoder_outputs)
        mel_outputs = mel_outputs * mask.unsqueeze(-1).float()
        return mel_outputs, encoder_output_lengths
