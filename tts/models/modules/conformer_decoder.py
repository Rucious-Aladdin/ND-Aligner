from typing import override

import torch
import torch.nn as nn


from ..submodules.conformer import ConformerEncoder


class ConformerSpecDecoder(nn.Module):
    """
    Conformer-based spectrogram decoder.

    Uses ConformerEncoder as the backbone:
        frame-level text features -> ConformerEncoder -> mel projection

    Input:
        x:         (B, C_in, T) or (B, T, C_in)
        spec_mask: (B, T) or (B, 1, T)
        cond:      (B, C_cond) or (B, C_cond, 1)

    Output:
        mel:       (B, out_dim, T)
    """

    def __init__(
        self,
        conformer_out_dim: int,
        conformer_input_dim: int,
        conformer_cond_dim: int,
        conformer_hidden_dim: int,
        conformer_num_layers: int,
        conformer_num_attention_heads: int,
        conformer_feed_forward_expansion_factor: int,
        conformer_conv_expansion_factor: int,
        conformer_input_dropout_p: float,
        conformer_feed_forward_dropout_p: float,
        conformer_attention_dropout_p: float,
        conformer_conv_dropout_p: float,
        conformer_conv_kernel_size: int,
        conformer_half_step_residual: bool,
        conformer_attn_window_size: int,
    ) -> None:
        super().__init__()

        self.input_dim = int(conformer_input_dim)
        self.cond_dim = int(conformer_cond_dim)
        self.hidden_dim = int(conformer_hidden_dim)
        self.out_dim = int(conformer_out_dim)

        self.conformer = ConformerEncoder(
            input_dim=self.input_dim,
            encoder_dim=self.hidden_dim,
            cond_in_channels=self.cond_dim,
            num_layers=conformer_num_layers,
            num_attention_heads=conformer_num_attention_heads,
            feed_forward_expansion_factor=conformer_feed_forward_expansion_factor,
            conv_expansion_factor=conformer_conv_expansion_factor,
            input_dropout_p=conformer_input_dropout_p,
            feed_forward_dropout_p=conformer_feed_forward_dropout_p,
            attention_dropout_p=conformer_attention_dropout_p,
            conv_dropout_p=conformer_conv_dropout_p,
            conv_kernel_size=conformer_conv_kernel_size,
            half_step_residual=conformer_half_step_residual,
            attn_window_size=conformer_attn_window_size,
        )

        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)

    @override
    def forward(
        self,
        x: torch.Tensor,
        spec_mask: torch.Tensor,  # (B, T)
        cond: torch.Tensor | None = None,  # (B, C_cond) or (B, C_cond, 1)
    ) -> torch.Tensor:
        spec_mask = spec_mask.bool()
        input_lengths = spec_mask.sum(dim=-1).long()

        encoded, _ = self.conformer(
            inputs=x,
            input_lengths=input_lengths,
            cond=cond,
            mask=spec_mask,
        )  # (B, T, hidden_dim)

        mel = self.out_proj(encoded)  # (B, T, out_dim)
        mel = mel.masked_fill(~spec_mask.unsqueeze(-1), 0.0)

        return mel.transpose(1, 2).contiguous()  # (B, out_dim, T)
