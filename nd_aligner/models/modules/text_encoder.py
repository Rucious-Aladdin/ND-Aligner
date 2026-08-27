from typing import override

import torch
import torch.nn as nn

from nd_aligner.models.modules.layers.film_blocks import FiLMResidualBlock
from nd_aligner.models.modules.utils.pos_encoding import PositionalEncoding1d


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        dim_out: int,
        dim_hidden: int,
        kernel_sizes: list[int],
        dim_cond: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one kernel size.")
        if dim_cond < 0:
            raise ValueError(f"dim_cond must be >= 0, got {dim_cond}.")

        self.dim_cond = dim_cond

        self.embedding = nn.Embedding(n_vocab, dim_hidden)
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=dim_hidden**-0.5,
        )

        self.pe = PositionalEncoding1d(channels=dim_hidden)

        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(
                    channels=dim_hidden,
                    cond_dim=dim_cond,
                    kernel_size=kernel_size,
                    dilation=1,
                    dropout=dropout,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.out_proj = nn.Conv1d(
            dim_hidden,
            dim_out,
            kernel_size=1,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      token ids, (B, T_text)
            x_mask: text mask, (B, T_text) or (B, 1, T_text)
            cond:   conditioning vector, (B, dim_cond)

        Returns:
            h_text: (B, dim_out, T_text)
        """
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(1)

        h_text = self.embedding(x).transpose(1, 2)
        x_mask = x_mask.to(
            device=h_text.device,
            dtype=h_text.dtype,
        )

        h_text = self.pe(h_text)
        h_text = h_text * x_mask

        if self.dim_cond > 0:
            for block in self.blocks:
                h_text = block(h_text, cond)
                h_text = h_text * x_mask
        else:
            for block in self.blocks:
                h_text = block(h_text, x_mask)

        h_text = self.out_proj(h_text)
        h_text = h_text * x_mask

        return h_text
