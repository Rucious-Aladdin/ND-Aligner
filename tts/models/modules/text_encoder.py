from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.models.layers.film_layer import FiLMLayer


class Conv1dBlock(nn.Module):
    """
    Conv1d -> LayerNorm -> GELU -> Residual
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
    ):
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Keep temporal length unchanged for both odd/even kernel sizes.

        Args:
            x: (B, C, T)

        Returns:
            padded x
        """
        k = self.kernel_size
        left = (k - 1) // 2
        right = k // 2
        return F.pad(x, (left, right))

    @override
    def forward(
        self,
        x: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, C, T_text)
            text_mask: (B, 1, T_text)

        Returns:
            x:         (B, C, T_text)
        """
        residual = x

        h = self._same_pad(x)
        h = self.conv(h)

        # LayerNorm over channel dimension.
        h = h.transpose(1, 2)  # (B, T_text, C)
        h = self.norm(h)
        h = h.transpose(1, 2)  # (B, C, T_text)

        h = self.act(h)

        x = residual + h
        x = x * text_mask

        return x


class TextEncoder(nn.Module):
    """
    Conv text encoder.

    Input:
        text_token_ids : token ids, (B, T_text)
        text_mask      : (B, T_text) or (B, 1, T_text)
        cond           : optional conditioning vector, (B, D_cond)

    Output:
        h_text         : token representation sequence, (B, dim_out, T_text)

    If dim_cond == 0:
        FiLM conditioning is disabled and cond may be None.

    If dim_cond > 0:
        FiLM conditioning is enabled and cond must be provided in forward().
    """

    def __init__(
        self,
        n_vocab: int,
        dim_out: int,
        dim_hidden: int,
        kernel_sizes: list[int],
        dim_cond: int = 0,
    ):
        super().__init__()

        if len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes must contain at least one kernel size.")
        if dim_cond < 0:
            raise ValueError(f"dim_cond must be >= 0, got {dim_cond}.")

        self.n_vocab = n_vocab
        self.hidden_channels = dim_hidden
        self.kernel_sizes = kernel_sizes
        self.dim_cond = int(dim_cond)

        self.out = nn.Embedding(n_vocab, dim_hidden)
        nn.init.normal_(self.out.weight, mean=0.0, std=dim_hidden**-0.5)

        self.conv_blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    channels=dim_hidden,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

        if self.dim_cond > 0:
            self.cond_film = FiLMLayer(
                channels=dim_hidden,
                cond_dim=self.dim_cond,
            )
        else:
            self.cond_film = None

        self.out_proj = nn.Conv1d(dim_hidden, dim_out, kernel_size=1)

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                token ids, (B, T_text)

            x_mask:
                text mask, (B, T_text) or (B, 1, T_text)

            cond:
                optional conditioning vector, (B, dim_cond).
                Required only when dim_cond > 0.

        Returns:
            h_text:
                (B, dim_out, T_text)
        """
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(1)

        x_mask = x_mask.to(dtype=self.out.weight.dtype)

        # (B, T_text) -> (B, T_text, hidden_channels)
        h_text = self.out(x)

        # (B, T_text, hidden_channels) -> (B, hidden_channels, T_text)
        h_text = h_text.transpose(1, 2)

        h_text = h_text * x_mask

        for block in self.conv_blocks:
            h_text = block(h_text, x_mask)

        h_text = h_text * x_mask

        if self.cond_film is not None:
            if cond is None:
                raise ValueError(
                    "cond must be provided when TextEncoder is initialized "
                    + f"with dim_cond={self.dim_cond}."
                )

            h_text = self.cond_film(h_text, cond)
            h_text = h_text * x_mask

        # (B, hidden_channels, T_text) -> (B, dim_out, T_text)
        h_text = self.out_proj(h_text)
        h_text = h_text * x_mask

        return h_text
