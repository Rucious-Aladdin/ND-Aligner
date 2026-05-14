from typing import override

import torch
import torch.nn as nn


class FiLMResidualBlock(nn.Module):
    """
    Conv1d -> LayerNorm -> FiLM(cond) -> GELU -> Dropout -> Residual.

    FiLM:
        h = h * (1 + gamma(cond)) + beta(cond)

    The FiLM projection is zero-initialized, so the block initially behaves
    like the non-FiLM residual block.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int,
        dilation: int,
        dropout_p: float,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="zeros",
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.film = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_p)

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim, T)
            mask: (B, 1, T)
            cond: (B, cond_dim) or (B, cond_dim, 1)

        Returns:
            x:    (B, hidden_dim, T)
        """
        residual = x

        x = self.conv(x)

        if x.size(2) > residual.size(2):
            x = x[:, :, : residual.size(2)]

        # LayerNorm normalizes over hidden_dim.
        # Conv1d gives (B, C, T), so convert to (B, T, C).
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        if cond.dim() == 3:
            cond = cond.squeeze(-1)

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        x = x * (1.0 + gamma) + beta

        x = self.act(x)
        x = self.drop(x)

        return (x + residual) * mask


class SpecEncoder(nn.Module):
    """
    A non-causal speaker-conditioned encoder that extracts frame-level
    speech features from the Mel-spectrogram.

    It returns h_spec for pairwise speech-text emission scoring.

    Returns:
        h_spec : hidden speech representation, (B, hidden_dim, T_s)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        dilation_sizes: list[int] | None = None,
    ):
        super().__init__()

        if dilation_sizes is None:
            dilation_sizes = [1, 1, 1, 1]

        if len(dilation_sizes) == 0:
            raise ValueError("dilation_sizes must contain at least one dilation value.")

        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dilation_sizes = dilation_sizes

        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        for dilation in dilation_sizes:
            self.layers.append(
                FiLMResidualBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_p=dropout_p,
                )
            )

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    Raw mel spectrogram inputs, (B, in_dim, T_s)
            mask: Spectrogram mask, (B, T_s) or (B, 1, T_s)
            cond: Speaker/global condition, (B, cond_dim) or (B, cond_dim, 1)

        Returns:
            h_spec: (B, hidden_dim, T_s)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).float()
        else:
            mask = mask.float()

        if cond.dim() == 3:
            cond = cond.squeeze(-1)

        out = self.input_proj(x) * mask

        for layer in self.layers:
            out = layer(out, mask, cond)

        h_spec = out * mask

        return h_spec
