from __future__ import annotations
from typing import override


import torch
import torch.nn as nn

from tts.models.layers.progress_linear import ProgressLinear


class ProgressEncoder(nn.Module):
    """
    Integrated TTS model that combines Stage 1 (Alignment & Coarse Mel)
    with Stage 2 (EDM Diffusion Refinement).
    """

    def __init__(
        self,
        in_dim: int,
        progress_hidden_dim: int = 192,
        apply_local_text_progress: bool = False,
        apply_global_text_progress: bool = False,
        apply_spec_progress: bool = False,
    ) -> None:
        super().__init__()

        # Applying Progress Conditioning-Aware
        self.progress_cond_dim = progress_hidden_dim

        self.local_text_progress_layer = None
        self.global_text_progress_layer = None
        self.spec_progress_layer = None
        self.progress_cond_dim = 0

        num_layers = 0
        if apply_global_text_progress:
            self.global_text_progress_layer = ProgressLinear(
                in_dim=in_dim,
                out_dim=progress_hidden_dim,
            )
            num_layers += 1

        if apply_local_text_progress:
            self.local_text_progress_layer = ProgressLinear(
                in_dim=in_dim,
                out_dim=progress_hidden_dim,
            )
            num_layers += 1

        if apply_spec_progress:
            self.spec_progress_layer = ProgressLinear(
                in_dim=in_dim,
                out_dim=progress_hidden_dim,
            )
            num_layers += 1

        self.progress_mlp = None
        if num_layers > 0:
            self.progress_cond_dim = progress_hidden_dim
            self.progress_mlp = nn.Sequential(
                nn.Conv1d(
                    in_channels=progress_hidden_dim,
                    out_channels=progress_hidden_dim * 2,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv1d(
                    in_channels=progress_hidden_dim * 2,
                    out_channels=in_dim,
                    kernel_size=1,
                ),
            )

    @override
    def forward(
        self,
        h_text: torch.Tensor,  # (B, C_text, T_text)
        aligned_feats: torch.Tensor,  # (B, C_text, T_mel)
        hard_gamma: torch.Tensor,  # (B, T_mel, T_text) - Viterbi hard alignment
        text_mask: torch.Tensor,  # (B, 1, T_text)
        spec_mask: torch.Tensor,  # (B, 1, T_mel)
    ) -> torch.Tensor:
        if self.progress_mlp is None:
            return aligned_feats  # No progress conditioning applied

        B, _, T_mel = aligned_feats.shape
        device = aligned_feats.device

        accumulated = torch.zeros(
            B, self.progress_cond_dim, T_mel, device=device, dtype=aligned_feats.dtype
        )

        if self.global_text_progress_layer is not None:
            global_prog = self.global_text_progress_layer(h_text, text_mask)
            global_prog_mel = torch.bmm(global_prog, hard_gamma.transpose(1, 2))
            accumulated = accumulated + global_prog_mel

        if self.local_text_progress_layer is not None:
            local_prog_mel = self.local_text_progress_layer.forward_sawtooth(
                x=aligned_feats,
                mask=spec_mask,
                hard_gamma=hard_gamma,
            )
            accumulated = accumulated + local_prog_mel

        if self.spec_progress_layer is not None:
            spec_prog_mel = self.spec_progress_layer(aligned_feats, spec_mask)
            accumulated = accumulated + spec_prog_mel

        aligned_feats = aligned_feats + self.progress_mlp(accumulated)

        return aligned_feats


class ResConv1dBlock(nn.Module):
    """
    Lightweight residual temporal smoothing block.

    Shape:
        x:    (B, C, T)
        mask: (B, 1, T)

    The block starts close to identity because the final pointwise projection is
    zero-initialized.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        padding = kernel_size // 2

        self.net = nn.Sequential(
            # cheap temporal smoothing
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.net(x * mask) * mask
        return (x + delta) * mask


class SmoothingNeuralNet(nn.Module):
    """
    Residual-stack temporal smoothing network for frame-level conditioning.

    This module returns a residual delta, not the final smoothed feature.

    Input:
        x:    (B, C, T)
        mask: (B, 1, T) or (B, T)

    Output:
        delta: (B, C, T)

    Usage:
        x = x + alpha * smoothing_net(x, mask)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 192,
        kernel_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                ResConv1dBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, C, T), got {tuple(x.shape)}")

        if mask is None:
            mask = torch.ones(
                x.size(0),
                1,
                x.size(-1),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

            if mask.shape[0] != x.shape[0] or mask.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} is incompatible with x shape {tuple(x.shape)}"
                )

            mask = mask.to(dtype=x.dtype, device=x.device)

        h = self.in_proj(x * mask) * mask

        for block in self.blocks:
            h = block(h, mask)

        delta = self.out_proj(h) * mask
        return delta


class ConditionAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        progress_hidden_dim: int = 128,
        smoothing_hidden_dim: int = 128,
        smoothing_kernel_size: int = 3,
        smoothing_num_layers: int = 4,
        apply_smoothing: bool = False,
        apply_local_text_progress: bool = False,
        apply_global_text_progress: bool = False,
        apply_spec_progress: bool = False,
    ) -> None:
        super().__init__()
        self.progress_encoder = ProgressEncoder(
            in_dim=in_dim,
            progress_hidden_dim=progress_hidden_dim,
            apply_local_text_progress=apply_local_text_progress,
            apply_global_text_progress=apply_global_text_progress,
            apply_spec_progress=apply_spec_progress,
        )

        self.smoothing_net = None
        if apply_smoothing:
            self.smoothing_net = SmoothingNeuralNet(
                in_dim=in_dim,
                hidden_dim=smoothing_hidden_dim,
                kernel_size=smoothing_kernel_size,
                num_layers=smoothing_num_layers,
            )

    @override
    def forward(
        self,
        h_text: torch.Tensor,  # (B, C_text, T_text)
        aligned_feats: torch.Tensor,  # (B, C_text, T_mel)
        hard_gamma: torch.Tensor,  # (B, T_mel, T_text) - Viterbi hard alignment
        text_mask: torch.Tensor,  # (B, 1, T_text)
        spec_mask: torch.Tensor,  # (B, 1, T_mel)
    ) -> torch.Tensor:
        x = self.progress_encoder(
            h_text=h_text,
            aligned_feats=aligned_feats,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            spec_mask=spec_mask,
        )

        if self.smoothing_net is not None:
            x = self.smoothing_net(x, spec_mask)

        return x * spec_mask


def _make_dummy_hard_gamma(
    durations: torch.Tensor,
    max_text_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        durations: (B, T_text), non-negative integer durations.
        max_text_len: int

    Returns:
        hard_gamma: (B, T_mel_max, T_text)
        y_lengths:  (B,)
    """
    B, T_text = durations.shape
    assert T_text == max_text_len

    y_lengths = durations.sum(dim=1)
    T_mel_max = int(y_lengths.max().item())

    hard_gamma = torch.zeros(B, T_mel_max, T_text, dtype=torch.float32)

    for b in range(B):
        cursor = 0
        for j in range(T_text):
            d = int(durations[b, j].item())
            if d <= 0:
                continue
            hard_gamma[b, cursor : cursor + d, j] = 1.0
            cursor += d

    return hard_gamma, y_lengths


if __name__ == "__main__":
    torch.manual_seed(1234)

    B = 2
    C = 256
    T_text = 7

    durations = torch.tensor(
        [
            [3, 5, 2, 4, 6, 3, 5],
            [4, 2, 5, 3, 4, 6, 0],
        ],
        dtype=torch.long,
    )

    hard_gamma, y_lengths = _make_dummy_hard_gamma(durations, max_text_len=T_text)
    T_mel = hard_gamma.size(1)

    x_lengths = torch.tensor([7, 6], dtype=torch.long)

    h_text = torch.randn(B, C, T_text)
    aligned_feats = torch.bmm(h_text, hard_gamma.transpose(1, 2))  # (B, C, T_mel)

    text_idx = torch.arange(T_text).unsqueeze(0)
    text_mask = (text_idx < x_lengths.unsqueeze(1)).unsqueeze(1).float()

    mel_idx = torch.arange(T_mel).unsqueeze(0)
    spec_mask = (mel_idx < y_lengths.unsqueeze(1)).unsqueeze(1).float()

    adapter = ConditionAdapter(
        in_dim=C,
        progress_hidden_dim=128,
        smoothing_hidden_dim=128,
        smoothing_kernel_size=3,
        smoothing_num_layers=2,
        apply_smoothing=True,
        apply_local_text_progress=True,
        apply_global_text_progress=True,
        apply_spec_progress=True,
    )

    out = adapter(
        h_text=h_text,
        aligned_feats=aligned_feats,
        hard_gamma=hard_gamma,
        text_mask=text_mask,
        spec_mask=spec_mask,
    )

    print("h_text:", h_text.shape)
    print("aligned_feats:", aligned_feats.shape)
    print("hard_gamma:", hard_gamma.shape)
    print("text_mask:", text_mask.shape)
    print("spec_mask:", spec_mask.shape)
    print("out:", out.shape)

    loss = out.pow(2).mean()
    loss.backward()

    num_grad_params = sum(p.grad is not None for p in adapter.parameters() if p.requires_grad)
    print("num trainable params:", sum(p.numel() for p in adapter.parameters()))
    print("num params with grad:", num_grad_params)
    print("test passed.")
