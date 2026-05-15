from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConv1dBlock(nn.Module):
    """
    Conv1d -> LayerNorm -> FiLM(cond) -> GELU -> Dropout -> Residual.

    Input/Output:
        x:    (B, hidden_dim, T)
        mask: (B, 1, T)
        cond: (B, cond_dim) or (B, cond_dim, 1)
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
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="zeros",
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.film = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_p)

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        h = self.conv(x)

        if h.size(2) > residual.size(2):
            h = h[:, :, : residual.size(2)]
        elif h.size(2) < residual.size(2):
            h = F.pad(h, (0, residual.size(2) - h.size(2)))

        h = h.transpose(1, 2)
        h = self.norm(h)
        h = h.transpose(1, 2)

        if cond.dim() == 3:
            cond = cond.squeeze(-1)

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        h = h * (1.0 + gamma) + beta
        h = self.act(h)
        h = self.drop(h)

        return (h + residual) * mask


class ContentProgressPredictor(nn.Module):
    """
    Learnable speech-side progress predictor.

    Returns:
        progress_matrix: (B, h_progress, T_spec, T_text)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cond_dim: int,
        h_progress: int,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        dilation_sizes: list[int] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if dilation_sizes is None:
            dilation_sizes = [1, 2, 4, 8]

        if len(dilation_sizes) == 0:
            raise ValueError("dilation_sizes must contain at least one dilation value.")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.h_progress = h_progress
        self.eps = eps

        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                FiLMConv1dBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_p=dropout_p,
                )
                for dilation in dilation_sizes
            ]
        )

        self.mass_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # (B, 1, T_spec, T_text) -> (B, h_progress, T_spec, T_text)
        self.progress_proj = nn.Conv2d(
            in_channels=1,
            out_channels=h_progress,
            kernel_size=1,
        )

        # (B, h_progress, T_spec, T_text) -> (B, 1, T_spec, T_text)
        self.progress_out_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=h_progress,
                out_channels=h_progress,
                kernel_size=3,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=h_progress,
                out_channels=1,
                kernel_size=3,
            ),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        spec_mask: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:            Mel spectrogram, (B, in_dim, T_spec)
            c:            Condition, (B, cond_dim) or (B, cond_dim, 1)
            spec_mask:    (B, T_spec) or (B, 1, T_spec)
            text_lengths: (B,)

        Returns:
            progress_matrix: (B, h_progress, T_spec, T_text_max)
        """
        if spec_mask.dim() == 2:
            mask = spec_mask.unsqueeze(1).to(dtype=x.dtype)
        else:
            mask = spec_mask.to(dtype=x.dtype)

        if c.dim() == 3:
            c = c.squeeze(-1)

        h = self.input_proj(x) * mask

        for layer in self.layers:
            h = layer(h, mask, c)

        # 1. Predict nonnegative mass.
        mass_logits = self.mass_head(h).squeeze(1)  # (B, T_spec)

        mask_2d = mask.squeeze(1)
        mass_raw = F.softplus(mass_logits) * mask_2d

        # Prevent all-zero collapse.
        mass_raw = mass_raw + self.eps * mask_2d

        # 2. Cumsum, then normalize to [0, 1].
        cum_end = torch.cumsum(mass_raw, dim=1)
        cum_start = cum_end - mass_raw

        total = cum_end[:, -1:].clamp_min(self.eps)

        progress_start = cum_start / total
        progress_end = cum_end / total

        progress_start = progress_start * mask_2d
        progress_end = progress_end * mask_2d

        # 3. Use progress_mid as speech-side progress.
        speech_progress_mid = 0.5 * (progress_start + progress_end)  # (B, T_spec)

        spec_lengths = mask_2d.sum(dim=1).long()

        return self.compute_progress_matrix(
            text_lengths=text_lengths,
            spec_lengths=spec_lengths,
            speech_progress=speech_progress_mid,
            pad_value=1.0,
        )

    def compute_progress_matrix(
        self,
        text_lengths: torch.Tensor,
        spec_lengths: torch.Tensor,
        speech_progress: torch.Tensor,
        pad_value: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            text_lengths:    (B,)
            spec_lengths:    (B,)
            speech_progress: (B, T_spec) or (B, 1, T_spec)
            pad_value:       value used for padded/invalid cells after projection

        Returns:
            progress_matrix: (B, h_progress, T_spec, T_text_max)
        """
        if speech_progress.dim() == 3:
            speech_progress = speech_progress.squeeze(1)

        B, T_spec = speech_progress.shape
        T_text = int(text_lengths.max().item())

        device = speech_progress.device
        dtype = speech_progress.dtype

        text_lengths = text_lengths.to(device=device)
        spec_lengths = spec_lengths.to(device=device)

        text_lengths_f = text_lengths.to(dtype=dtype).clamp_min(1.0)

        # ------------------------------------------------------------
        # 1. Text progress midpoint.
        #
        # Token j occupies approximately:
        #   [j / N, (j + 1) / N]
        #
        # Midpoint:
        #   (j + 0.5) / N
        # ------------------------------------------------------------
        j_float = torch.arange(T_text, device=device, dtype=dtype).view(1, 1, T_text)
        text_progress_mid = (j_float + 0.5) / text_lengths_f.view(B, 1, 1)

        # ------------------------------------------------------------
        # 2. Signed pairwise difference.
        # ------------------------------------------------------------
        speech_progress = speech_progress.view(B, T_spec, 1)
        progress_diff = speech_progress - text_progress_mid  # (B, T_spec, T_text)

        # ------------------------------------------------------------
        # 3. Build validity mask.
        # ------------------------------------------------------------
        t_idx = torch.arange(T_spec, device=device).view(1, T_spec, 1)
        j_idx = torch.arange(T_text, device=device).view(1, 1, T_text)

        spec_valid = t_idx < spec_lengths.view(B, 1, 1)
        text_valid = j_idx < text_lengths.view(B, 1, 1)

        valid = spec_valid & text_valid  # (B, T_spec, T_text)

        # ------------------------------------------------------------
        # 4. Project scalar progress difference to h_progress channels.
        # ------------------------------------------------------------
        progress_matrix = progress_diff.unsqueeze(1)  # (B, 1, T_spec, T_text)
        progress_matrix = self.progress_proj(progress_matrix)

        # Mask after projection so invalid cells have a fixed value
        # regardless of Conv2d bias/weights.
        progress_matrix = progress_diff.unsqueeze(1)  # (B, 1, T_spec, T_text)
        progress_hidden = self.progress_proj(progress_matrix)
        progress_matrix = self.progress_out_proj(progress_hidden)

        progress_matrix = progress_matrix.masked_fill(
            ~valid.unsqueeze(1),
            pad_value,
        )

        return progress_matrix
