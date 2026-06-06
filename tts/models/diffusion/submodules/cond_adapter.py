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


class FilmResConv1dBlock(nn.Module):
    """
    FiLM-conditioned residual temporal smoothing block.

    Supports:
        cond: (B, C_cond)     global condition
        cond: (B, C_cond, 1)  global condition
        cond: (B, C_cond, T)  frame-wise condition

    Shape:
        x:    (B, C, T)
        cond: (B, C_cond) or (B, C_cond, T)
        mask: (B, 1, T)
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}.")

        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

        # Use 1x1 Conv so it works for both global and frame-wise condition.
        self.film = nn.Conv1d(
            in_channels=cond_dim,
            out_channels=channels * 2,
            kernel_size=1,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1)

        mask = mask.to(dtype=x.dtype, device=x.device)
        cond = cond.to(dtype=x.dtype, device=x.device)

        h = self.depthwise(x * mask)
        h = self.act(h)
        h = self.pointwise(h)

        gamma, beta = self.film(cond).chunk(2, dim=1)

        # If cond was (B, C_cond, 1), gamma/beta are (B, C, 1)
        # and broadcast over T automatically.
        h = h * (1.0 + gamma) + beta
        h = self.drop(h)

        delta = h * mask
        return (x + delta) * mask


class SmoothingNeuralNet(nn.Module):
    """
    FiLM-conditioned residual-stack temporal smoothing network.

    Input:
        x:    (B, C, T)
        cond: (B, C_cond)
        mask: (B, 1, T) or (B, T)

    Output:
        delta: (B, C, T)

    Usage:
        x = x + smoothing_net(x, spk, mask)
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dim: int = 192,
        kernel_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                FilmResConv1dBlock(
                    channels=hidden_dim,
                    cond_dim=cond_dim,
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
        cond: torch.Tensor,
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
            h = block(h, cond=cond, mask=mask)

        delta = self.out_proj(h) * mask
        return delta


class ConditionAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        spk_dim: int = 192,
        progress_hidden_dim: int = 128,
        smoothing_hidden_dim: int = 128,
        smoothing_kernel_size: int = 3,
        smoothing_num_layers: int = 4,
        apply_local_text_progress: bool = False,
        apply_global_text_progress: bool = False,
        apply_spec_progress: bool = False,
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        if spk_dim <= 0:
            raise ValueError(f"spk_dim must be positive, got {spk_dim}.")

        self.in_dim = in_dim
        self.spk_dim = spk_dim

        self.progress_encoder = ProgressEncoder(
            in_dim=in_dim,
            progress_hidden_dim=progress_hidden_dim,
            apply_local_text_progress=apply_local_text_progress,
            apply_global_text_progress=apply_global_text_progress,
            apply_spec_progress=apply_spec_progress,
        )

        self.smoothing_net = SmoothingNeuralNet(
            in_dim=in_dim,
            cond_dim=spk_dim,
            hidden_dim=smoothing_hidden_dim,
            kernel_size=smoothing_kernel_size,
            num_layers=smoothing_num_layers,
        )

        # Null tokens for CFG / condition masking.
        self.null_text = nn.Parameter(torch.randn(in_dim))
        self.null_spk = nn.Parameter(torch.randn(spk_dim))

    @override
    def forward(
        self,
        h_text: torch.Tensor,  # (B, C_text, T_text)
        aligned_feats: torch.Tensor,  # (B, C_text, T_mel)
        spk: torch.Tensor,  # (B, spk_dim)
        hard_gamma: torch.Tensor,  # (B, T_mel, T_text)
        text_mask: torch.Tensor,  # (B, 1, T_text)
        spec_mask: torch.Tensor,  # (B, 1, T_mel)
        *,
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        B, _C, _T_mel = aligned_feats.shape

        spec_mask = spec_mask.to(dtype=aligned_feats.dtype, device=aligned_feats.device)

        # 1. Build frame-level text condition.
        x = self.progress_encoder(
            h_text=h_text,
            aligned_feats=aligned_feats,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            spec_mask=spec_mask,
        )
        x = x * spec_mask

        # 2. Apply text condition dropout / continuous region masking.
        if text_cond_drop_prob > 0.0:
            drop_mask = torch.rand(B, device=x.device) < text_cond_drop_prob
            x = torch.where(
                drop_mask.view(B, 1, 1),
                self.null_text.view(1, -1, 1).to(dtype=x.dtype, device=x.device),
                x,
            )

        x = self._mask_continuous_region(
            x=x,
            ratio=text_cond_mask_ratio,
            null_val=self.null_text,
            mask=spec_mask,
        )
        x = x * spec_mask

        # 3. Speaker condition dropout / temporal region masking.
        if spk_cond_drop_prob > 0.0:
            drop_mask = torch.rand(B, device=spk.device) < spk_cond_drop_prob
            spk = torch.where(
                drop_mask.view(B, 1),
                self.null_spk.view(1, -1).to(dtype=spk.dtype, device=spk.device),
                spk,
            )

        # If speaker region masking is enabled, make speaker condition frame-wise.
        # spk: (B, D) -> (B, D, T_mel)
        if spk_cond_mask_ratio > 0.0:
            spk_cond = spk.unsqueeze(-1).expand(-1, -1, x.size(-1))
            spk_cond = self._mask_continuous_region(
                x=spk_cond,
                ratio=spk_cond_mask_ratio,
                null_val=self.null_spk,
                mask=spec_mask,
            )
        else:
            spk_cond = spk

        # 4. Speaker-conditioned temporal smoothing.
        x = x + self.smoothing_net(
            x=x,
            cond=spk_cond,
            mask=spec_mask,
        )

        return x * spec_mask

    def _mask_continuous_region(
        self,
        x: torch.Tensor,
        ratio: float,
        null_val: nn.Parameter,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Masks a continuous region of the input condition.

        Cases:
            x: (B, D, T)
                Masks a continuous time region.
                Used for frame-level text condition or frame-wise speaker condition.

            x: (B, D)
                Masks a continuous channel/dimension region.
                Usually not recommended for unordered global embeddings.

        Args:
            x:
                Condition tensor.
            ratio:
                Mask ratio. If <= 0, no masking.
            null_val:
                Null condition parameter, shape (D,).
            mask:
                Optional time mask for 3D x, shape (B, 1, T) or (B, T).
        """
        if ratio <= 0.0:
            return x

        B, D = x.shape[0], x.shape[1]

        if null_val.numel() != D:
            raise ValueError(f"null_val has {null_val.numel()} elements, but x channel dim is {D}.")

        null_val = null_val.to(
            dtype=x.dtype, device=x.device
        )  # pyright: ignore[reportAssignmentType]

        if x.ndim == 3:
            T = x.shape[2]

            if mask is not None:
                if mask.dim() == 3:
                    mask_2d = mask.squeeze(1)
                elif mask.dim() == 2:
                    mask_2d = mask
                else:
                    raise ValueError(
                        f"mask must have shape (B, 1, T) or (B, T), got {tuple(mask.shape)}"
                    )

                lengths = mask_2d.to(device=x.device).sum(dim=1).long()
            else:
                lengths = torch.full((B,), T, device=x.device, dtype=torch.long)

            mask_lens = (lengths.float() * ratio).long()

            # If ratio > 0 and length > 0, mask at least one frame.
            mask_lens = torch.where(
                (mask_lens <= 0) & (lengths > 0),
                torch.ones_like(mask_lens),
                mask_lens,
            )
            mask_lens = torch.minimum(mask_lens, lengths)

            random_offsets = torch.rand(B, device=x.device)
            start_indices = (random_offsets * (lengths - mask_lens).clamp(min=0)).long()

            indices = torch.arange(T, device=x.device).view(1, T)
            region_mask = (indices >= start_indices.view(B, 1)) & (
                indices < (start_indices + mask_lens).view(B, 1)
            )
            region_mask = region_mask.unsqueeze(1)  # (B, 1, T)

            return torch.where(
                region_mask,
                null_val.view(1, D, 1),
                x,
            )

        if x.ndim == 2:
            mask_len = int(D * ratio)

            if mask_len <= 0:
                return x

            if mask_len >= D:
                return null_val.view(1, D).expand(B, -1)

            start_idx = torch.randint(
                low=0,
                high=D - mask_len + 1,
                size=(B,),
                device=x.device,
            )

            indices = torch.arange(D, device=x.device).view(1, D)
            region_mask = (indices >= start_idx.view(B, 1)) & (
                indices < (start_idx + mask_len).view(B, 1)
            )

            return torch.where(
                region_mask,
                null_val.view(1, D),
                x,
            )

        raise ValueError(f"x must have shape (B, D, T) or (B, D), got {tuple(x.shape)}")
