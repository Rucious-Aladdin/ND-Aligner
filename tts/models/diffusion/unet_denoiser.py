from typing import override

import torch
import torch.nn as nn

from .submodules.unet_backbone import UnetBackbone
from .abc.denoiser import Denoiser


class UnetDenoiserNetwork(Denoiser):
    """
    Wrapper around UnetBackbone for mel-space denoising.

    Responsibilities:
      1) Normalize tensor shapes / masks
      2) Apply continuous region masking and full condition drop for CFG
      3) Provide multi-condition guided inference (base, independent, sequential)
    """

    def __init__(
        self,
        n_mels: int = 80,
        pho_cond_dim: int = 192,
        dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        groups: int = 8,
        spk_emb_dim: int = 192,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.pho_cond_dim = pho_cond_dim

        self.text_proj = (
            nn.Conv1d(pho_cond_dim, n_mels, 1) if pho_cond_dim != n_mels else nn.Identity()
        )

        self.backbone = UnetBackbone(
            dim=dim,
            dim_mults=dim_mults,
            groups=groups,
            spk_emb_dim=spk_emb_dim,
        )

        # Null tokens for CFG
        self.null_spk = nn.Parameter(torch.randn(spk_emb_dim))
        self.null_text = nn.Parameter(torch.randn(pho_cond_dim))

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy mel, shape (B, n_mels, T).
            mask: Time mask, shape (B, 1, T) or (B, T).
            text: Conditioning tensor, shape (B, D_cond, T).
            t: Diffusion time / sigma, shape (B,) or scalar.
            spk: Speaker embedding, shape (B, spk_emb_dim).
        """
        B = x.shape[0]

        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
        elif t.ndim == 0:
            t = t.expand(B)

        # Apply masking during training
        if self.training:
            # Full Dropout logic
            if text_cond_drop_prob > 0:
                drop_mask = torch.rand(B, device=x.device) < text_cond_drop_prob
                text = torch.where(drop_mask.view(B, 1, 1), self.null_text.view(1, -1, 1), text)

            if spk_cond_drop_prob > 0:
                drop_mask = torch.rand(B, device=x.device) < spk_cond_drop_prob
                spk = torch.where(drop_mask.view(B, 1), self.null_spk.view(1, -1), spk)

            # Continuous Region Masking
            # Pass mask to handle varying sequence lengths in the batch
            text = self._mask_continuous_region(
                text, text_cond_mask_ratio, self.null_text, mask=mask
            )
            spk = self._mask_continuous_region(spk, spk_cond_mask_ratio, self.null_spk)

        text = self.text_proj(text * mask)

        return self.backbone(
            x=x,
            mask=mask,
            text=text,
            t=t,
            spk=spk,
        )

    @override
    @torch.no_grad()
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
    ) -> torch.Tensor:
        """
        Multi-condition CFG implementation.
        Strategies:
            'base': standard CFG with one scale for all conditions.
            'independent': S_null + Σ scale_i * (S_cond_i - S_null)
            'sequential': Hierarchical CFG (S_∅ -> S_text -> S_text+spk)

        guidance_scale: (text_cond_strength, speaker_cond_strength)
        """
        if isinstance(guidance_scale, (float, int)):
            guidance_scale = (float(guidance_scale), float(guidance_scale))

        # Check if any guidance is needed
        if all(s == 1.0 for s in guidance_scale) and cfg_mode == "base":
            return self.forward(
                x=x,
                mask=mask,
                text=text,
                t=t,
                spk=spk,
            )

        if cfg_mode == "base":
            return self._forward_cfg_base(x, mask, text, t, spk, guidance_scale[0])
        elif cfg_mode == "independent":
            return self._forward_cfg_independent(x, mask, text, t, spk, guidance_scale)
        elif cfg_mode == "sequential":
            return self._forward_cfg_sequential(x, mask, text, t, spk, guidance_scale)
        else:
            raise ValueError(f"Unknown CFG mode: {cfg_mode}")

    def _forward_cfg_base(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        scale: float,
    ):
        B = x.shape[0]
        # Null branch: both dropped
        text_null = self.null_text.view(1, -1, 1).expand(B, -1, -1)
        spk_null = self.null_spk.view(1, -1).expand(B, -1)

        pred_null = self.forward(x, mask, text_null, t, spk_null)
        pred_cond = self.forward(x, mask, text, t, spk)

        return pred_null + scale * (pred_cond - pred_null)

    def _forward_cfg_independent(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        scales: tuple[float, float],
    ):
        """Result = Null + scale_text * (Text - Null) + scale_spk * (Spk - Null)"""
        B = x.shape[0]
        scale_text, scale_spk = scales

        # All Null
        text_null = self.null_text.view(1, -1, 1).expand(B, -1, -1)
        spk_null = self.null_spk.view(1, -1).expand(B, -1)
        pred_null = self.forward(x, mask, text_null, t, spk_null)

        logits = pred_null.clone()

        if scale_text != 0.0:
            # S(text, null)
            pred_text_only = self.forward(x, mask, text, t, spk_null)
            logits = logits + scale_text * (pred_text_only - pred_null)

        if scale_spk != 0.0:
            # S(null, spk)
            pred_spk_only = self.forward(x, mask, text_null, t, spk)
            logits = logits + scale_spk * (pred_spk_only - pred_null)

        return logits

    def _forward_cfg_sequential(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        scales: tuple[float, float],
    ):
        """Result = S(∅) + scale_text * (S(text) - S(∅)) + scale_spk * (S(text, spk) - S(text))"""
        B = x.shape[0]
        scale_text, scale_spk = scales

        # Step 0: Base Null
        text_null = self.null_text.view(1, -1, 1).expand(B, -1, -1)
        spk_null = self.null_spk.view(1, -1).expand(B, -1)
        pred_null = self.forward(x, mask, text_null, t, spk_null)

        final_logits = pred_null.clone()

        # Step 1: Text
        pred_text_only = self.forward(x, mask, text, t, spk_null)
        if scale_text != 0.0:
            final_logits = final_logits + scale_text * (pred_text_only - pred_null)

        # Step 2: Text + Spk
        pred_full = self.forward(x, mask, text, t, spk)
        if scale_spk != 0.0:
            final_logits = final_logits + scale_spk * (pred_full - pred_text_only)

        return final_logits

    def _mask_continuous_region(
        self,
        x: torch.Tensor,
        ratio: float,
        null_val: nn.Parameter,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Masks a continuous region of the input tensor along its last dimension.
        Handles varying lengths in a batch if mask is provided.
        """
        if ratio <= 0:
            return x

        B, D = x.shape[0], x.shape[1]

        if x.ndim == 3:  # (B, D, T)
            T = x.shape[2]

            if mask is not None:
                # Calculate actual length for each sample in batch
                # mask: (B, 1, T) or (B, T)
                lengths = mask.view(B, T).sum(dim=1).long()
            else:
                lengths = torch.full((B,), T, device=x.device, dtype=torch.long)

            # Calculate mask length for each sample based on its actual length
            mask_lens = (lengths * ratio).long()

            # Ensure we don't have zero-length masking if ratio > 0
            mask_lens = torch.where(
                (mask_lens <= 0) & (lengths > 0), torch.ones_like(mask_lens), mask_lens
            )

            # Generate random start points within valid lengths: [0, lengths - mask_lens]
            # torch.randint doesn't support a tensor of 'high' values, so we use rand and scale.
            random_offsets = torch.rand(B, device=x.device)
            start_indices = (random_offsets * (lengths - mask_lens).clamp(min=0)).long()

            # Create boolean mask for the regions to drop
            indices = torch.arange(T, device=x.device).view(1, T)  # (1, T)
            m = (indices >= start_indices.view(B, 1)) & (
                indices < (start_indices + mask_lens).view(B, 1)
            )
            m = m.unsqueeze(1)  # (B, 1, T)

            return torch.where(m, null_val.view(1, D, 1), x)
        else:  # (B, D)
            # For speaker embedding, we treat D as the dimension to mask
            mask_len = int(D * ratio)
            if mask_len <= 0:
                return x
            if mask_len >= D:
                return null_val.view(1, D).expand(B, -1)

            start_idx = torch.randint(0, D - mask_len + 1, (B,), device=x.device)
            indices = torch.arange(D, device=x.device).view(1, D)
            mask = (indices >= start_idx.view(B, 1)) & (indices < (start_idx + mask_len).view(B, 1))

            return torch.where(mask, null_val.view(1, D), x)
