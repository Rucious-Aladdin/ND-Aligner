from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from .submodules.conformer_backbone import ConformerBackbone
from .submodules.cond_adapter import ConditionAdapter
from .abc.denoiser import Denoiser, DenoiserNetwork


class ConformerDenoiserNetwork(DenoiserNetwork):
    """
    Pure Conformer denoising network.

    This class does NOT perform:
      - condition dropout
      - condition region masking
      - CFG branch construction
      - ConditionAdapter calls

    It consumes:
        x:    noisy mel, (B, n_mels, T)
        cond: ConditionAdapter output, (B, cond_dim, T)
        mask: valid-frame mask, (B, 1, T) or (B, T)
        t:    diffusion noise-level, (B,) or scalar tensor

    Important:
        cond is NOT fused into x before the backbone.
        cond is passed to ConformerBackbone's condition path.
    """

    def __init__(
        self,
        n_mels: int = 80,
        conformer_cond_dim: int = 256,
        conformer_hidden_dim: int = 256,
        conformer_num_layers: int = 6,
        conformer_num_attention_heads: int = 4,
        conformer_feed_forward_expansion_factor: int = 4,
        conformer_conv_expansion_factor: int = 2,
        conformer_input_dropout_p: float = 0.1,
        conformer_feed_forward_dropout_p: float = 0.1,
        conformer_attention_dropout_p: float = 0.1,
        conformer_conv_dropout_p: float = 0.1,
        conformer_conv_kernel_size: int = 31,
        conformer_half_step_residual: bool = True,
        conformer_attn_window_size: int = 3,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.cond_dim = conformer_cond_dim
        self.encoder_dim = conformer_hidden_dim

        self.x_proj = nn.Conv1d(
            in_channels=n_mels,
            out_channels=conformer_hidden_dim,
            kernel_size=1,
        )

        self.backbone = ConformerBackbone(
            input_dim=conformer_hidden_dim,
            encoder_dim=conformer_hidden_dim,
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
            cond_in_channels=conformer_cond_dim,
            attn_window_size=conformer_attn_window_size,
        )

        self.out_proj = nn.Linear(conformer_hidden_dim, n_mels)

    @override
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Noisy mel, (B, n_mels, T).
            t:
                Diffusion noise-level/time, (B,).
            cond:
                ConditionAdapter output, (B, cond_dim, T).
            mask:
                Valid-frame mask, (B, 1, T) or (B, T).

        Returns:
            Denoising prediction, (B, n_mels, T).
        """
        B = x.size(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        mask = mask.to(dtype=x.dtype, device=x.device)

        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)
            if t.dim() == 0:
                t = t.expand(B)
            else:
                t = t.view(B)

        cond = cond.to(dtype=x.dtype, device=x.device)

        # Only noisy mel is projected into the backbone input.
        # ConditionAdapter output stays as condition.
        h = self.x_proj(x * mask) * mask  # (B, H, T)
        h_seq = h.transpose(1, 2).contiguous()  # (B, T, H)

        # Frame-level cond for ConformerBackbone condition path.
        cond_seq = cond.transpose(1, 2).contiguous()  # (B, T, C_cond)

        mask_2d = mask.squeeze(1).bool()  # (B, T)

        h_out = self.backbone(
            x=h_seq,
            t=t,
            cond=cond_seq,
            mask=mask_2d,
        )

        out = self.out_proj(h_out)  # (B, T, n_mels)
        out = out.transpose(1, 2).contiguous()  # (B, n_mels, T)

        return out * mask


class ConformerDenoiser(Denoiser):
    """
    Multi-condition CFG wrapper for ConformerDenoiserNetwork.

    This class owns:
      - ConditionAdapter
      - ConformerDenoiserNetwork
      - CFG branch construction

    ConditionAdapter is mandatory. Therefore h_text, hard_gamma, and text_mask
    are explicit required keyword-only inputs.
    """

    def __init__(
        self,
        network: ConformerDenoiserNetwork,
        cond_adapter: ConditionAdapter,
    ) -> None:
        super().__init__()

        self.network = network
        self.cond_adapter = cond_adapter

        # Cache is valid only within the same utterance / same condition set.
        # Clear it at the beginning of each sampling call.
        self._cond_cache: dict[str, torch.Tensor] = {}

    @override
    def clear_condition_cache(self) -> None:
        self._cond_cache.clear()

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        cond = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=text_cond_drop_prob,
            spk_cond_drop_prob=spk_cond_drop_prob,
            text_cond_mask_ratio=text_cond_mask_ratio,
            spk_cond_mask_ratio=spk_cond_mask_ratio,
            cache_key=None,
        )

        return self.network(
            x=x,
            t=self._normalize_t(t, x),
            cond=cond,
            mask=mask,
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
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
    ) -> torch.Tensor:
        if isinstance(guidance_scale, (float, int)):
            guidance_scale = (float(guidance_scale), float(guidance_scale))

        if all(s == 1.0 for s in guidance_scale) and cfg_mode == "base":
            return self.forward(
                x=x,
                mask=mask,
                text=text,
                t=t,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
            )

        if cfg_mode == "base":
            if isinstance(guidance_scale, float):
                scale = guidance_scale
            else:
                scale = guidance_scale[0]

            return self._forward_cfg_base(
                x=x,
                mask=mask,
                text=text,
                t=t,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
                scale=scale,
            )

        if cfg_mode == "independent":
            return self._forward_cfg_independent(
                x=x,
                mask=mask,
                text=text,
                t=t,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
                scales=guidance_scale,
            )

        if cfg_mode == "sequential":
            return self._forward_cfg_sequential(
                x=x,
                mask=mask,
                text=text,
                t=t,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
                scales=guidance_scale,
            )

        raise ValueError(f"Unknown CFG mode: {cfg_mode}")

    def _forward_cfg_base(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        t = self._normalize_t(t, x)

        cond_null = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=1.0,
            spk_cond_drop_prob=1.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="null",
        )

        cond_full = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=0.0,
            spk_cond_drop_prob=0.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="full",
        )

        pred_null = self.network(x=x, t=t, cond=cond_null, mask=mask)
        pred_full = self.network(x=x, t=t, cond=cond_full, mask=mask)

        return pred_null + scale * (pred_full - pred_null)

    def _forward_cfg_independent(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        scales: tuple[float, float],
    ) -> torch.Tensor:
        t = self._normalize_t(t, x)
        scale_text, scale_spk = scales

        cond_null = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=1.0,
            spk_cond_drop_prob=1.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="null",
        )

        pred_null = self.network(x=x, t=t, cond=cond_null, mask=mask)
        out = pred_null.clone()

        if scale_text != 0.0:
            cond_text_only = self._build_condition(
                text=text,
                mask=mask,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
                text_cond_drop_prob=0.0,
                spk_cond_drop_prob=1.0,
                text_cond_mask_ratio=0.0,
                spk_cond_mask_ratio=0.0,
                cache_key="text_only",
            )

            pred_text_only = self.network(
                x=x,
                t=t,
                cond=cond_text_only,
                mask=mask,
            )

            out = out + scale_text * (pred_text_only - pred_null)

        if scale_spk != 0.0:
            cond_spk_only = self._build_condition(
                text=text,
                mask=mask,
                spk=spk,
                h_text=h_text,
                hard_gamma=hard_gamma,
                text_mask=text_mask,
                text_cond_drop_prob=1.0,
                spk_cond_drop_prob=0.0,
                text_cond_mask_ratio=0.0,
                spk_cond_mask_ratio=0.0,
                cache_key="spk_only",
            )

            pred_spk_only = self.network(
                x=x,
                t=t,
                cond=cond_spk_only,
                mask=mask,
            )

            out = out + scale_spk * (pred_spk_only - pred_null)

        return out

    def _forward_cfg_sequential(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor | float,
        spk: torch.Tensor,
        *,
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        scales: tuple[float, float],
    ) -> torch.Tensor:
        t = self._normalize_t(t, x)
        scale_text, scale_spk = scales

        cond_null = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=1.0,
            spk_cond_drop_prob=1.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="null",
        )

        cond_text_only = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=0.0,
            spk_cond_drop_prob=1.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="text_only",
        )

        cond_full = self._build_condition(
            text=text,
            mask=mask,
            spk=spk,
            h_text=h_text,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            text_cond_drop_prob=0.0,
            spk_cond_drop_prob=0.0,
            text_cond_mask_ratio=0.0,
            spk_cond_mask_ratio=0.0,
            cache_key="full",
        )

        pred_null = self.network(x=x, t=t, cond=cond_null, mask=mask)
        pred_text_only = self.network(x=x, t=t, cond=cond_text_only, mask=mask)
        pred_full = self.network(x=x, t=t, cond=cond_full, mask=mask)

        out = pred_null.clone()

        if scale_text != 0.0:
            out = out + scale_text * (pred_text_only - pred_null)

        if scale_spk != 0.0:
            out = out + scale_spk * (pred_full - pred_text_only)

        return out

    def _build_condition(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
        spk: torch.Tensor,
        *,
        h_text: torch.Tensor,
        hard_gamma: torch.Tensor,
        text_mask: torch.Tensor,
        text_cond_drop_prob: float,
        spk_cond_drop_prob: float,
        text_cond_mask_ratio: float,
        spk_cond_mask_ratio: float,
        cache_key: str | None,
    ) -> torch.Tensor:
        if cache_key is not None and cache_key in self._cond_cache:
            return self._cond_cache[cache_key]

        cond = self.cond_adapter(
            h_text=h_text,
            aligned_feats=text,
            spk=spk,
            hard_gamma=hard_gamma,
            text_mask=text_mask,
            spec_mask=mask,
            text_cond_drop_prob=text_cond_drop_prob,
            spk_cond_drop_prob=spk_cond_drop_prob,
            text_cond_mask_ratio=text_cond_mask_ratio,
            spk_cond_mask_ratio=spk_cond_mask_ratio,
        )

        if cache_key is not None:
            self._cond_cache[cache_key] = cond

        return cond

    def _normalize_t(
        self,
        t: torch.Tensor | float,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        B = ref.size(0)

        if not torch.is_tensor(t):
            return torch.full(
                (B,),
                float(t),
                device=ref.device,
                dtype=ref.dtype,
            )

        t = t.to(device=ref.device, dtype=ref.dtype)

        if t.dim() == 0:
            return t.expand(B)

        return t.view(B)
