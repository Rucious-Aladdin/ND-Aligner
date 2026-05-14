from __future__ import annotations

from typing import Any, override

import torch
import torch.nn as nn

from .denoiser_net import DenoiserNetwork


class KarrasScoreEstimator(nn.Module):
    """
    Implements EDM (Karras et al. 2022) preconditioning and Classifier-Free Guidance.
    Handles data centering (mu_data) and variance scaling (sigma_data).
    """

    def __init__(
        self,
        denoiser: DenoiserNetwork,
        sigma_data: float = 2.0990,
        mu_data: float = -4.9307,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        self.mu_data = mu_data
        self.p_mean = p_mean
        self.p_std = p_std

    def get_coefficients(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes EDM preconditioning coefficients.
        """
        sigma_sq = sigma**2
        sigma_data_sq = self.sigma_data**2

        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.sigma_data / (sigma_sq + sigma_data_sq).sqrt()
        c_in = 1 / (sigma_sq + sigma_data_sq).sqrt()
        c_noise = 0.25 * sigma.log()

        return c_skip, c_out, c_in, c_noise

    def get_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Computes the effective loss weight lambda(sigma).
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def forward_diffusion(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the input according to the given sigma.
        If sigma is None, it is sampled using log-normal distribution (for training).

        Returns:
            y_noisy: Noisy input (B, n_mels, T)
            sigma: Noise levels used (B,)
        """
        if sigma is None:
            rnd_normal = torch.randn([x.size(0)], device=x.device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        y_noisy = x + torch.randn_like(x) * sigma.view(-1, 1, 1)
        return y_noisy, sigma

    def reverse_diffusion(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        spk: torch.Tensor,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Denoises the input using EDM preconditioning and centering.
        (EDM Reverse Path: D_theta(x; sigma))

        Args:
            guidance_scale: (text_cond_strength, speaker_cond_strength)
        """
        x_centered = x - self.mu_data
        c_skip, c_out, c_in, c_noise = self.get_coefficients(sigma)

        c_skip = c_skip.view(-1, 1, 1)
        c_out = c_out.view(-1, 1, 1)
        c_in = c_in.view(-1, 1, 1)

        # Determine if we should use CFG
        is_cfg = False
        if isinstance(guidance_scale, (float, int)):
            if guidance_scale != 1.0:
                is_cfg = True

        elif isinstance(guidance_scale, (tuple, list)):  # type: ignore
            if any(s != 1.0 for s in guidance_scale) or cfg_mode != "base":
                is_cfg = True

        # Apply abstracted denoiser
        if not is_cfg:
            f_theta = self.denoiser(
                x=x_centered * c_in,
                mask=mask,
                text=text,
                t=c_noise,
                spk=spk,
                text_cond_drop_prob=text_cond_drop_prob,
                spk_cond_drop_prob=spk_cond_drop_prob,
                text_cond_mask_ratio=text_cond_mask_ratio,
                spk_cond_mask_ratio=spk_cond_mask_ratio,
            )
        else:
            # We assume the denoiser provides forward_with_cfg
            f_theta = self.denoiser.forward_with_cfg(
                x=x_centered * c_in,
                mask=mask,
                text=text,
                t=c_noise,
                spk=spk,
                guidance_scale=guidance_scale,
                cfg_mode=cfg_mode,
            )

        denoised_centered = c_skip * x_centered + c_out * f_theta
        return denoised_centered + self.mu_data

    @override
    def forward(
        self,
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Default forward pass delegates to reverse_diffusion.
        """
        return self.reverse_diffusion(*args, **kwargs)
