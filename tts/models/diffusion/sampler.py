from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .score_estimator import KarrasScoreEstimator


class KarrasSampler(nn.Module):
    """
    Karras (EDM) sampler that supports both deterministic (Heun) and stochastic sampling.
    """

    def __init__(self, estimator: KarrasScoreEstimator) -> None:
        super().__init__()
        self.estimator = estimator

    def get_sigmas(
        self,
        n_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float = 7.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Polynomial schedule (EDM Eq. 5 / Eq. 269).
        """
        step_indices = torch.arange(n_steps, dtype=torch.float32, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Add t_N = 0
        return t_steps

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
        spk: torch.Tensor,
        n_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
        stochastic: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Main entry point for sampling.

        Args:
            mu: Conditioning tensor (aligned_feats).
            mask: Mel mask.
            spk: Speaker embedding.
            n_steps: Number of sampling steps.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            rho: Polynomial schedule exponent.
            guidance_scale: Classifier-free guidance scale (float or tuple).
            cfg_mode: CFG strategy ('base', 'independent', 'sequential').
            stochastic: Whether to use stochastic sampling (Algorithm 2).
            **kwargs: Extra parameters for stochastic sampling (s_churn, s_tmin, s_tmax, s_noise).
        """
        if stochastic:
            return self._sample_stochastic(
                text=text,
                mask=mask,
                spk=spk,
                n_steps=n_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                guidance_scale=guidance_scale,
                cfg_mode=cfg_mode,
                **kwargs,
            )
        else:
            return self._sample_deterministic(
                text=text,
                mask=mask,
                spk=spk,
                n_steps=n_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                guidance_scale=guidance_scale,
                cfg_mode=cfg_mode,
            )

    def _sample_deterministic(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
        spk: torch.Tensor,
        n_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        guidance_scale: float | tuple[float, float],
        cfg_mode: str,
    ) -> torch.Tensor:
        B, _, T = text.shape
        device = text.device

        # 1. Initialize x_0 ~ N(mu_data, sigma_max^2 * I)
        x = (
            torch.randn([B, self.estimator.denoiser.n_mels, T], device=device) * sigma_max
            + self.estimator.mu_data
        )
        sigmas = self.get_sigmas(n_steps, sigma_min, sigma_max, rho, device)

        for i in range(n_steps):
            t_curr, t_next = sigmas[i], sigmas[i + 1]
            t_curr_b = t_curr.expand(B)

            # ODE Step: d_i = (x - D(x; t)) / t
            denoised = self.estimator.reverse_diffusion(
                x, t_curr_b, mask, text, spk, guidance_scale, cfg_mode
            )
            d_i = (x - denoised) / t_curr
            x_next = x + (t_next - t_curr) * d_i

            if t_next != 0:
                denoised_next = self.estimator.reverse_diffusion(
                    x_next, t_next.expand(B), mask, text, spk, guidance_scale, cfg_mode
                )
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - t_curr) * 0.5 * (d_i + d_prime)

            x = x_next
        return x

    def _sample_stochastic(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
        spk: torch.Tensor,
        n_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        guidance_scale: float | tuple[float, float],
        cfg_mode: str,
        s_churn: float = 40.0,
        s_tmin: float = 0.05,
        s_tmax: float = 50.0,
        s_noise: float = 1.003,
    ) -> torch.Tensor:
        B, _, T = text.shape
        device = text.device

        x = (
            torch.randn([B, self.estimator.denoiser.n_mels, T], device=device) * sigma_max
            + self.estimator.mu_data
        )
        sigmas = self.get_sigmas(n_steps, sigma_min, sigma_max, rho, device)

        for i in range(n_steps):
            t_curr, t_next = sigmas[i], sigmas[i + 1]

            # --- Churn ---
            gamma = min(s_churn / n_steps, 2**0.5 - 1) if s_tmin <= t_curr <= s_tmax else 0.0
            hat_t = t_curr + gamma * t_curr

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + (hat_t**2 - t_curr**2).sqrt() * eps

            # --- Heun Step from hat_t to t_next ---
            hat_t_b = hat_t.expand(B)
            denoised = self.estimator.reverse_diffusion(
                x, hat_t_b, mask, text, spk, guidance_scale, cfg_mode
            )
            d_i = (x - denoised) / hat_t
            x_next = x + (t_next - hat_t) * d_i

            if t_next != 0:
                denoised_next = self.estimator.reverse_diffusion(
                    x_next, t_next.expand(B), mask, text, spk, guidance_scale, cfg_mode
                )
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - hat_t) * 0.5 * (d_i + d_prime)

            x = x_next
        return x
