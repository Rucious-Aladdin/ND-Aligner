from __future__ import annotations

from typing import override, Any

import torch

from .abc.denoiser import Denoiser
from .abc.score_estimator import ScoreEstimator
from .abc.sampler import Sampler


class KarrasDiffusionModel(ScoreEstimator, Sampler):
    """
    Karras (EDM) score estimator and sampler.
        This class implements both the ScoreEstimator and Sampler interfaces, so it can be used
        directly for training and sampling without needing a separate Sampler wrapper.
        The forward method implements the reverse diffusion (denoising) step, and the sample method
        implements the Karras sampling loop.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        n_mels: int = 80,
        sigma_data: float = 2.0990,  # empirically estimated
        mu_data: float = -4.9307,  # empirically estimated
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser

        self.n_mels = n_mels
        self.sigma_data = sigma_data
        self.mu_data = mu_data
        self.p_mean = p_mean
        self.p_std = p_std

    # (x -> noisy_x) at nois-level=sigma
    @override
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

    # (x_noisy -> x0_hat) at noise-level=sigma
    # maybe used for training time (explicitly puts drop_prob and mask_ratio)
    # always guidance_scale=1.0 for training time
    @override
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        spk: torch.Tensor,
        *,
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Default forward pass delegates to reverse_diffusion.
        """

        x_normed, x_centered, c_noise = self.__normalize(x, sigma)

        denoise_centered = self.denoiser(
            x=x_normed,
            mask=mask,
            text=text,
            t=c_noise,
            spk=spk,
            text_cond_drop_prob=text_cond_drop_prob,
            spk_cond_drop_prob=spk_cond_drop_prob,
            text_cond_mask_ratio=text_cond_mask_ratio,
            spk_cond_mask_ratio=spk_cond_mask_ratio,
            **kwargs,
        )
        return self.__denormalize(x_centered, denoise_centered, sigma)

    # (x_noisy -> x0_hat) at noise-level=sigma
    # inferecne time forward pass with CFG
    @override
    def reverse_diffusion(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
        text: torch.Tensor,
        spk: torch.Tensor,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Denoises the input using EDM preconditioning and centering.
        (EDM Reverse Path: D_theta(x; sigma))

        Args:
            guidance_scale: (text_cond_strength, speaker_cond_strength)
        """
        x_normed, x_centered, c_noise = self.__normalize(x, sigma)

        f_theta = self.denoiser.forward_with_cfg(
            x=x_normed,
            mask=mask,
            text=text,
            t=c_noise,
            spk=spk,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            **kwargs,
        )

        return self.__denormalize(x_centered, f_theta, sigma)

    @override
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
            text: Conditioning tensor (aligned_feats).
            mask: Mel mask.
            spk: Speaker embedding.
            n_steps: Number of sampling steps.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            rho: Polynomial schedule exponent.
            guidance_scale: Classifier-free guidance scale (text_cond_strength, speaker_cond_strength) or float.
            cfg_mode: CFG strategy ('base', 'independent', 'sequential').
            stochastic: Whether to use stochastic sampling (Algorithm 2).
            **kwargs: Extra parameters for stochastic sampling (s_churn, s_tmin, s_tmax, s_noise).
        """
        if stochastic:
            gen_x0 = self._sample_stochastic(
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
            gen_x0 = self._sample_deterministic(
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

        self.denoiser.clear_condition_cache()
        return gen_x0

    @override
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

    @override
    def get_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Computes the effective loss weight lambda(sigma).
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    @override
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
        **kwargs: Any,
    ) -> torch.Tensor:
        B, _, T = text.shape
        device = text.device

        # 1. Initialize x_0 ~ N(mu_data, sigma_max^2 * I)
        x = torch.randn([B, self.n_mels, T], device=device) * sigma_max + self.mu_data
        sigmas = self.get_sigmas(n_steps, sigma_min, sigma_max, rho, device)

        for i in range(n_steps):
            t_curr, t_next = sigmas[i], sigmas[i + 1]
            t_curr_b = t_curr.expand(B)

            # ODE Step: d_i = (x - D(x; t)) / t
            denoised = self.reverse_diffusion(
                x=x,
                sigma=t_curr_b,
                mask=mask,
                text=text,
                spk=spk,
                guidance_scale=guidance_scale,
                cfg_mode=cfg_mode,
                **kwargs,
            )
            d_i = (x - denoised) / t_curr
            x_next = x + (t_next - t_curr) * d_i

            if t_next != 0:
                denoised_next = self.reverse_diffusion(
                    x=x_next,
                    sigma=t_next.expand(B),
                    mask=mask,
                    text=text,
                    spk=spk,
                    guidance_scale=guidance_scale,
                    cfg_mode=cfg_mode,
                    **kwargs,
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
        **kwargs: Any,
    ) -> torch.Tensor:
        B, _, T = text.shape
        device = text.device

        x = torch.randn([B, self.n_mels, T], device=device) * sigma_max + self.mu_data
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
            denoised = self.reverse_diffusion(
                x=x,
                sigma=hat_t_b,
                mask=mask,
                text=text,
                spk=spk,
                guidance_scale=guidance_scale,
                cfg_mode=cfg_mode,
                **kwargs,
            )
            d_i = (x - denoised) / hat_t
            x_next = x + (t_next - hat_t) * d_i

            if t_next != 0:
                denoised_next = self.reverse_diffusion(
                    x=x_next,
                    sigma=t_next.expand(B),
                    mask=mask,
                    text=text,
                    spk=spk,
                    guidance_scale=guidance_scale,
                    cfg_mode=cfg_mode,
                    **kwargs,
                )
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - hat_t) * 0.5 * (d_i + d_prime)

            x = x_next
        return x

    def __denormalize(
        self,
        x_centered: torch.Tensor,
        f_theta: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, _, _ = self.get_coefficients(sigma)

        c_skip = c_skip.view(-1, 1, 1)
        c_out = c_out.view(-1, 1, 1)

        denoised_centered = c_skip * x_centered + c_out * f_theta
        return denoised_centered + self.mu_data

    def __normalize(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_centered = x - self.mu_data
        _, _, c_in, c_noise = self.get_coefficients(sigma)
        return x_centered * c_in.view(-1, 1, 1), x_centered, c_noise
