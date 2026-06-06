from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Sampler(nn.Module, abc.ABC):
    """
    Abstract base class for diffusion samplers.

    Public interface:
        - get_sigmas: construct the noise schedule
        - sample: generate samples from conditioning inputs
    """

    @abstractmethod
    def get_sigmas(
        self,
        n_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float = 7.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Construct a noise-level schedule.

        Returns:
            sigmas: (n_steps + 1,), usually ending with zero.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abstractmethod
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
        Generate a sample from the diffusion model.

        Args:
            text:
                Conditioning tensor. In your current setup this is usually
                raw aligned_feats or adapted condition depending on denoiser design.
            mask:
                Valid-frame mask, usually (B, 1, T).
            spk:
                Speaker condition, usually (B, spk_dim).
            n_steps:
                Number of reverse diffusion steps.
            sigma_min:
                Minimum noise level.
            sigma_max:
                Maximum noise level.
            rho:
                Karras schedule curvature.
            guidance_scale:
                CFG scale. Can be scalar or (text_scale, spk_scale).
            cfg_mode:
                CFG strategy, e.g. "base", "independent", "sequential".
            stochastic:
                Whether to use stochastic sampling.
            **kwargs:
                Extra sampler / denoiser-specific inputs.
                For ConformerDenoiser this should carry h_text, hard_gamma, text_mask, etc.

        Returns:
            Generated mel/sample tensor, usually (B, n_mels, T).
        """
        raise NotImplementedError()
