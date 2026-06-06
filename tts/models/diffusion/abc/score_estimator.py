from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Any, override

import torch
import torch.nn as nn


class ScoreEstimator(nn.Module, abc.ABC):
    """
    Abstract base class for diffusion/score estimators.

    A ScoreEstimator wraps a denoiser network and defines:
        - forward diffusion/noising process
        - reverse denoising/preconditioned prediction
        - noise-dependent coefficients
        - training loss weights
    """

    @override
    @abstractmethod
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
        Usually delegates to reverse_diffusion.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_coefficients(
        self,
        sigma: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns noise-level-dependent preconditioning coefficients.

        Common EDM return:
            c_skip, c_out, c_in, c_noise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_loss_weight(
        self,
        sigma: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Returns the loss weighting term for a given noise level.
        """
        raise NotImplementedError()

    @abstractmethod
    def forward_diffusion(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to clean data.

        Args:
            x: clean input, usually (B, n_feats, T)
            sigma: optional noise level, (B,)

        Returns:
            x_noisy: noisy input
            sigma: noise level used
        """
        raise NotImplementedError()

    @abstractmethod
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
        Denoises noisy input.

        Args:
            x: noisy input, usually (B, n_feats, T)
            sigma: noise level, (B,)
            mask: valid-frame mask, usually (B, 1, T)
            text: frame-level text condition, usually (B, C_text, T)
            spk: speaker condition, usually (B, C_spk)
            guidance_scale: CFG scale
            cfg_mode: CFG mode
            text_cond_drop_prob: text condition dropout probability
            spk_cond_drop_prob: speaker condition dropout probability
            text_cond_mask_ratio: text condition masking ratio
            spk_cond_mask_ratio: speaker condition masking ratio

        Returns:
            denoised output, usually (B, n_feats, T)
        """
        raise NotImplementedError()
