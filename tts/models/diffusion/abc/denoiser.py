from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Any, override

import torch
import torch.nn as nn


class Denoiser(nn.Module, abc.ABC):
    """
    Abstract base class for denoiser networks used in diffusion.
    """

    @override
    @abstractmethod
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
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
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
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def clear_condition_cache(self) -> None:
        """
        Optional method to clear any cached conditioning information. This is needed for some
        denoiser designs that cache adapted conditions for efficiency, but
        the cache needs to be cleared between sampling steps.
        """
        return


class DenoiserNetwork(nn.Module, abc.ABC):
    """
    Abstract base class for denoiser networks used in diffusion.
    """

    @override
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError()
