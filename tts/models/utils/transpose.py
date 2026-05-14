from typing import override

import torch
import torch.nn as nn


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple[int, ...]):
        super(Transpose, self).__init__()
        self.shape = shape

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)
