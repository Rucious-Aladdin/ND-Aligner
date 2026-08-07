import math
from typing import override

import torch
import torch.nn as nn


class PositionalEncoding1d(nn.Module):
    def __init__(self, channels: int, max_len: int = 4096):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))

        self.pe: torch.Tensor
        pe = torch.zeros(max_len, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.T.unsqueeze(0))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        return x + self.pe[:, :, : x.size(2)]
