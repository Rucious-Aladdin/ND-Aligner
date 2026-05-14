from typing import Any, override

import torch
import torch.nn as nn


class Flip(nn.Module):
    @override
    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        reverse: bool = False,
        **kwargs: Any,
    ):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x
