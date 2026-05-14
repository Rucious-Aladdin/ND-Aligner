from typing import Any, override

import torch
import torch.nn as nn


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs, **kwargs) x module_factor + inputs x input_factor)

    [Update]: Now supports passing **kwargs (like mask, cond) to the inner module.
    """

    def __init__(
        self,
        module: nn.Module,
        module_factor: float = 1.0,
        input_factor: float = 1.0,
    ):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    @override
    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return (self.module(inputs, **kwargs) * self.module_factor) + (inputs * self.input_factor)
