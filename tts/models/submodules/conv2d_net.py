from typing import override

import torch
import torch.nn as nn


def valid_group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        groups: int = 8,
    ):
        super().__init__()

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("kernel_size must contain odd values.")

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.GroupNorm(
            num_groups=valid_group_count(dim_out, groups),
            num_channels=dim_out,
        )
        self.act = nn.GELU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Conv2dNet(nn.Module):
    """
    Plain Conv2D network over pairwise speech-text feature.

    Input : (B, C_in, T_s, T_t)
    Output: (B, C_out, T_s, T_t)

    This preserves both T_s and T_t lengths.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 16,
        dim_out: int = 1,
        num_layers: int = 3,
        kernel_size: tuple[int, int] = (3, 3),
        groups: int = 8,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []

        layers.append(
            ConvNormAct2d(
                dim_in=dim_in,
                dim_out=dim_hidden,
                kernel_size=kernel_size,
                groups=groups,
            )
        )

        for _ in range(num_layers - 1):
            layers.append(
                ConvNormAct2d(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    kernel_size=kernel_size,
                    groups=groups,
                )
            )

        layers.append(
            nn.Conv2d(
                dim_hidden,
                dim_out,
                kernel_size=1,
            )
        )

        self.net = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
