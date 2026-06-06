from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


@staticmethod
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
        padding: tuple[int, int] = (1, 1),
        groups: int = 8,
    ):
        super().__init__()
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


class ResBlock2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        groups: int = 8,
        kernel_size: tuple[int, int] = (3, 3),
        padding: tuple[int, int] = (1, 1),
    ):
        super().__init__()

        self.block1 = ConvNormAct2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.block2 = ConvNormAct2d(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )

        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.res_conv(x)


class SequentialAxisConvNormAct2d(nn.Module):
    """
    Replaces Conv2d(kernel_size=(3, 1), padding=(1, 0))
    with Conv1d over the speech axis for each text position independently.

    Input : (B, C_in, T_s, T_t)
    Output: (B, C_out, T_s, T_t)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 8,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
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
        B, C, T_s, T_t = x.shape

        # (B, C, T_s, T_t) -> (B*T_t, C, T_s)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_t, C, T_s)

        x = self.conv(x)

        C_out = x.size(1)
        T_s_out = x.size(2)

        # (B*T_t, C_out, T_s) -> (B, C_out, T_s, T_t)
        x = x.view(B, T_t, C_out, T_s_out)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        x = self.act(x)

        return x


class SequentialAxisResBlock2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        groups: int = 8,
        kernel_size: tuple[int, int] = (3, 1),
        padding: tuple[int, int] = (1, 0),
    ):
        super().__init__()

        assert kernel_size[1] == 1
        assert padding[1] == 0

        self.block1 = SequentialAxisConvNormAct2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size[0],
            padding=padding[0],
            groups=groups,
        )
        self.block2 = SequentialAxisConvNormAct2d(
            dim_out,
            dim_out,
            kernel_size=kernel_size[0],
            padding=padding[0],
            groups=groups,
        )

        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.res_conv(x)


class SequentialAxisDownsample(nn.Module):
    """
    Replaces Conv2d(kernel_size=(3,1), stride=(2,1), padding=(1,0)).
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_in,
            dim_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T_s, T_t = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_t, C, T_s)

        x = self.conv(x)

        C_out = x.size(1)
        T_s_out = x.size(2)

        x = x.view(B, T_t, C_out, T_s_out)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class SequentialAxisUpsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_in,
            dim_out,
            kernel_size=3,
            padding=1,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        target_speech_len: int,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(target_speech_len, x.size(-1)),
            mode="nearest",
        )

        B, C, T_s, T_t = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_t, C, T_s)

        x = self.conv(x)

        C_out = x.size(1)
        T_s_out = x.size(2)

        x = x.view(B, T_t, C_out, T_s_out)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class UNet2d(nn.Module):
    """
    U-Net over pairwise speech-text feature.

    Input : (B, H, T_s, T_t)
    Output: (B, 1, T_s, T_t)

    Downsampling is applied only along T_s:
        T_s -> T_s / 2 -> T_s / 4

    T_t is never downsampled.
    """

    def __init__(
        self,
        dim_in: int,
        base_dim: int = 16,
        groups: int = 8,
    ):
        super().__init__()

        d0 = base_dim
        d1 = base_dim * 2
        d2 = base_dim * 2

        # Text RF contribution: +4
        self.in_block = ResBlock2d(
            dim_in,
            d0,
            groups=groups,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.down1 = nn.Sequential(
            SequentialAxisDownsample(d0, d1),
            SequentialAxisResBlock2d(
                dim_in=d1,
                dim_out=d1,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.down2 = nn.Sequential(
            SequentialAxisDownsample(d1, d2),
            SequentialAxisResBlock2d(
                dim_in=d2,
                dim_out=d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.mid = nn.Sequential(
            SequentialAxisResBlock2d(
                dim_in=d2,
                dim_out=d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            SequentialAxisResBlock2d(
                dim_in=d2,
                dim_out=d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.up1 = SequentialAxisUpsample(d2, d1)
        self.dec1 = SequentialAxisResBlock2d(
            dim_in=d1 + d1,
            dim_out=d1,
            groups=groups,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

        self.up2 = SequentialAxisUpsample(d1, d0)
        self.dec2 = SequentialAxisResBlock2d(
            dim_in=d0 + d0,
            dim_out=d0,
            groups=groups,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

        # Text RF contribution: +2
        # Total text RF = 1 + 4 + 2 = 7 tokens.
        self.out_block = nn.Sequential(
            ConvNormAct2d(
                d0,
                d0,
                kernel_size=(3, 3),
                padding=(1, 1),
                groups=groups,
            ),
            nn.Conv2d(d0, 1, kernel_size=1),
        )

    @staticmethod
    def _pad_speech_to_multiple(
        x: torch.Tensor,
        multiple: int = 4,
    ) -> tuple[torch.Tensor, int]:
        T_s = x.size(-2)
        pad_s = (multiple - (T_s % multiple)) % multiple

        if pad_s > 0:
            # F.pad order for 4D is (left, right, top, bottom).
            # Here top/bottom correspond to the T_s axis.
            x = F.pad(x, (0, 0, 0, pad_s))

        return x, pad_s

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: pairwise feature, (B, H, T_s, T_t)

        Returns:
            score: (B, 1, T_s, T_t)
        """
        orig_T_s = x.size(-2)

        x, pad_s = self._pad_speech_to_multiple(x, multiple=4)

        h0 = self.in_block(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)

        x = self.mid(h2)

        x = self.up1(x, target_speech_len=h1.size(-2))
        x = torch.cat([x, h1], dim=1)
        x = self.dec1(x)

        x = self.up2(x, target_speech_len=h0.size(-2))
        x = torch.cat([x, h0], dim=1)
        x = self.dec2(x)

        x = self.out_block(x)

        if pad_s > 0:
            x = x[:, :, :orig_T_s, :]

        return x
