import math
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.models.utils.positional_encoding import PositionalEncoding


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
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
            num_groups=_valid_group_count(dim_out, groups),
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
    ):
        super().__init__()
        self.block1 = ConvNormAct2d(dim_in, dim_out, groups=groups)
        self.block2 = ConvNormAct2d(dim_out, dim_out, groups=groups)

        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.res_conv(x)


class Downsample(nn.Module):
    """
    Downsample only along speech/time axis.

    Input : (B, C, T_s, T_t)
    Output: (B, C_out, ceil(T_s / 2), T_t)

    실제 forward에서는 T_s를 4의 배수로 pad하므로 두 번 downsample 후 shape mismatch가 줄어든다.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample only along speech/time axis.
    Token axis T_t is preserved.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

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
        return self.conv(x)


class UNet2d(nn.Module):
    """
    U-Net over pairwise score feature.

    - input : (B, H, T_s, T_t)
    - output: (B, 1, T_s, T_t)

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

        self.in_block = ResBlock2d(dim_in, d0, groups=groups)

        self.down1 = nn.Sequential(
            Downsample(d0, d1),
            ResBlock2d(d1, d1, groups=groups),
        )

        self.down2 = nn.Sequential(
            Downsample(d1, d2),
            ResBlock2d(d2, d2, groups=groups),
        )

        self.mid = nn.Sequential(
            ResBlock2d(d2, d2, groups=groups),
            ResBlock2d(d2, d2, groups=groups),
        )

        self.up1 = Upsample(d2, d1)
        self.dec1 = ResBlock2d(d1 + d1, d1, groups=groups)

        self.up2 = Upsample(d1, d0)
        self.dec2 = ResBlock2d(d0 + d0, d0, groups=groups)

        self.out_block = nn.Sequential(
            ConvNormAct2d(d0, d0, groups=groups),
            nn.Conv2d(d0, 1, kernel_size=1),
        )

    @staticmethod
    def _pad_speech_to_multiple(
        x: torch.Tensor,
        multiple: int = 4,
    ) -> tuple[torch.Tensor, int]:
        """
        T_s가 4의 배수가 아니어도 두 번 downsample/upsample 후 원래 길이로 crop 가능하게 pad.
        """
        T_s = x.size(-2)
        pad_s = (multiple - (T_s % multiple)) % multiple

        if pad_s > 0:
            # F.pad order for 4D is (left, right, top, bottom)
            # 여기서 top/bottom은 T_s axis.
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

        h0 = self.in_block(x)  # (B, d0, T_s, T_t)
        h1 = self.down1(h0)  # (B, d1, T_s/2, T_t)
        h2 = self.down2(h1)  # (B, d2, T_s/4, T_t)

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


class UNetTransitionPredictor(nn.Module):
    """
    U-Net transition predictor.

    Pairwise input:
        [
            f_s(z_spec_t) + PE_t ;
            f_x(mu_text_j) + PE_j ;
            detached log_emission[t, j] ;
            cond ;
            t/spec_len ;
            j/text_len ;
            t/spec_len - j/text_len ;
            spec_len/text_len
        ]
    """

    def __init__(
        self,
        dim_latent: int,
        dim_cond: int,
        latent_channels: int = 64,
        cond_channels: int = 32,
        base_dim: int = 16,
        groups: int = 8,
        transition_scale: float = -3.0,
        pos_enc_max_len: int = 10000,
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.cond_channels = cond_channels

        self.spec_proj = nn.Sequential(
            nn.Linear(dim_latent, dim_latent * 2),
            nn.LayerNorm(dim_latent * 2),
            nn.GELU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(dim_latent, dim_latent * 2),
            nn.LayerNorm(dim_latent * 2),
            nn.GELU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )

        self.spec_pos_enc = PositionalEncoding(
            d_model=dim_latent,
            max_len=pos_enc_max_len,
        )
        self.text_pos_enc = PositionalEncoding(
            d_model=dim_latent,
            max_len=pos_enc_max_len,
        )

        self.cond_proj = nn.Linear(dim_cond, cond_channels)

        # [spec_feat ; text_feat ; log_emission ; cond ; progress(4)]
        self.unet = UNet2d(
            dim_in=2 * dim_latent + 1 + cond_channels + 4,
            base_dim=base_dim,
            groups=groups,
        )

        self.transition_scale = nn.Parameter(torch.tensor(float(transition_scale)))

    @staticmethod
    def _make_progress_features(
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        T_s: int,
        T_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns:
            progress: (B, T_s, T_t, 4)

        Channels:
            t_pos        = t / (spec_lengths[b] - 1)
            j_pos        = j / (text_lengths[b] - 1)
            diag         = t_pos - j_pos
            length_ratio = spec_lengths[b] / text_lengths[b]
        """
        B = spec_lengths.size(0)

        t = torch.arange(T_s, device=device, dtype=dtype).view(1, T_s, 1, 1)
        j = torch.arange(T_t, device=device, dtype=dtype).view(1, 1, T_t, 1)

        spec_len = spec_lengths.to(device=device, dtype=dtype).view(B, 1, 1, 1)
        text_len = text_lengths.to(device=device, dtype=dtype).view(B, 1, 1, 1)

        t_pos = t / (spec_len - 1.0).clamp_min(1.0)
        j_pos = j / (text_len - 1.0).clamp_min(1.0)
        diag = t_pos - j_pos

        length_ratio = (spec_len / text_len.clamp_min(1.0)).expand(B, T_s, T_t, 1)

        t_pos = t_pos.expand(B, T_s, T_t, 1)
        j_pos = j_pos.expand(B, T_s, T_t, 1)
        diag = diag.expand(B, T_s, T_t, 1)

        return torch.cat([t_pos, j_pos, diag, length_ratio], dim=-1)

    @override
    def forward(
        self,
        z_spec: torch.Tensor,  # (B, C, T_s)
        mu_text: torch.Tensor,  # (B, C, T_t)
        log_emission: torch.Tensor,  # (B, T_s, T_t)
        spec_lengths: torch.Tensor,  # (B,)
        text_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
    ) -> torch.Tensor:
        """
        Returns:
            advance_delta: (B, T_s, T_t)
        """
        B, C, T_s = z_spec.shape
        _, _, T_t = mu_text.shape

        z_spec_t = z_spec.transpose(1, 2).contiguous()  # (B, T_s, C)
        mu_text_t = mu_text.transpose(1, 2).contiguous()  # (B, T_t, C)

        # First project into transition feature space.
        z_spec_t = self.spec_proj(z_spec_t)
        mu_text_t = self.text_proj(mu_text_t)

        # Then add positional encoding.
        z_spec_t = self.spec_pos_enc(z_spec_t)
        mu_text_t = self.text_pos_enc(mu_text_t)

        z_pair = z_spec_t.unsqueeze(2).expand(B, T_s, T_t, C)
        mu_pair = mu_text_t.unsqueeze(1).expand(B, T_s, T_t, C)

        # Use emission landscape as a detached hint.
        emission_pair = log_emission.detach().unsqueeze(-1)

        cond_feat = self.cond_proj(cond)
        cond_pair = cond_feat.view(B, 1, 1, self.cond_channels).expand(
            B,
            T_s,
            T_t,
            self.cond_channels,
        )

        progress = self._make_progress_features(
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
            T_s=T_s,
            T_t=T_t,
            device=z_spec.device,
            dtype=z_spec.dtype,
        )

        pairwise = torch.cat(
            [z_pair, mu_pair, emission_pair, cond_pair, progress],
            dim=-1,
        )  # (B, T_s, T_t, 2C + 1 + cond_channels + 4)

        pairwise = pairwise.permute(0, 3, 1, 2).contiguous()

        advance_delta = self.unet(pairwise).squeeze(1)
        advance_delta = advance_delta * self.transition_scale.exp()

        return advance_delta
