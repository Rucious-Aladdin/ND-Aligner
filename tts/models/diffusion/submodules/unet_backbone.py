from collections.abc import Callable
from typing import Any, override

import numpy as np
import torch
from einops import rearrange

from tts.models.layers.film_layer import FiLMLayer
from tts.models.layers.fourier_embedding import GaussianFourierProjection


class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def nparams(self) -> int:
        num_params: int = 0
        for param in self.parameters():
            if param.requires_grad:
                num_params += int(np.prod(param.shape))
        return num_params

    def relocate_input(self, x: list[Any]) -> list[Any]:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return x

        for i in range(len(x)):
            item = x[i]
            if isinstance(item, torch.Tensor) and item.device != device:
                x[i] = item.to(device)

        return x


class Mish(BaseModule):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.conv = torch.nn.ConvTranspose2d(dim, dim_out, 4, 2, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.conv = torch.nn.Conv2d(dim, dim_out, 3, 2, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) * self.g


class FiLMBlock(BaseModule):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        cond_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.film = FiLMLayer(dim_out, cond_dim)
        self.act = Mish()

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(x * mask)
        h = self.norm(h)
        h = self.film(h, cond)
        h = self.act(h)
        return h * mask


class ResnetBlock(BaseModule):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        cond_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.block1 = FiLMBlock(dim, dim_out, cond_dim=cond_dim, groups=groups)
        self.block2 = FiLMBlock(dim_out, dim_out, cond_dim=cond_dim, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.block1(x, mask, cond)
        h = self.block2(h, mask, cond)
        return h + self.res_conv(x * mask)


class LinearAttention(BaseModule):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape  # pyright: ignore[reportUnusedVariable]
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out,
            "b heads c (h w) -> b (heads c) h w",
            heads=self.heads,
            h=h,
            w=w,
        )
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    @override
    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class UnetBackbone(BaseModule):
    def __init__(
        self,
        dim: int,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        groups: int = 8,
        spk_emb_dim: int = 64,
        n_feats: int = 80,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats

        # time embedding -> dim
        self.time_pos_emb = GaussianFourierProjection(dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            Mish(),
            torch.nn.Linear(dim * 4, dim),
        )

        # speaker embedding -> dim
        self.spk_mlp = torch.nn.Sequential(
            torch.nn.Linear(spk_emb_dim, dim * 4),
            Mish(),
            torch.nn.Linear(dim * 4, dim),
        )

        # text global embedding -> dim
        self.text_mlp = torch.nn.Sequential(
            torch.nn.Conv1d(n_feats, dim * 4, 1),
            Mish(),
            torch.nn.Conv1d(dim * 4, n_feats, 1),
        )

        # final conditioning dim = [time_emb ; spk_emb]
        cond_dim = dim * 2

        # input channels are only [mu, x]
        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, cond_dim=cond_dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, cond_dim=cond_dim, groups=groups),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, groups=groups)

        # mirror original Grad-TTS style:
        # up path uses reversed(in_out[1:]) so final hidden channel returns to `dim`
        for dim_in, dim_out in reversed(in_out[1:]):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, cond_dim=cond_dim, groups=groups),
                        ResnetBlock(dim_in, dim_in, cond_dim=cond_dim, groups=groups),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_block = FiLMBlock(dim, dim, cond_dim=cond_dim, groups=groups)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    @override
    def forward(
        self,
        x: torch.Tensor,  # [B, n_feats, T]
        mask: torch.Tensor,  # [B, 1, T]
        text: torch.Tensor,  # [B, n_feats, T]
        t: torch.Tensor,  # [B]
        spk: torch.Tensor,  # [B, spk_emb_dim]
    ) -> torch.Tensor:
        # embeddings
        t = self.time_pos_emb(t)
        t = self.time_mlp(t)
        s = self.spk_mlp(spk)

        cond = torch.cat([t, s], dim=1)  # [B, 3 * dim]

        # input: only [text, x]
        text = self.text_mlp(text) * mask
        x = torch.stack([text, x], dim=1)  # [B, 2, n_feats, T]
        mask = mask.unsqueeze(1)  # [B, 1, 1, T]

        hiddens: list[torch.Tensor] = []
        masks = [mask]

        for (
            resnet1,
            resnet2,
            attn,
            downsample,
        ) in self.downs:  # pyright:ignore[reportGeneralTypeIssues]
            mask_down = masks[-1]
            x = resnet1(x, mask_down, cond)
            x = resnet2(x, mask_down, cond)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        # remove the extra mask appended after the final stage
        masks = masks[:-1]

        x = self.mid_block1(x, masks[-1], cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, masks[-1], cond)

        for resnet1, resnet2, attn, upsample in self.ups:  # pyright:ignore[reportGeneralTypeIssues]
            mask_up = masks.pop()
            h = hiddens.pop()

            x = torch.cat((x, h), dim=1)
            x = resnet1(x, mask_up, cond)
            x = resnet2(x, mask_up, cond)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, masks.pop(), cond)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
