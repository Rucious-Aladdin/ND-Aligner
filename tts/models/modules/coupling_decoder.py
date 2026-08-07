from typing import NamedTuple, override

import torch
import torch.nn as nn

from tts.models.modules.layers.film_blocks import FiLMResidualBlock
from tts.models.modules.utils.pos_encoding import PositionalEncoding1d


class CouplingDecoderOutput(NamedTuple):
    mel_outputs: list[torch.Tensor]  # each: (B, T_mel, n_mels)
    mel_hat: torch.Tensor  # (B, T_mel, n_mels)
    mel_losses: list[torch.Tensor]  # raw per-stage L1 losses
    loss: torch.Tensor  # decay-weighted scalar loss


class FiLMRefineBlock(nn.Module):
    """
    A refinement block with effective temporal receptive field = kernel_size.

    Structure:
        1x1 input projection:  in_channels -> hidden_channels
        1x1 SpecDecoderBlock:  hidden_channels -> hidden_channels
        kx1 SpecDecoderBlock:  hidden_channels -> hidden_channels
        1x1 SpecDecoderBlock:  hidden_channels -> hidden_channels
        1x1 output projection: hidden_channels -> out_channels

    Only the middle kx1 SpecDecoderBlock expands temporal receptive field.
    The input/output projections and 1x1 blocks do not increase temporal RF.

    Args:
        in_channels: Input channel dimension.
        out_channels: Output channel dimension.
        hidden_channels: Internal channel dimension.
        cond_dim: Conditioning dimension.
        kernel_size: Temporal kernel size of the middle block.
        dilation: Dilation of the middle temporal block.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        cond_dim: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size should be odd to preserve symmetric temporal context, got {kernel_size}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.in_proj = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=1,
                    dilation=1,
                    dropout=dropout,
                ),
                FiLMResidualBlock(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                ),
                FiLMResidualBlock(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=1,
                    dilation=1,
                    dropout=dropout,
                ),
            ]
        )

        self.out_proj = nn.Conv1d(
            hidden_channels,
            out_channels,
            kernel_size=1,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, T)
            cond: (B, D_cond)
            mask: (B, T), True = valid position.

        Returns:
            x: (B, C_out, T)
        """
        if mask is not None:
            if mask.shape != (x.shape[0], x.shape[2]):
                raise ValueError(
                    f"Expected mask shape {(x.shape[0], x.shape[2])}, got {tuple(mask.shape)}."
                )

        x = self.in_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, cond)

            if mask is not None:
                x = x * mask.unsqueeze(1).to(dtype=x.dtype)

        x = self.out_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)

        return x


class CouplingDecoder(nn.Module):
    """
    Progressive coupling decoder with shared refinement block and shared mel head.

    Stage structure:
        stage 0:
            stem_refine: in_channels -> hidden_channels, RF=1
            mel_head

        stage k >= 1:
            shared refine: hidden_channels -> hidden_channels, RF += kernel_size - 1
            mel_head

    Effective RF when dilation=1:
        stage 0: RF = 1
        stage 1: RF = kernel_size
        stage 2: RF = 1 + 2 * (kernel_size - 1)
        ...

    Args:
        in_channels: Input aligned feature dimension.
        out_channels: Output mel dimension.
        hidden_channels: Decoder hidden dimension.
        cond_dim: External conditioning vector dimension.
        kernel_size: Temporal kernel size of shared refinement block.
        num_refinement_steps: Number of refinement applications after stem.
        dilation: Dilation of the middle temporal conv in FiLMRefineBlock.
        dropout: Dropout probability.
        cond_proj_dim: Projected cond dimension before concat with step embedding.
        step_emb_dim: Refinement step embedding dimension.
        loss_weight: Base reconstruction loss weight.
        loss_decay: Geometric decay factor for progressive losses.
        normalize_loss_weights: If True, stage weights sum to loss_weight.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        cond_dim: int,
        cond_proj_dim: int,
        kernel_size: int,
        num_refinement_steps: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        step_emb_dim: int = 64,
        loss_decay_factor: float = 0.5,
        normalize_loss_weights: bool = True,
    ):
        super().__init__()

        if num_refinement_steps < 0:
            raise ValueError(f"num_refinement_steps must be >= 0, got {num_refinement_steps}.")

        if loss_decay_factor < 0:
            raise ValueError(f"loss_decay must be >= 0, got {loss_decay_factor}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.num_refinement_steps = num_refinement_steps
        self.dilation = dilation
        self.loss_decay_factor = loss_decay_factor
        self.normalize_loss_weights = normalize_loss_weights
        self.cond_proj_dim = cond_proj_dim
        self.step_emb_dim = step_emb_dim
        self.refine_cond_dim = cond_proj_dim + step_emb_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim),
            nn.GELU(),
            nn.LayerNorm(cond_proj_dim),
        )

        self.pe = PositionalEncoding1d(channels=hidden_channels)

        # step_idx = 0 for stem, 1..num_refinement_steps for refinement.
        self.step_emb = nn.Embedding(num_refinement_steps + 1, step_emb_dim)

        self.in_proj = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        # RF=1 stem: in_channels -> hidden_channels.
        self.stem_refine = FiLMRefineBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            cond_dim=self.refine_cond_dim,
            kernel_size=1,
            dilation=1,
            dropout=dropout,
        )

        # Shared refinement: hidden_channels -> hidden_channels.
        self.refine = FiLMRefineBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            cond_dim=self.refine_cond_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )

        # Shared mel head. No cond, no step embedding.
        self.mel_head = nn.Linear(hidden_channels, out_channels)

    @override
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> CouplingDecoderOutput:
        """
        Args:
            x: Aligned features (B, T_mel, C)
            cond: Conditioning vector (B, D_cond)
            mask: Spec mask (B, T_mel), True = valid position.
            target: Target mel-spectrogram (B, T_mel, n_mels)

        Returns:
            CouplingDecoderOutput
        """
        if target is None:
            raise ValueError(
                "target must be provided in forward() to compute mel_losses and loss. "
                + "Use inference() when target is unavailable."
            )

        mel_outputs = self._decode(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=self.num_refinement_steps,
        )

        mel_losses = [
            self._compute_l1_loss(
                pred=mel,
                target=target,
                mask=mask,
            )
            for mel in mel_outputs
        ]

        weights = torch.tensor(
            [self.loss_decay_factor**i for i in range(len(mel_losses))],
            device=mel_losses[0].device,
            dtype=mel_losses[0].dtype,
        )
        if self.normalize_loss_weights:
            weights = weights / weights.sum().clamp_min(1e-8)
        loss = torch.sum(weights * torch.stack(mel_losses))

        return CouplingDecoderOutput(
            mel_outputs=mel_outputs,
            mel_hat=mel_outputs[-1],
            mel_losses=mel_losses,
            loss=loss,
        )

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
        num_refinement_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Aligned features (B, T_mel, C)
            cond: Conditioning vector (B, D_cond)
            mask: Spec mask (B, T_mel), True = valid position.
            num_refinement_steps:
                None -> use all refinement steps.
                int  -> use only the given number of refinement steps.

        Returns:
            mel_hat: (B, T_mel, n_mels)
        """
        mel_outputs = self._decode(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=num_refinement_steps,
        )

        return mel_outputs[-1]

    def _decode(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None,
        num_refinement_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            x: (B, T_mel, C)
            cond: (B, D_cond)
            mask: (B, T_mel), True = valid position.
            num_refinement_steps: number of refinement steps after stem.

        Returns:
            mel_outputs: list of (B, T_mel, n_mels)
        """
        b, t, _c = x.shape

        if mask is not None:
            if mask.shape != (b, t):
                raise ValueError(f"Expected mask shape {(b, t)}, got {tuple(mask.shape)}.")

        if num_refinement_steps is None:
            num_refinement_steps = self.num_refinement_steps

        if num_refinement_steps < 0 or num_refinement_steps > self.num_refinement_steps:
            raise ValueError(
                "num_refinement_steps must be in "
                + f"[0, {self.num_refinement_steps}], got {num_refinement_steps}."
            )

        # (B, T, C) -> (B, C, T)
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        h = self.pe(h)
        if mask is not None:
            h = h * mask.unsqueeze(1).to(dtype=h.dtype)

        mel_outputs: list[torch.Tensor] = []

        # init_global_condition
        cond_p = self.cond_proj(cond)

        # Stage 0 condition
        step = torch.full(
            (cond.shape[0],),
            0,
            device=cond.device,
            dtype=torch.long,
        )
        step_p = self.step_emb(step)
        stage_cond = torch.cat([cond_p, step_p], dim=-1)

        # Stage 0: Stem Refinment...
        h = self.stem_refine(h, stage_cond, mask=mask)
        mel_outputs.append(self.mel_head(h.transpose(1, 2)))

        # Stage 1..K: repeated shared residual refinement.
        for step_idx in range(1, num_refinement_steps + 1):
            step = torch.full(
                (cond.shape[0],),
                step_idx,
                device=cond.device,
                dtype=torch.long,
            )
            step_p = self.step_emb(step)
            stage_cond = torch.cat([cond_p, step_p], dim=-1)

            h = h + self.refine(h, stage_cond, mask=mask)

            if mask is not None:
                h = h * mask.unsqueeze(1).to(dtype=h.dtype)

            mel_outputs.append(self.mel_head(h.transpose(1, 2)))

        return mel_outputs

    def _compute_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, T, n_mels)
            target: (B, T, n_mels)
            mask: (B, T), True = valid position.

        Returns:
            scalar raw L1 loss.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}."
            )

        diff = torch.abs(pred - target)

        if mask is None:
            return diff.mean()

        if mask.shape != pred.shape[:2]:
            raise ValueError(f"Expected mask shape {pred.shape[:2]}, got {tuple(mask.shape)}.")

        mask_f = mask.unsqueeze(-1).to(dtype=diff.dtype)
        denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]

        return (diff * mask_f).sum() / denom


if __name__ == "__main__":
    import resource

    torch.manual_seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def print_memory(tag: str) -> None:
        # Linux 기준: ru_maxrss 단위는 KB.
        cpu_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        msg = f"[MEM] {tag} | CPU peak RSS: {cpu_peak_rss_mb:.2f} MB"

        if torch.cuda.is_available():
            torch.cuda.synchronize()

            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
            max_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

            msg += (
                f" | CUDA allocated: {allocated_mb:.2f} MB"
                f" | CUDA reserved: {reserved_mb:.2f} MB"
                f" | CUDA max allocated: {max_allocated_mb:.2f} MB"
                f" | CUDA max reserved: {max_reserved_mb:.2f} MB"
            )

        print(msg)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print_memory("start")

    batch_size = 4
    t_mel = 800

    in_channels = 256
    out_channels = 80
    hidden_channels = 384
    cond_dim = 192
    cond_proj_dim = 128

    num_refinement_steps = 3
    kernel_size = 5
    dilation = 1
    dropout = 0.1
    loss_decay_factor = 0.5

    decoder = CouplingDecoder(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        cond_dim=cond_dim,
        cond_proj_dim=cond_proj_dim,
        kernel_size=kernel_size,
        num_refinement_steps=num_refinement_steps,
        dilation=dilation,
        dropout=dropout,
        step_emb_dim=64,
        loss_decay_factor=loss_decay_factor,
        normalize_loss_weights=True,
    ).to(device)

    decoder.train()

    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    param_memory_mb = sum(p.numel() * p.element_size() for p in decoder.parameters()) / 1024**2

    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    print(f"parameter memory: {param_memory_mb:.2f} MB")

    print_memory("after decoder init")

    x = torch.randn(batch_size, t_mel, in_channels, device=device)
    cond = torch.randn(batch_size, cond_dim, device=device)
    target = torch.randn(batch_size, t_mel, out_channels, device=device)

    mask = torch.ones(batch_size, t_mel, dtype=torch.bool, device=device)
    mask[0, -10:] = False
    mask[1, -20:] = False

    print_memory("after input allocation")

    out = decoder(
        x=x,
        cond=cond,
        mask=mask,
        target=target,
    )

    print_memory("after forward")

    print("num mel outputs:", len(out.mel_outputs))
    print("mel_hat shape:", tuple(out.mel_hat.shape))
    print("raw losses:", [float(v.detach().cpu()) for v in out.mel_losses])
    print("weighted loss:", float(out.loss.detach().cpu()))

    assert len(out.mel_outputs) == num_refinement_steps + 1

    for mel in out.mel_outputs:
        assert mel.shape == (batch_size, t_mel, out_channels)

    assert out.mel_hat.shape == (batch_size, t_mel, out_channels)
    assert out.mel_hat is out.mel_outputs[-1]
    assert len(out.mel_losses) == num_refinement_steps + 1
    assert torch.isfinite(out.loss)

    raw_weights = torch.tensor(
        [loss_decay_factor**i for i in range(num_refinement_steps + 1)],
        device=device,
        dtype=out.loss.dtype,
    )
    expected_weights = raw_weights / raw_weights.sum().clamp_min(1e-8)
    expected_loss = torch.sum(expected_weights * torch.stack(out.mel_losses))

    print("loss weights:", expected_weights.detach().cpu().tolist())

    assert torch.allclose(out.loss, expected_loss, atol=1e-6), (
        float(out.loss.detach().cpu()),
        float(expected_loss.detach().cpu()),
    )

    out.loss.backward()

    print_memory("after backward")

    assert decoder.mel_head.weight.grad is not None
    assert torch.isfinite(decoder.mel_head.weight.grad).all()

    assert decoder.refine.out_proj.weight.grad is not None
    assert torch.isfinite(decoder.refine.out_proj.weight.grad).all()

    assert decoder.stem_refine.out_proj.weight.grad is not None
    assert torch.isfinite(decoder.stem_refine.out_proj.weight.grad).all()

    print("backward ok")

    decoder.eval()

    with torch.no_grad():
        mel_full = decoder.inference(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=None,
        )
        mel_0 = decoder.inference(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=0,
        )
        mel_1 = decoder.inference(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=1,
        )

    print_memory("after inference")

    print("mel_full shape:", tuple(mel_full.shape))
    print("mel_0 shape:", tuple(mel_0.shape))
    print("mel_1 shape:", tuple(mel_1.shape))

    assert mel_full.shape == (batch_size, t_mel, out_channels)
    assert mel_0.shape == (batch_size, t_mel, out_channels)
    assert mel_1.shape == (batch_size, t_mel, out_channels)

    try:
        decoder.inference(
            x=x,
            cond=cond,
            mask=mask,
            num_refinement_steps=num_refinement_steps + 1,
        )
        raise AssertionError("Expected ValueError for invalid num_refinement_steps.")
    except ValueError:
        print("invalid num_refinement_steps check ok")

    try:
        decoder(
            x=x,
            cond=cond,
            mask=mask,
            target=None,
        )
        raise AssertionError("Expected ValueError when target is None.")
    except ValueError:
        print("target None check ok")

    print("all tests passed")
