from typing import NamedTuple, override

import torch
import torch.nn as nn

from .layers.film_block_conv2d import FiLMResidualConv2D
from .utils.pos_encoding import PositionalEncoding1d
from .utils.sequence_mask import sequence_mask


class CouplingConv2dDecoderOutput(NamedTuple):
    # Logging only.
    # Each: [B, T, D_mel]
    mel_outputs: list[torch.Tensor]

    # Logging only.
    # Each: [B, T, N]
    # L2 norm of candidate mel vectors over D_mel.
    out_norms: list[torch.Tensor]

    # Final logging reconstruction.
    # [B, T, D_mel]
    mel_hat: torch.Tensor

    # Raw per-stage expected reconstruction losses.
    mel_losses: list[torch.Tensor]

    # Decay-weighted reconstruction loss.
    loss: torch.Tensor


class FiLMRefineBlock2D(nn.Module):
    """
    2D counterpart of FiLMRefineBlock.

    Structure:
        1x1 input projection
        -> 1x1 FiLMResidualConv2D
        -> kxk FiLMResidualConv2D
        -> 1x1 FiLMResidualConv2D
        -> 1x1 output projection

    Only the middle kxk block expands the spatial receptive field.

    Input:
        x:    [B, C_in, T, N]
        cond: [B, D_cond]
        mask: [B, T, N]

    Output:
        [B, C_out, T, N]
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
            raise ValueError("kernel_size should be odd, " + f"got {kernel_size}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                FiLMResidualConv2D(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=1,
                    dilation=1,
                    dropout=dropout,
                ),
                FiLMResidualConv2D(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                ),
                FiLMResidualConv2D(
                    channels=hidden_channels,
                    cond_dim=cond_dim,
                    kernel_size=1,
                    dilation=1,
                    dropout=dropout,
                ),
            ]
        )

        self.out_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
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
            x:
                [B, C_in, T, N]

            cond:
                [B, D_cond]

            mask:
                [B, T, N]
                True = valid cell

        Returns:
            [B, C_out, T, N]
        """
        if mask is not None:
            expected_shape = (
                x.shape[0],
                x.shape[2],
                x.shape[3],
            )

            if mask.shape != expected_shape:
                raise ValueError(
                    f"Expected mask shape {expected_shape}, " + f"got {tuple(mask.shape)}."
                )

        x = self.in_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)

        for block in self.blocks:
            x = block(x, cond)

            if mask is not None:
                x = x * mask.unsqueeze(1).to(x.dtype)

        x = self.out_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)

        return x


class CouplingConv2dDecoder(nn.Module):
    """
    Progressive T x N candidate reconstruction decoder.

    Input:
        h_text: [B, C_text, N]
        gamma:  [B, T, N]

    Grid construction:
        h_text
            -> broadcast over T
            -> [B, C_text, T, N]

        gamma
            -> [B, 1, T, N]

        concat
            -> [B, C_text + 1, T, N]

    Progressive structure:

        Stage 0:
            RF = 1x1
            stem_refine:
                1x1 -> 1x1 -> 1x1

        Stage >= 1:
            shared refinement:
                1x1 -> 3x3 -> 1x1

        With dilation=1:
            stage 0 : RF = 1x1
            stage 1 : RF = 3x3
            stage 2 : RF = 5x5
            stage 3 : RF = 7x7
            ...

    gamma is NOT detached.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        cond_dim: int,
        cond_proj_dim: int,
        kernel_size: int,
        num_refinement_steps: int = 6,
        dilation: int = 1,
        dropout: float = 0.1,
        step_emb_dim: int = 64,
        loss_decay_factor: float = 1.0,
        normalize_loss_weights: bool = True,
    ):
        super().__init__()

        if num_refinement_steps < 0:
            raise ValueError("num_refinement_steps must be >= 0, " + f"got {num_refinement_steps}.")

        if loss_decay_factor < 0:
            raise ValueError("loss_decay_factor must be >= 0, " + f"got {loss_decay_factor}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.cond_dim = cond_dim
        self.cond_proj_dim = cond_proj_dim
        self.step_emb_dim = step_emb_dim
        self.refine_cond_dim = cond_proj_dim + step_emb_dim

        self.num_refinement_steps = num_refinement_steps
        self.dilation = dilation

        self.loss_decay_factor = loss_decay_factor
        self.normalize_loss_weights = normalize_loss_weights

        # Global condition
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim),
            nn.GELU(),
            nn.LayerNorm(cond_proj_dim),
        )

        self.step_emb = nn.Embedding(num_refinement_steps + 1, step_emb_dim)

        # [text || gamma] -> hidden
        self.in_proj = nn.Conv2d(
            in_channels=in_channels + 1,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        # Same temporal PE philosophy as original TCD.
        self.pe = PositionalEncoding1d(
            channels=hidden_channels,
        )

        # Stage 0
        # 1x1 -> 1x1 -> 1x1
        self.stem_refine = FiLMRefineBlock2D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            cond_dim=self.refine_cond_dim,
            kernel_size=1,
            dilation=1,
            dropout=dropout,
        )

        # Shared refinement
        # 1x1 -> 3x3 -> 1x1
        self.refine = FiLMRefineBlock2D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            cond_dim=self.refine_cond_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )

        # Shared candidate mel head
        self.mel_head = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    @override
    def forward(
        self,
        h_text: torch.Tensor,
        gamma: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> CouplingConv2dDecoderOutput:

        b, _c, n = h_text.shape
        bg, t, ng = gamma.shape

        if b != bg:
            raise ValueError(f"Batch mismatch: {b} vs {bg}.")

        if n != ng:
            raise ValueError(f"Text length mismatch: {n} vs {ng}.")

        if target.shape != (
            b,
            self.out_channels,
            t,
        ):
            raise ValueError(
                "Expected target shape "
                + f"{(b, self.out_channels, t)}, "
                + f"got {tuple(target.shape)}."
            )

        # Masks
        spec_mask = sequence_mask(spec_lengths, t)
        text_mask = sequence_mask(text_lengths, n)
        grid_mask = spec_mask.unsqueeze(-1) & text_mask.unsqueeze(1)

        # Decode + compute stage-wise reconstruction losses.
        mel_outputs, mel_losses, out_norms = self._decode(
            h_text=h_text,
            gamma=gamma,
            target=target,
            cond=cond,
            spec_mask=spec_mask,
            text_mask=text_mask,
            grid_mask=grid_mask,
            spec_lengths=spec_lengths,
            num_refinement_steps=self.num_refinement_steps,
        )

        weights = torch.tensor(
            [self.loss_decay_factor**i for i in range(len(mel_losses))],
            device=mel_losses[0].device,
            dtype=mel_losses[0].dtype,
        )

        if self.normalize_loss_weights:
            weights = weights / weights.sum().clamp_min(1e-8)

        loss = torch.sum(weights * torch.stack(mel_losses))

        return CouplingConv2dDecoderOutput(
            mel_outputs=mel_outputs,
            out_norms=out_norms,
            mel_hat=mel_outputs[-1],
            mel_losses=mel_losses,
            loss=loss,
        )

    def _decode(
        self,
        *,
        h_text: torch.Tensor,
        gamma: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
        spec_mask: torch.Tensor,
        text_mask: torch.Tensor,
        grid_mask: torch.Tensor,
        spec_lengths: torch.Tensor,
        num_refinement_steps: int | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Returns:
            mel_outputs:
                Logging-only posterior-weighted reconstructions.
                Each tensor: [B, T, D_mel]

            mel_losses:
                Differentiable scalar reconstruction loss
                for each refinement stage.

        Important:
            The full candidate field [B, D_mel, T, N]
            is never stored in mel_outputs.
            It only exists temporarily for loss computation.
        """

        _bg, t, _ng = gamma.shape

        if num_refinement_steps is None:
            num_refinement_steps = self.num_refinement_steps

        if not (0 <= num_refinement_steps <= self.num_refinement_steps):
            raise ValueError(
                "num_refinement_steps must be in "
                + f"[0, {self.num_refinement_steps}], "
                + f"got {num_refinement_steps}."
            )

        # ========================================================
        # Construct T x N grid
        # ========================================================

        # [B, C, N] -> [B, C, 1, N] -> [B, C, T, N]
        text_grid = h_text.unsqueeze(2).expand(-1, -1, t, -1)
        # [B, T, N] -> [B, 1, T, N]
        gamma_grid = gamma.unsqueeze(1).to(h_text.dtype)

        # [B, C_text + 1, T, N]
        h = torch.cat([text_grid, gamma_grid], dim=1)

        # [B, C_hidden, T, N]
        h = self.in_proj(h)

        # Temporal positional encoding.
        h = self._apply_temporal_pe(h)
        grid_mask_f = grid_mask.unsqueeze(1).to(h.dtype)
        h = h * grid_mask_f

        # Global condition
        cond_p = self.cond_proj(cond)
        mel_outputs: list[torch.Tensor] = []
        out_norms: list[torch.Tensor] = []
        mel_losses: list[torch.Tensor] = []

        # Stage 0
        stage_cond = self._make_stage_condition(cond_p, step_idx=0)
        h = self.stem_refine(h, stage_cond, mask=grid_mask)
        h = h * grid_mask_f

        # Full candidate reconstruction field, [B, D, T, N].
        candidate_mel = self.mel_head(h)
        mel_loss = self._compute_expected_l1_loss(
            pred=candidate_mel,
            target=target,
            gamma=gamma,
            spec_mask=spec_mask,
            text_mask=text_mask,
            spec_lengths=spec_lengths,
        )
        mel_losses.append(mel_loss)

        with torch.no_grad():
            # Posterior-weighted mel for logging.
            # [B, T, D]
            mel_output = torch.einsum(
                "btn,bdtn->btd",
                gamma,
                candidate_mel,
            )

            mel_output = mel_output * spec_mask.unsqueeze(-1).to(mel_output.dtype)

            # Candidate-wise output norm.
            # candidate_mel: [B, D, T, N] -> norm over D -> [B, T, N]
            out_norm = torch.linalg.vector_norm(
                candidate_mel.float(),
                ord=2,
                dim=1,
            )

            out_norm = out_norm * grid_mask.to(out_norm.dtype)

        mel_outputs.append(mel_output)
        out_norms.append(out_norm)

        del candidate_mel

        # Stage 1..K
        for step_idx in range(1, num_refinement_steps + 1):
            stage_cond = self._make_stage_condition(cond_p, step_idx=step_idx)

            # Same outer residual structure as original TCD.
            h = h + self.refine(h, stage_cond, mask=grid_mask)
            h = h * grid_mask_f

            # Temporary full candidate field. [B, D, T, N]
            candidate_mel = self.mel_head(h)
            mel_loss = self._compute_expected_l1_loss(
                pred=candidate_mel,
                target=target,
                gamma=gamma,
                spec_mask=spec_mask,
                text_mask=text_mask,
                spec_lengths=spec_lengths,
            )
            mel_losses.append(mel_loss)

            with torch.no_grad():
                mel_output = torch.einsum("btn,bdtn->btd", gamma, candidate_mel)
                mel_output = mel_output * spec_mask.unsqueeze(-1).to(mel_output.dtype)
                out_norm = torch.linalg.vector_norm(candidate_mel.float(), ord=2, dim=1)
                out_norm = out_norm * grid_mask.to(out_norm.dtype)

            mel_outputs.append(mel_output)
            out_norms.append(out_norm)

            del candidate_mel

        return mel_outputs, mel_losses, out_norms

    def _apply_temporal_pe(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, C, T, N]

        Returns:
            [B, C, T, N]
        """

        b, c, t, n = h.shape

        # [B,C,T,N] -> [B,N,C,T] -> [B*N,C,T]
        h = h.permute(0, 3, 1, 2).reshape(b * n, c, t)
        h = self.pe(h)

        # [B*N,C,T] -> [B,N,C,T] -> [B,C,T,N]
        h = h.reshape(b, n, c, t).permute(0, 2, 3, 1)

        return h

    def _make_stage_condition(
        self,
        cond_p: torch.Tensor,
        *,
        step_idx: int,
    ) -> torch.Tensor:
        step = torch.full(
            (cond_p.shape[0],),
            step_idx,
            device=cond_p.device,
            dtype=torch.long,
        )
        step_p = self.step_emb(step)
        return torch.cat([cond_p, step_p], dim=-1)

    def _compute_expected_l1_loss(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: torch.Tensor,
        spec_mask: torch.Tensor,
        text_mask: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred:
            [B, D, T, N]

        target:
            [B, D, T]

        gamma:
            [B, T, N]
        """

        # --------------------------------------------------------
        # Candidate reconstruction cost
        #
        # [B,D,T,N]
        # -> mean over mel dimension
        # -> [B,T,N]
        # --------------------------------------------------------

        cost = torch.abs(pred.float() - target.float().unsqueeze(-1)).mean(dim=1)
        valid_grid = spec_mask.unsqueeze(-1) & text_mask.unsqueeze(1)
        gamma_f = gamma.float() * valid_grid.to(torch.float32)

        # [B,T,N] -> [B,T]
        frame_loss = (gamma_f * cost).sum(dim=-1)
        spec_mask_f = spec_mask.to(frame_loss.dtype)
        loss_per_sample = (frame_loss * spec_mask_f).sum(dim=-1) / spec_lengths.to(
            frame_loss.dtype
        ).clamp_min(1.0)
        return loss_per_sample.mean()
