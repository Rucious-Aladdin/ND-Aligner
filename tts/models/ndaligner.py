from typing import NamedTuple, cast, override

import torch
import torch.nn.functional as F

from tts.models.modules.crf_aligner import MonotonicCRFAligner
from tts.models.modules.hifigan_vocoder import Generator
from tts.models.modules.spec_decoder import SpecDecoder
from tts.models.modules.spec_encoder import SpecEncoder
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder
from tts.models.modules.text_encoder import TextEncoder
from tts.models.utils.sequence_mask import sequence_mask

from .base_model import BaseModel


class AlignerForward(NamedTuple):
    # alignment / duration
    soft_attn: torch.Tensor
    soft_dur: torch.Tensor
    hard_attn: torch.Tensor | None
    hard_dur: torch.Tensor | None

    # text / speech representations
    h_text: torch.Tensor
    h_spec: torch.Tensor

    # forward-backward internals
    log_alpha: torch.Tensor
    log_beta: torch.Tensor
    log_gamma: torch.Tensor
    raw_log_z: torch.Tensor
    norm_log_z: torch.Tensor

    # pairwise CRF scores
    log_b: torch.Tensor
    raw_unary: torch.Tensor
    masked_raw_unary: torch.Tensor

    viterbi_path: torch.Tensor | None
    viterbi_logp: torch.Tensor | None

    # reconstruction
    recon: torch.Tensor

    # losses
    crf_loss: torch.Tensor
    diag_loss: torch.Tensor
    recon_loss: torch.Tensor
    viterbi_kl_loss: torch.Tensor
    viterbi_ot_loss: torch.Tensor


class AlignerFeatures(NamedTuple):
    soft_dur: torch.Tensor
    soft_attn: torch.Tensor

    hard_attn: torch.Tensor | None
    hard_dur: torch.Tensor | None

    h_text: torch.Tensor
    h_spec: torch.Tensor

    # forward-backward internals
    log_alpha: torch.Tensor
    log_beta: torch.Tensor
    log_gamma: torch.Tensor
    raw_log_z: torch.Tensor
    norm_log_z: torch.Tensor

    # pairwise CRF scores
    log_b: torch.Tensor
    raw_unary: torch.Tensor
    masked_raw_unary: torch.Tensor

    viterbi_path: torch.Tensor | None = None
    viterbi_logp: torch.Tensor | None = None


def compute_beta_binomial_loss(
    gamma_posterior: torch.Tensor,  # (B, T_s, T_t)
    y_lengths: torch.Tensor,  # (B,)
    x_lengths: torch.Tensor,  # (B,)
    *,
    omega: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Beta-binomial diagonal prior loss for monotonic alignment.

    Computes:

        L = -(1 / sum_b y_lengths[b]) sum_{b,t,j}
                gamma[b,t,j] * log prior_bb[b,t,j]

    where:

        prior_bb[t, j]
            = BetaBinomial(
                j; n=x_lengths[b]-1,
                alpha=omega * (t + 1),
                beta =omega * (T - t)
              )

    Args:
        gamma_posterior:
            Soft alignment posterior, (B, T_s, T_t).

        y_lengths:
            Valid speech/mel lengths, (B,).

        x_lengths:
            Valid text/token lengths, (B,).

        omega:
            Beta-binomial concentration scale.
            Lower omega => wider prior.
            Higher omega => sharper diagonal prior.

        eps:
            Numerical epsilon.

        detach_gamma:
            If True, the loss does not backprop through gamma_posterior.

    Returns:
        Scalar loss.
    """
    if gamma_posterior.dim() != 3:
        raise ValueError(
            f"gamma_posterior must have shape (B, T_s, T_t), "
            + f"got {tuple(gamma_posterior.shape)}."
        )

    B, T_s, T_t = gamma_posterior.shape
    device = gamma_posterior.device
    dtype = gamma_posterior.dtype

    y_lengths = y_lengths.to(device=device).long()
    x_lengths = x_lengths.to(device=device).long()

    if torch.any(y_lengths <= 0):
        raise ValueError("All y_lengths must be positive.")
    if torch.any(x_lengths <= 0):
        raise ValueError("All x_lengths must be positive.")
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}.")

    gamma = gamma_posterior

    t = torch.arange(T_s, device=device, dtype=dtype).view(1, T_s, 1)
    j = torch.arange(T_t, device=device, dtype=dtype).view(1, 1, T_t)

    T = y_lengths.to(dtype=dtype).view(B, 1, 1)
    N = x_lengths.to(dtype=dtype).view(B, 1, 1)

    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)
    j_idx = torch.arange(T_t, device=device).view(1, 1, T_t)

    valid = (t_idx < y_lengths.view(B, 1, 1)) & (j_idx < x_lengths.view(B, 1, 1))
    t1 = torch.minimum(t + 1.0, T)

    alpha = omega * t1
    beta = omega * (T - t1 + 1.0)

    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    n = (N - 1.0).clamp_min(0.0)
    j_eff = torch.minimum(j, n)

    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(j_eff + 1.0) - torch.lgamma(n - j_eff + 1.0)

    log_beta_num = (
        torch.lgamma(j_eff + alpha)
        + torch.lgamma(n - j_eff + beta)
        - torch.lgamma(n + alpha + beta)
    )
    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    log_prior = log_comb + log_beta_num - log_beta_den  # (B, T_s, T_t)
    neg_large = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9

    log_prior = log_prior.masked_fill(~valid, neg_large)
    log_prior = log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)
    log_prior = log_prior.masked_fill(~valid, neg_large)

    gamma = gamma.masked_fill(~valid, 0.0)
    loss_per_frame = -(gamma * log_prior).sum(dim=-1)  # (B, T_s)

    valid_frame = t_idx.squeeze(-1) < y_lengths.view(B, 1)
    loss_per_frame = loss_per_frame.masked_fill(~valid_frame, 0.0)
    denom = y_lengths.to(dtype=dtype).sum().clamp_min(1.0)
    return loss_per_frame.sum() / denom


def compute_viterbi_kl_loss(
    log_gamma: torch.Tensor,
    viterbi_attn: torch.Tensor,
    spec_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the negative log-likelihood (NLL) of the optimal Viterbi path.
    This effectively acts as a KL divergence loss between the hard Viterbi
    distribution (from MAS) and the aligner's likelihood distribution.

    Args:
        log_gamma (torch.Tensor): Log-likelihood matrix of shape (B, T_mel, T_text).
            Typically computed as log-probabilities of mel frames given text features.
        viterbi_attn (torch.Tensor): Binary hard alignment path of shape (B, T_mel, T_text)
            obtained from `find_maximum_path`.
        spec_mask (torch.Tensor): Binary mask for spectrogram sequences of shape (B, T_mel).
            1.0 for valid frames, 0.0 for padding.

    Returns:
        torch.Tensor: Scalar tensor representing the averaged NLL loss.
            The loss is normalized by the total number of valid frames in the batch.
    """
    viterbi_log_probs = (viterbi_attn * log_gamma).sum(dim=-1)
    kl_per_frame = -viterbi_log_probs
    masked_kl = kl_per_frame * spec_mask
    total_kl = masked_kl.sum()
    total_frames = spec_mask.sum() + 1e-8

    loss = total_kl / total_frames

    return loss


def compute_viterbi_ot_loss(
    log_gamma: torch.Tensor,
    viterbi_attn: torch.Tensor,
    text_mask: torch.Tensor,
    spec_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a frame-wise 1D Optimal Transport loss between the predicted
    alignment distribution and the hard Viterbi alignment path.

    This uses the discrete 1D Wasserstein-1 distance on the text-token axis:
        W1(p, q) = sum_j |CDF_p(j) - CDF_q(j)|

    Args:
        log_gamma (torch.Tensor): Log-likelihood / logit matrix of shape (B, T_mel, T_text).
        viterbi_attn (torch.Tensor): Binary hard alignment path (B, T_mel, T_text).
        text_mask (torch.Tensor): Binary mask for text tokens (B, T_text) or (B, 1, T_text).
        spec_mask (torch.Tensor): Binary mask for spectrogram (B, T_mel).

    Returns:
        torch.Tensor: Scalar tensor representing the averaged frame-wise OT loss.
    """
    if text_mask.dim() == 2:
        text_mask = text_mask.unsqueeze(1)  # (B, 1, T_text)
    masked_log_gamma = log_gamma.masked_fill(text_mask == 0, -1e4)
    pred_attn = torch.softmax(masked_log_gamma, dim=-1)
    ot_diff = (pred_attn.cumsum(dim=-1) - viterbi_attn.cumsum(dim=-1)).abs()
    ot_per_frame = (ot_diff * text_mask).sum(dim=-1)
    masked_ot = ot_per_frame * spec_mask
    loss = masked_ot.sum() / (spec_mask.sum() + 1e-8)

    return loss


class NDAligner(BaseModel):
    def __init__(
        self,
        text_encoder: TextEncoder,
        spec_encoder: SpecEncoder,
        crf_aligner: MonotonicCRFAligner,
        spec_decoder: SpecDecoder | None = None,
        use_delta_mel: bool = False,
        use_delta_delta_mel: bool = False,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.spec_encoder = spec_encoder
        self.crf_aligner = crf_aligner

        self.spec_decoder = spec_decoder  # Optional Modules

        self.use_delta_mel = use_delta_mel
        self.use_delta_delta_mel = use_delta_delta_mel

    @override
    def forward(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        compute_viterbi_loss: bool = False,
        compute_diagonal_loss: bool = False,
    ) -> AlignerForward:
        assert self.spec_decoder is not None

        _, n_mels, T_mel = y.shape

        idx = torch.arange(T_mel, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        # 2. Pairwise CRF alignment
        out = self.compute_alignments(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            compute_hard_path=compute_viterbi_loss,
        )

        recon = self.spec_decoder(
            x=torch.bmm(out.soft_attn, out.h_text.transpose(1, 2)),
            cond=cond,
            mask=spec_mask,
        )

        crf_loss = -out.norm_log_z.mean()

        recon_loss = F.l1_loss(
            recon * spec_mask.unsqueeze(1),
            y * spec_mask.unsqueeze(1),
            reduction="sum",
        )
        recon_loss = recon_loss / (spec_mask.sum() * n_mels).clamp_min(1.0)

        diag_loss = torch.zeros((), device=x.device)

        if compute_diagonal_loss:
            diag_loss = compute_beta_binomial_loss(
                gamma_posterior=out.soft_attn,
                y_lengths=y_lengths,
                x_lengths=x_lengths,
                omega=1.0,
            )

        viterbi_kl_loss = torch.zeros((), device=x.device)
        viterbi_ot_loss = torch.zeros((), device=x.device)

        if compute_viterbi_loss and out.hard_attn is not None:
            spec_mask_2d = spec_mask.squeeze(1) if spec_mask.dim() == 3 else spec_mask
            text_mask_bool = sequence_mask(x_lengths, x.size(1))

            viterbi_kl_loss = compute_viterbi_kl_loss(
                log_gamma=out.log_gamma,
                viterbi_attn=out.hard_attn,
                spec_mask=spec_mask_2d,
            )

            viterbi_ot_loss = compute_viterbi_ot_loss(
                log_gamma=out.log_gamma,
                viterbi_attn=out.hard_attn,
                text_mask=text_mask_bool,
                spec_mask=spec_mask_2d,
            )

        return AlignerForward(
            soft_dur=out.soft_dur,
            soft_attn=out.soft_attn,
            hard_attn=out.hard_attn,
            hard_dur=out.hard_dur,
            h_text=out.h_text,
            h_spec=out.h_spec,
            log_alpha=out.log_alpha,
            log_beta=out.log_beta,
            log_gamma=out.log_gamma,
            raw_log_z=out.raw_log_z,
            norm_log_z=out.norm_log_z,
            raw_unary=out.raw_unary,
            masked_raw_unary=out.masked_raw_unary,
            log_b=out.log_b,
            viterbi_path=out.viterbi_path,
            viterbi_logp=out.viterbi_logp,
            recon=recon.transpose(1, 2),
            recon_loss=recon_loss,
            crf_loss=crf_loss,
            diag_loss=diag_loss,
            viterbi_kl_loss=viterbi_kl_loss,
            viterbi_ot_loss=viterbi_ot_loss,
        )

    @override
    @torch.no_grad
    def inference(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        compute_hard_path: bool = True,
    ) -> AlignerFeatures:
        assert self.spec_encoder is not None
        assert self.crf_aligner is not None

        return self.compute_alignments(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            compute_hard_path=compute_hard_path,
        )

    def compute_alignments(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        compute_hard_path: bool = False,
    ) -> AlignerFeatures:
        """
        Extracts Only Alignment w/o reconstruction using decoder.
        """

        assert self.spec_encoder is not None
        assert self.crf_aligner is not None

        _, _n_mels, T_mel = y.shape

        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).to(dtype=y.dtype)

        idx = torch.arange(T_mel, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        # 1. Text / speech representations (for alignment only)
        h_text = self.text_encoder(
            x=x,
            x_mask=text_mask,
        )  # (B, C_text, T_text)

        y_feat = self._build_spec_encoder_input(y)

        h_spec = self.spec_encoder(
            x=y_feat,
            mask=spec_mask.unsqueeze(1),
            cond=cond,
        )  # (B, C_spec, T_mel)

        device = h_text.device
        dtype = h_text.dtype

        if text_mask.dim() == 2:
            text_mask = text_mask.unsqueeze(1)
        text_mask = text_mask.to(device=device, dtype=dtype)

        viterbi_path = viterbi_logp = None
        hard_attn = hard_dur = None

        assert spec_mask is not None
        assert self.crf_aligner is not None

        if spec_mask.dim() == 3:
            spec_mask_2d = spec_mask.squeeze(1)
        else:
            spec_mask_2d = spec_mask

        spec_mask_2d = spec_mask_2d.to(device=device, dtype=dtype)

        if compute_hard_path:
            (
                soft_gamma,
                log_alpha,
                log_beta,
                log_gamma,
                raw_log_z,
                norm_log_z,
                raw_unary,
                masked_raw_unary,
                log_b,
                soft_dur,
                hard_attn,
                hard_dur,
                viterbi_path,
                viterbi_logp,
            ) = self.crf_aligner(
                h_spec=h_spec,
                spec_mask=spec_mask_2d.unsqueeze(1),
                h_text=h_text,
                text_mask=text_mask,
                cond=cond,
                return_hard=True,
            )
        else:
            (
                soft_gamma,
                log_alpha,
                log_beta,
                log_gamma,
                raw_log_z,
                norm_log_z,
                raw_unary,
                masked_raw_unary,
                log_b,
                soft_dur,
            ) = self.crf_aligner(
                h_spec=h_spec,
                spec_mask=spec_mask_2d.unsqueeze(1),
                h_text=h_text,
                text_mask=text_mask,
                cond=cond,
                return_hard=False,
            )

        return AlignerFeatures(
            # attention / durations
            soft_dur=soft_dur,
            soft_attn=soft_gamma,
            hard_dur=hard_dur,
            hard_attn=hard_attn,
            # text / speech representations
            h_text=h_text,
            h_spec=h_spec,
            # forward-backward internals
            log_alpha=log_alpha,
            log_beta=log_beta,
            log_gamma=log_gamma,
            raw_log_z=raw_log_z,
            norm_log_z=norm_log_z,
            # unary-potentials
            log_b=log_b,
            raw_unary=raw_unary,
            masked_raw_unary=masked_raw_unary,
            # viterbi(hard)-aware features
            viterbi_path=viterbi_path,
            viterbi_logp=viterbi_logp,
        )

    @staticmethod
    def _compute_delta_feature(
        x: torch.Tensor,
        width: int = 5,
    ) -> torch.Tensor:
        """
        Regression-style delta feature along time axis.

        Args:
            x: (B, C, T)
            width: odd integer, usually 5.

        Returns:
            delta: (B, C, T)
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, C, T), got {tuple(x.shape)}")

        if width % 2 == 0 or width < 3:
            raise ValueError(f"width must be odd and >= 3, got {width}")

        if x.size(-1) <= 1:
            return torch.zeros_like(x)

        n = width // 2
        denom = 2 * sum(i * i for i in range(1, n + 1))

        kernel = torch.arange(-n, n + 1, device=x.device, dtype=x.dtype)
        kernel = kernel / denom
        kernel = kernel.view(1, 1, width)

        b, c, t = x.shape
        x_flat = x.reshape(b * c, 1, t)
        x_flat = F.pad(x_flat, (n, n), mode="replicate")

        delta = F.conv1d(x_flat, kernel)
        delta = delta.reshape(b, c, t)

        return delta

    @torch.no_grad()
    def _build_spec_encoder_input(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds speech-side input feature for SpecEncoder.

        Args:
            y: raw log-mel, (B, n_mels, T)

        Returns:
            y_feat:
                - mel only: (B, n_mels, T)
                - mel + delta: (B, 2 * n_mels, T)
                - mel + delta + delta-delta: (B, 3 * n_mels, T)
        """
        features = [y]

        if self.use_delta_mel:
            with torch.no_grad():
                delta_y = self._compute_delta_feature(y, width=5)
            features.append(delta_y)

        if self.use_delta_delta_mel:
            if not self.use_delta_mel:
                raise ValueError("use_delta_delta_mel=True requires use_delta_mel=True.")

            with torch.no_grad():
                delta_delta_y = self._compute_delta_feature(delta_y, width=5)
            features.append(delta_delta_y)

        if len(features) == 1:
            return y

        return torch.cat(features, dim=1)


class NDAlignerTrainingModuleForward(NamedTuple):
    aligner_output: AlignerForward
    loss: torch.Tensor


class NDAlignerLossWeights(NamedTuple):
    crf_loss_weight: float
    diag_loss_weight: float = 0.0
    recon_loss_weight: float = 0.0
    viterbi_kl_loss_weight: float = 0.0
    viterbi_ot_loss_weight: float = 0.0


class NDAlignerTrainingModule(BaseModel):
    """
    Training/evaluation wrapper for NDAligner.

    This module delegates alignment computation to the inner NDAligner and adds:
        - weighted loss aggregation from AlignerForwardOutput
        - optional vocoder-based mel-to-waveform conversion
        - optional speaker-embedding extraction

    The wrapped NDAligner remains responsible for monotonic CRF alignment,
    posterior computation, Viterbi decoding, and auxiliary loss terms.
    """

    def __init__(
        self,
        nd_aligner: NDAligner,
        vocoder: Generator | None = None,
        speaker_encoder: ECAPASpeakerEncoder | None = None,
    ):
        super().__init__()

        self.nd_aligner = nd_aligner

        # External Modules
        self.speaker_encoder = speaker_encoder
        self.vocoder = vocoder

    @override
    def forward(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        loss_weights: NDAlignerLossWeights,
        compute_viterbi_loss: bool = False,
        compute_diagonal_loss: bool = False,
    ) -> NDAlignerTrainingModuleForward:
        out = cast(
            AlignerForward,
            self.nd_aligner(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                cond=cond,
                compute_diagonal_loss=compute_diagonal_loss,
                compute_viterbi_loss=compute_viterbi_loss,
            ),
        )

        total_loss = self._compute_total_loss(
            out=out,
            loss_weights=loss_weights,
        )

        return NDAlignerTrainingModuleForward(
            aligner_output=out,
            loss=total_loss,
        )

    @torch.no_grad()
    def mel2wav(self, x: torch.Tensor) -> torch.Tensor:
        assert self.vocoder is not None, "Vocoder is not initialized."

        if x.shape[1] != self.vocoder.h.num_mels:
            x = x.transpose(1, 2)

        with torch.no_grad():
            wav = self.vocoder(x)

        return wav

    @torch.no_grad()
    def compute_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        assert self.speaker_encoder is not None, "Speaker encoder is not initialized."
        return self.speaker_encoder(waveform)

    def _compute_total_loss(
        self,
        out: AlignerForward,
        loss_weights: NDAlignerLossWeights,
    ) -> torch.Tensor:
        total_loss = out.crf_loss.new_zeros(())

        for name, weight in loss_weights._asdict().items():
            if weight == 0.0:
                continue

            loss_name = name.removesuffix("_weight")
            loss = getattr(out, loss_name)

            total_loss = total_loss + weight * loss

        return total_loss
