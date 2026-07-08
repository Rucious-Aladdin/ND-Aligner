import dataclasses
from typing import NamedTuple, cast, override

import torch
import torch.nn.functional as F

from tts.models.modules.crf_aligner import LinearCRFAligner
from tts.models.modules.hifigan_vocoder import Generator
from tts.models.modules.spec_decoder import SpecDecoder
from tts.models.modules.spec_encoder import SpecEncoder
from tts.models.modules.text_encoder import TextEncoder
from tts.models.utils.input_maker import AlignerInputMaker
from tts.models.utils.sequence_mask import sequence_mask

from ..config.ndaligner.model_config import NDAlignerConfigs
from ..config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from .utils.base_model import BaseModel


def init_nd_aligner(
    config: NDAlignerConfigs | None = None,
    load_input_maker: bool = True,
    device: str = "cpu",
):

    if config is None:
        config = NDAlignerConfigs()

    text_encoder = TextEncoder(**dataclasses.asdict(config.txt_enc))
    spec_encoder = SpecEncoder(**dataclasses.asdict(config.spec_enc))
    aligner = LinearCRFAligner(**dataclasses.asdict(config.aligner))
    spec_decoder = SpecDecoder(**dataclasses.asdict(config.spec_dec))

    # for inference

    if load_input_maker:
        input_maker = AlignerInputMaker(
            audio_config=config.audio,
            preprocess_config=config.preprocess,
            tokenizer_type=config.tokenizer_type,
            fastspeech2_lexicon_path=config.fastspeech2_tokenizer_lexion_path,
            device=device,
        )
    else:
        input_maker = None

    model = NDAligner(
        text_encoder=text_encoder,
        spec_encoder=spec_encoder,
        crf_aligner=aligner,
        spec_decoder=spec_decoder,
        input_maker=input_maker,
        use_delta_feat=config.use_delta_feat,
        use_delta_delta_feat=config.use_delta_delta_feat,
        use_optional_skip_sep=config.use_optional_skip_sep,
        separator_token_id=config.separator_token_id,
    )

    return model.to(device)


def init_nd_aligner_training_module(
    config: NDAlignerTrainingModuleConfigs | None = None,
    load_vocoder: bool = False,
    device: str = "cpu",
):

    if config is None:
        config = NDAlignerTrainingModuleConfigs()

    nd_aligner = init_nd_aligner(
        config=config.nd_aligner,
        device=device,
    )

    vocoder = None

    if load_vocoder:
        vocoder = Generator.from_config_path(
            config_path=config.vocoder.config_path,
            ckpt_path=config.vocoder.ckpt_path,
        ).to(device)

    return NDAlignerTrainingModule(
        nd_aligner=nd_aligner,
        vocoder=vocoder,
    )


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


class NDAligner(BaseModel):
    def __init__(
        self,
        text_encoder: TextEncoder,
        spec_encoder: SpecEncoder,
        crf_aligner: LinearCRFAligner,
        spec_decoder: SpecDecoder | None = None,
        input_maker: AlignerInputMaker | None = None,
        use_delta_feat: bool = False,
        use_delta_delta_feat: bool = False,
        use_optional_skip_sep: bool = False,
        separator_token_id: int = -1,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.spec_encoder = spec_encoder
        self.crf_aligner = crf_aligner

        self.spec_decoder = spec_decoder  # Optional Modules
        self.input_maker = input_maker  # Optional Modules

        self.use_delta_feat = use_delta_feat
        self.use_delta_delta_feat = use_delta_delta_feat

        self.use_optional_skip_sep = use_optional_skip_sep
        self.separator_token_id = separator_token_id

        if self.use_optional_skip_sep and (self.separator_token_id < 0):
            raise ValueError(
                f"Set to Use Seperator Skipping Toplogy, "
                + f"but seperator token-id({self.separator_token_id}) is not initialized,"
                + f"set use_optional_skip_sep=False or"
                + f"set proper seperator_token_id at initialization."
            )

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,  # alignment input feature: mel or linspec, (B, C_align, T)
        y_lengths: torch.Tensor,
        cond: torch.Tensor,
        y_recon: torch.Tensor | None = None,  # reconstruction target: mel, (B, n_mels, T)
        y_recon_lengths: torch.Tensor | None = None,
        compute_viterbi_loss: bool = False,
        compute_diagonal_loss: bool = False,
    ) -> AlignerForward:
        assert self.spec_decoder is not None

        _, _c_align, T_spec = y.shape

        if y_recon is None:
            y_recon = y

        if y_recon_lengths is None:
            y_recon_lengths = y_lengths

        if not torch.equal(y_lengths, y_recon_lengths):
            raise ValueError(
                "y_lengths and y_recon_lengths must match. "
                + f"got y_lengths={y_lengths.tolist()}, "
                + f"y_recon_lengths={y_recon_lengths.tolist()}"
            )

        _, n_recon_channels, _ = y_recon.shape

        idx = torch.arange(T_spec, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        out = self.compute_alignments(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            compute_soft_path=True,
            compute_hard_path=compute_viterbi_loss,
        )

        recon = self.spec_decoder(
            x=torch.bmm(out.soft_attn, out.h_text.transpose(1, 2)),
            cond=cond,
            mask=spec_mask,
        )

        crf_loss = -out.norm_log_z.mean()

        y_recon = y_recon.to(device=recon.device, dtype=recon.dtype)
        recon_mask = spec_mask.to(device=recon.device, dtype=recon.dtype).unsqueeze(1)

        recon_loss = F.l1_loss(
            recon * recon_mask,
            y_recon * recon_mask,
            reduction="sum",
        )
        recon_loss = recon_loss / (spec_mask.sum() * n_recon_channels).clamp_min(1.0)

        diag_loss = torch.zeros((), device=x.device)

        if compute_diagonal_loss:
            diag_loss = compute_beta_binomial_loss(
                gamma_posterior=out.soft_attn,
                y_lengths=y_lengths,
                x_lengths=x_lengths,
                omega=1.0,
            )

        viterbi_kl_loss = torch.zeros((), device=x.device)

        if compute_viterbi_loss and out.hard_attn is not None:
            spec_mask_2d = spec_mask.squeeze(1) if spec_mask.dim() == 3 else spec_mask

            viterbi_kl_loss = compute_viterbi_kl_loss(
                log_gamma=out.log_gamma,
                viterbi_attn=out.hard_attn,
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
        )

    @override
    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        compute_soft_path: bool = True,
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
            compute_soft_path=compute_soft_path,
            compute_hard_path=compute_hard_path,
        )

    def compute_alignments(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        y: torch.Tensor,  # (B, n_mels, T_mel)
        y_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        compute_soft_path: bool = True,
        compute_hard_path: bool = False,
    ) -> AlignerFeatures:
        """
        Extract alignments without reconstruction.

        Modes:
            compute_soft_path=True, compute_hard_path=False:
                forward-backward only

            compute_soft_path=True, compute_hard_path=True:
                forward-backward + Viterbi

            compute_soft_path=False, compute_hard_path=True:
                Viterbi only

        At least one of compute_soft_path and compute_hard_path must be True.
        """

        if (not compute_soft_path) and (not compute_hard_path):
            raise RuntimeError(
                "At least one of compute_soft_path and compute_hard_path must be True."
            )

        assert self.spec_encoder is not None
        assert self.crf_aligner is not None

        _, _n_mels, T_mel = y.shape

        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).to(dtype=y.dtype)

        optional_separator_mask = self._make_optional_separator_mask(
            x=x,
            x_lengths=x_lengths,
        )

        idx = torch.arange(T_mel, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        # ------------------------------------------------------------
        # 1. Text / speech representations
        # ------------------------------------------------------------
        h_text = self.text_encoder(
            x=x,
            x_mask=text_mask,
            cond=cond if getattr(self.text_encoder, "dim_cond", 0) > 0 else None,
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

        if spec_mask.dim() == 3:
            spec_mask_2d = spec_mask.squeeze(1)
        else:
            spec_mask_2d = spec_mask

        spec_mask_2d = spec_mask_2d.to(device=device, dtype=dtype)

        # ------------------------------------------------------------
        # 2. CRF alignment
        # ------------------------------------------------------------
        if compute_hard_path:
            crf_outputs = self.crf_aligner(
                h_spec=h_spec,
                spec_mask=spec_mask_2d.unsqueeze(1),
                h_text=h_text,
                text_mask=text_mask,
                cond=cond,
                optional_separator_mask=optional_separator_mask,
                return_soft=compute_soft_path,
                return_hard=True,
            )

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
            ) = crf_outputs

        else:
            crf_outputs = self.crf_aligner(
                h_spec=h_spec,
                spec_mask=spec_mask_2d.unsqueeze(1),
                h_text=h_text,
                text_mask=text_mask,
                cond=cond,
                optional_separator_mask=optional_separator_mask,
                return_soft=True,
                return_hard=False,
            )

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
            ) = crf_outputs

            hard_attn = None
            hard_dur = None
            viterbi_path = None
            viterbi_logp = None

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
            # unary potentials
            log_b=log_b,
            raw_unary=raw_unary,
            masked_raw_unary=masked_raw_unary,
            # viterbi-aware features
            viterbi_path=viterbi_path,
            viterbi_logp=viterbi_logp,
        )

    def _make_optional_separator_mask(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor | None:
        """
        Build an optional-separator mask for the CRF topology.

        Returns None when optional separator skipping is disabled or when the
        current batch contains no valid separator tokens. Returning None is
        intentional: LinearCRFAligner then uses the original strict stay/advance
        topology and avoids constructing skip-transition tensors.

        A True entry means the corresponding text token may receive zero
        duration by being skipped through the j-2 -> j transition.
        """
        if not self.use_optional_skip_sep:
            return None

        text_mask = sequence_mask(x_lengths, x.size(1))
        optional_separator_mask = x.eq(self.separator_token_id) & text_mask
        optional_separator_mask = optional_separator_mask.clone()

        # The start and terminal states are forced by the CRF topology.
        # Do not allow them to become optional even if a malformed tokenizer
        # emits the separator id at an endpoint.
        if optional_separator_mask.size(1) > 0:
            batch_idx = torch.arange(
                optional_separator_mask.size(0),
                device=optional_separator_mask.device,
            )
            last_idx = (x_lengths.to(device=optional_separator_mask.device).long() - 1).clamp_min(0)

            optional_separator_mask[:, 0] = False
            optional_separator_mask[batch_idx, last_idx] = False

        if not torch.any(optional_separator_mask):
            return None

        return optional_separator_mask

    @staticmethod
    def _compute_delta_feature(x: torch.Tensor) -> torch.Tensor:
        """
        First-order adjacent-frame difference along the time axis.

        Args:
            x: (B, C, T)
        Returns:
            delta: (B, C, T)
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, C, T), got {tuple(x.shape)}")

        if x.size(-1) <= 1:
            return torch.zeros_like(x)

        delta = torch.zeros_like(x)
        delta[..., 1:] = x[..., 1:] - x[..., :-1]

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

        if self.use_delta_feat:
            with torch.no_grad():
                delta_y = self._compute_delta_feature(y)
            features.append(delta_y)

        if self.use_delta_delta_feat:
            if not self.use_delta_feat:
                raise ValueError("use_delta_delta_mel=True requires use_delta_mel=True.")

            with torch.no_grad():
                delta_delta_y = self._compute_delta_feature(delta_y)
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
    ):
        super().__init__()

        self.nd_aligner = nd_aligner

        # External Modules
        self.vocoder = vocoder

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,  # alignment input feature
        y_lengths: torch.Tensor,
        cond: torch.Tensor,
        loss_weights: NDAlignerLossWeights,
        y_recon: torch.Tensor | None = None,  # mel reconstruction target
        y_recon_lengths: torch.Tensor | None = None,
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
                y_recon=y_recon,
                y_recon_lengths=y_recon_lengths,
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
        assert self.nd_aligner.input_maker is not None, "Speaker encoder is not initialized."
        return self.nd_aligner.input_maker.speaker_encoder(waveform)

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
