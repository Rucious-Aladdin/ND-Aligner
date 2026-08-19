import dataclasses
from pathlib import Path
from typing import NamedTuple, cast, override

import librosa
import torch
import torch.nn.functional as F

from tts.models.modules.coupling_decoder import CouplingDecoder, CouplingDecoderOutput
from tts.models.modules.coupling_decoder_conv2d import (
    CouplingConv2dDecoder,
    CouplingConv2dDecoderOutput,
)
from tts.models.modules.crf_aligner import LinearCRFAligner, validate_decoding_strategy
from tts.models.modules.decoder import Decoder
from tts.models.modules.spec_encoder import SpecEncoder
from tts.models.modules.text_encoder import TextEncoder
from tts.models.modules.utils.sequence_mask import sequence_mask
from tts.models.utils.input_maker import AlignerInputMaker

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

    text_encoder = TextEncoder(**dataclasses.asdict(config.txt_enc)).to(device)
    spec_encoder = SpecEncoder(**dataclasses.asdict(config.spec_enc)).to(device)
    aligner = LinearCRFAligner(**dataclasses.asdict(config.aligner)).to(device)

    dec_cfg = config.spec_dec
    if config.spec_dec.decoder_type == "conv1d":
        spec_decoder = Decoder(
            in_channels=dec_cfg.in_channels,
            out_channels=dec_cfg.out_channels,
            hidden_channels=dec_cfg.hidden_channels,
            cond_dim=dec_cfg.cond_dim,
            kernel_sizes=dec_cfg.kernel_sizes,
            dilation_base=dec_cfg.dilation_base,
            dropout=dec_cfg.dropout,
        ).to(device)
    elif config.spec_dec.decoder_type == "coupling":
        spec_decoder = CouplingDecoder(
            in_channels=dec_cfg.in_channels,
            out_channels=dec_cfg.out_channels,
            hidden_channels=dec_cfg.hidden_channels,
            cond_dim=dec_cfg.cond_dim,
            cond_proj_dim=dec_cfg.coupling_cond_proj_dim,
            kernel_size=dec_cfg.coupling_kernel_size,
            num_refinement_steps=dec_cfg.coupling_num_refine_steps,
            dilation=dec_cfg.dilation_base,
            dropout=dec_cfg.dropout,
            step_emb_dim=dec_cfg.coupling_step_emb_dim,
            loss_decay_factor=dec_cfg.coupling_loss_decay_factor,
            normalize_loss_weights=dec_cfg.coupling_normalize_loss_weights,
        ).to(device)
    elif config.spec_dec.decoder_type == "coupling_conv2d":
        spec_decoder = CouplingConv2dDecoder(
            in_channels=dec_cfg.in_channels,
            out_channels=dec_cfg.out_channels,
            hidden_channels=dec_cfg.hidden_channels,
            cond_dim=dec_cfg.cond_dim,
            cond_proj_dim=dec_cfg.coupling_conv2d_cond_proj_dim,
            kernel_size=dec_cfg.coupling_conv2d_kernel_size,
            num_refinement_steps=dec_cfg.coupling_conv2d_num_refine_steps,
            dilation=dec_cfg.dilation_base,
            dropout=dec_cfg.dropout,
            step_emb_dim=dec_cfg.coupling_conv2d_step_emb_dim,
            loss_decay_factor=dec_cfg.coupling_conv2d_loss_decay_factor,
            normalize_loss_weights=dec_cfg.coupling_conv2d_normalize_loss_weights,
        ).to(device)

    # for inference
    input_maker = (
        AlignerInputMaker(
            audio_config=config.audio,
            preprocess_config=config.preprocess,
            tokenizer_type=config.tokenizer_type,
            fastspeech2_lexicon_path=config.fastspeech2_tokenizer_lexion_path,
            zero_nonspeech_region=False,
            trim_nonspeech_region=False,
            device=device,
        )
        if load_input_maker
        else None
    )

    return NDAligner(
        text_encoder=text_encoder,
        spec_encoder=spec_encoder,
        crf_aligner=aligner,
        spec_decoder=spec_decoder,
        input_maker=input_maker,
        use_delta_feat=config.use_delta_feat,
        use_delta_delta_feat=config.use_delta_delta_feat,
        use_optional_skip_sep=config.use_optional_skip_sep,
        separator_token_id=config.separator_token_id,
        viterbi_ste_training=config.viterbi_ste_training,
    )


def init_nd_aligner_training_module(
    config: NDAlignerTrainingModuleConfigs | None = None,
    load_input_maker: bool = True,
    device: str = "cpu",
):
    if config is None:
        config = NDAlignerTrainingModuleConfigs()

    nd_aligner = init_nd_aligner(
        config=config.nd_aligner,
        load_input_maker=load_input_maker,
        device=device,
    )

    return NDAlignerTrainingModule(nd_aligner=nd_aligner)


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

    # Optional
    coupling_dec_out: CouplingDecoderOutput | CouplingConv2dDecoderOutput | None


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


class AlignerInference(NamedTuple):
    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    attn: torch.Tensor  # (B, T_s, T_t)
    durations: (
        torch.Tensor
    )  # (B, T_t) <- IntegerDuration for hard-path, FloatDuration for soft-path

    # basics
    texts: list[str]  # scripts
    phones: list[str]
    words: list[str]

    # grid output
    phoneme_grid: list[list[tuple[float, float, str]]] | None  # (start_sec, end_sec, phone)
    word_grid: list[list[tuple[float, float, str]]] | None  # (start_sec, end_sec, word)


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


class NDAligner(BaseModel):
    def __init__(
        self,
        text_encoder: TextEncoder,
        spec_encoder: SpecEncoder,
        crf_aligner: LinearCRFAligner,
        spec_decoder: Decoder | CouplingDecoder | CouplingConv2dDecoder | None = None,
        input_maker: AlignerInputMaker | None = None,
        use_delta_feat: bool = False,
        use_delta_delta_feat: bool = False,
        use_optional_skip_sep: bool = False,
        separator_token_id: int = -1,
        viterbi_ste_training: bool = False,
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

        self.viterbi_ste_training = viterbi_ste_training
        self.decoding_strategy = "viterbi"

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
        compute_hard_path: bool = False,
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

        # Viterbi STE needs both the soft posterior and the hard Viterbi path.
        # Keep the caller-facing compute_hard_path option, but force hard-path
        # extraction whenever STE training is enabled.
        out = self.compute_alignments(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            compute_soft_path=True,
            compute_hard_path=(compute_hard_path or self.viterbi_ste_training),
        )

        if self.viterbi_ste_training:
            assert out.hard_attn is not None

            # Straight-through estimator:
            #   forward : hard Viterbi alignment
            #   backward: identity gradient w.r.t. the soft posterior alignment
            #
            # Numerically, ste_attn == hard_attn in the forward pass, while
            # d(ste_attn)/d(soft_attn) = 1.
            ste_attn = out.soft_attn + (out.hard_attn - out.soft_attn).detach()
            recon_attn = ste_attn
        else:
            recon_attn = out.soft_attn

        aligned_h = torch.bmm(
            recon_attn,
            out.h_text.transpose(1, 2),
        )  # (B, T_spec, C_text)

        crf_loss = -out.norm_log_z.mean()

        valid_spec_mask = spec_mask.bool()  # (B, T_spec), True = valid

        y_recon_bt = y_recon.transpose(1, 2).contiguous()
        y_recon_bt = y_recon_bt.to(
            device=aligned_h.device,
            dtype=aligned_h.dtype,
        )

        coupling_dec_out = None
        if isinstance(self.spec_decoder, CouplingDecoder):
            # Coupling decoder:
            #   input  : (B, T_spec, C_text)
            #   target : (B, T_spec, n_mels)
            #   output : mel_hat (B, T_spec, n_mels)
            dec_out = self.spec_decoder(
                x=aligned_h,
                cond=cond,
                mask=valid_spec_mask,
                target=y_recon_bt,
            )

            recon = dec_out.mel_hat
            recon_loss = dec_out.loss
            coupling_dec_out = dec_out
        elif isinstance(self.spec_decoder, CouplingConv2dDecoder):
            dec_out = self.spec_decoder.forward(
                h_text=out.h_text,
                gamma=recon_attn,
                target=y_recon,
                cond=cond,
                spec_lengths=y_lengths,
                text_lengths=x_lengths,
            )

            recon = dec_out.mel_hat
            recon_loss = dec_out.loss
            coupling_dec_out = dec_out
        else:
            recon = self.spec_decoder(
                x=aligned_h,
                cond=cond,
                mask=spec_mask,
            )  # (B, T_spec, n_mels)

            recon_mask = spec_mask.to(
                device=recon.device,
                dtype=recon.dtype,
            ).unsqueeze(
                -1
            )  # (B, T_spec, 1)

            recon_loss = F.l1_loss(
                recon * recon_mask,
                y_recon_bt * recon_mask,
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
            recon=recon,
            recon_loss=recon_loss,
            crf_loss=crf_loss,
            diag_loss=diag_loss,
            coupling_dec_out=coupling_dec_out,
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
                forward-backward + selected hard decoder

            compute_soft_path=False, compute_hard_path=True:
                hard decoding only (posterior strategies run forward-backward internally)

        At least one of compute_soft_path and compute_hard_path must be True.


        Note:
            "posterior_viterbi" and "mea" require forward-backward posterior
            computation internally, even when compute_soft_path=False.
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

        opt_sep_mask = self._make_opt_sep_mask(
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
                opt_sep_mask=opt_sep_mask,
                return_soft=compute_soft_path,
                return_hard=True,
                decoding_strategy=self.decoding_strategy,
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
                opt_sep_mask=opt_sep_mask,
                return_soft=True,
                return_hard=False,
                decoding_strategy=self.decoding_strategy,
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

    def set_decoding_strategy(self, strategy: str):
        """
        strategy:
            "viterbi": standard Viterbi decoding from CRF node potentials.
            "posterior_viterbi": Viterbi-style decoding from log posterior marginals.
            "mea": maximum-expected-accuracy decoding from posterior marginals.
        """
        validate_decoding_strategy(strategy)
        self.decoding_strategy = strategy

    def _make_opt_sep_mask(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor | None:
        if not self.use_optional_skip_sep:
            return None

        text_mask = sequence_mask(x_lengths, x.size(1))
        opt_sep_mask = x.eq(self.separator_token_id) & text_mask
        opt_sep_mask = opt_sep_mask.clone()

        if opt_sep_mask.size(1) > 0:
            batch_idx = torch.arange(
                opt_sep_mask.size(0),
                device=opt_sep_mask.device,
            )
            last_idx = (x_lengths.to(device=opt_sep_mask.device).long() - 1).clamp_min(0)

            opt_sep_mask[:, 0] = False
            opt_sep_mask[batch_idx, last_idx] = False

        if not torch.any(opt_sep_mask):
            return None

        return opt_sep_mask

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
    def inference_from_wavs(
        self,
        wav_paths: str | Path | list[str | Path],
        texts: str | list[str],
        compute_phone_grid: bool = False,
        compute_word_grid: bool = False,
        include_space_token_to_grid: bool = True,
        decoding_strategy: str = "viterbi",
    ) -> AlignerInference:
        assert self.spec_encoder is not None
        assert self.crf_aligner is not None
        assert self.input_maker is not None

        """
        Run hard-path alignment directly from waveform paths.

        If AlignerInputMaker trims the waveform, the returned attention and
        durations are restored to the frame coordinate of the original,
        untrimmed waveform. Removed leading frames are assigned to the first
        token and removed trailing frames are assigned to the last valid token.

        When include_space_token_to_grid=True, positive-duration separator
        tokens are included in both phoneme_grid and word_grid.
        """

        # ------------------------------------------------------------
        # 1. Normalize arguments and build model inputs.
        # ------------------------------------------------------------
        if isinstance(wav_paths, (str, Path)):
            wav_path_list = [Path(wav_paths)]
        else:
            wav_path_list = [Path(path) for path in wav_paths]

        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        model_device = next(self.parameters()).device
        batch = self.input_maker.make_with_audio(
            wav_paths=wav_path_list,  # type: ignore
            scripts=text_list,
        )

        # ------------------------------------------------------------
        # 2. Compute a hard alignment on the trimmed spectrogram.
        # ------------------------------------------------------------
        assert batch.cond is not None

        features = self.compute_alignments(
            x=batch.x,
            x_lengths=batch.x_lengths,
            y=batch.y,
            y_lengths=batch.y_lengths,
            cond=batch.cond,
            compute_soft_path=False,
            compute_hard_path=True,
        )

        assert features.hard_attn is not None
        assert features.hard_dur is not None

        trimmed_attn = features.hard_attn
        trimmed_durations = features.hard_dur.round().long()

        # ------------------------------------------------------------
        # 3. Recover original, untrimmed spectrogram lengths.
        # ------------------------------------------------------------
        model_sr = int(self.input_maker.model_sr)
        hop_length = int(self.input_maker.audio_config.hop_length)
        batch_size = len(wav_path_list)

        original_model_length_list: list[int] = []

        for wav_path in wav_path_list:
            wav_np, _ = librosa.load(str(wav_path), sr=model_sr, mono=True)
            original_length = int(wav_np.shape[0])

            if original_length <= 0:
                raise RuntimeError(f"Loaded an empty waveform: {wav_path}")
            original_model_length_list.append(original_length)

        original_model_lengths = torch.tensor(
            original_model_length_list,
            dtype=torch.long,
            device=model_device,
        )

        start_offset_16k = batch.wav_16k_start_offset.to(device=model_device, dtype=torch.long)
        end_offset_16k = batch.wav_16k_end_offset.to(device=model_device, dtype=torch.long)

        model_start_samples = torch.div(
            start_offset_16k * model_sr,
            16_000,
            rounding_mode="floor",
        )
        model_end_samples = torch.div(
            (end_offset_16k + 1) * model_sr + 16_000 - 1,
            16_000,
            rounding_mode="floor",
        )
        model_start_samples = torch.minimum(
            model_start_samples.clamp_min(0), (original_model_lengths - 1).clamp_min(0)
        )
        model_end_samples = torch.minimum(model_end_samples, original_model_lengths)
        model_end_samples = torch.maximum(model_end_samples, model_start_samples + 1)

        original_spec_lengths = self.input_maker.compute_spec_lengths(original_model_lengths)
        prefix_spec_lengths = self.input_maker.compute_spec_lengths(model_start_samples).clamp_min(
            0
        )
        suffix_spec_lengths = original_spec_lengths - prefix_spec_lengths - batch.y_lengths

        # ------------------------------------------------------------
        # 4. Restore attention and durations to the original frame grid.
        # ------------------------------------------------------------
        max_original_spec_length = int(original_spec_lengths.max().item())
        max_text_length = int(batch.x.size(1))

        attn = trimmed_attn.new_zeros(
            (
                batch_size,
                max_original_spec_length,
                max_text_length,
            )
        )

        durations = torch.zeros(
            (batch_size, max_text_length),
            dtype=torch.long,
            device=model_device,
        )

        for batch_idx in range(batch_size):
            text_length = int(batch.x_lengths[batch_idx].item())
            trimmed_spec_length = int(batch.y_lengths[batch_idx].item())
            original_spec_length = int(original_spec_lengths[batch_idx].item())
            prefix_length = int(prefix_spec_lengths[batch_idx].item())
            suffix_length = int(suffix_spec_lengths[batch_idx].item())

            if text_length <= 0:
                raise RuntimeError(f"Sample {batch_idx} contains no valid text tokens.")

            last_token_idx = text_length - 1
            trimmed_start = prefix_length
            trimmed_end = trimmed_start + trimmed_spec_length

            if prefix_length > 0:
                attn[batch_idx, :prefix_length, 0] = 1.0

            attn[batch_idx, trimmed_start:trimmed_end, :text_length] = trimmed_attn[
                batch_idx, :trimmed_spec_length, :text_length
            ]

            if suffix_length > 0:
                attn[batch_idx, trimmed_end:original_spec_length, last_token_idx] = 1.0

            durations[batch_idx, :text_length] = trimmed_durations[batch_idx, :text_length]
            durations[batch_idx, 0] += prefix_length
            durations[batch_idx, last_token_idx] += suffix_length

        # ------------------------------------------------------------
        # 5. Decode the exact token sequence used by the aligner.
        # ------------------------------------------------------------
        tokenizer = self.input_maker.tokenizer
        separator_id = int(tokenizer.seperator_id)

        phone_symbols_per_sample: list[list[str]] = []
        phone_strings: list[str] = []
        word_strings: list[str] = []

        for batch_idx, text in enumerate(batch.texts):
            text_length = int(batch.x_lengths[batch_idx].item())
            token_ids_1d = batch.x[batch_idx, :text_length].detach().cpu()
            symbols_raw = tokenizer.decode_to_symbols(token_ids_1d)
            symbols = [str(symbol) for symbol in symbols_raw]

            phone_symbols_per_sample.append(symbols)
            phone_strings.append(tokenizer.to_token_string(text))
            word_strings.append(" ".join(text.split()))

        sec_per_frame = hop_length / model_sr

        # ------------------------------------------------------------
        # 6. Optional phoneme grid.
        # ------------------------------------------------------------
        phoneme_grid: list[list[tuple[float, float, str]]] | None = None

        if compute_phone_grid:
            phoneme_grid = []

            for batch_idx, symbols in enumerate(phone_symbols_per_sample):
                text_length = int(batch.x_lengths[batch_idx].item())

                current_frame = 0
                sample_grid: list[tuple[float, float, str]] = []

                for token_idx in range(text_length):
                    duration = int(durations[batch_idx, token_idx].item())

                    start_frame = current_frame
                    current_frame += duration

                    if duration <= 0:
                        continue

                    token_id = int(batch.x[batch_idx, token_idx].item())

                    if not include_space_token_to_grid and token_id == separator_id:
                        continue

                    sample_grid.append(
                        (
                            start_frame * sec_per_frame,
                            current_frame * sec_per_frame,
                            symbols[token_idx],
                        )
                    )

                phoneme_grid.append(sample_grid)

        # ------------------------------------------------------------
        # 7. Optional word grid.
        # ------------------------------------------------------------
        word_grid: list[list[tuple[float, float, str]]] | None = None

        if compute_word_grid:
            mapper = self.input_maker.word_mapper
            word_grid = []

            for batch_idx, text in enumerate(batch.texts):
                text_length = int(batch.x_lengths[batch_idx].item())

                symbols = phone_symbols_per_sample[batch_idx]
                ref_words = text.split()

                matched = mapper(ref_seqs=ref_words, hyp_seqs=symbols)
                cumulative_frames = torch.cat(
                    [durations.new_zeros(1), durations[batch_idx, :text_length].cumsum(dim=0)],
                    dim=0,
                )

                match_by_hyp_start: dict[
                    int,
                    tuple[slice, slice],
                ] = {
                    cast(int, hyp_slice.start): (ref_slice, hyp_slice)
                    for ref_slice, hyp_slice in zip(
                        matched.ref_matched_indices,
                        matched.hyp_matched_indices,
                        strict=True,
                    )
                }

                sample_grid: list[tuple[float, float, str]] = []

                token_idx = 0

                while token_idx < text_length:
                    matched_item = match_by_hyp_start.get(token_idx)

                    if matched_item is not None:
                        ref_slice, hyp_slice = matched_item

                        ref_start = cast(int, ref_slice.start)
                        ref_stop = cast(int, ref_slice.stop)
                        hyp_start = cast(int, hyp_slice.start)
                        hyp_stop = cast(int, hyp_slice.stop)

                        start_frame = int(cumulative_frames[hyp_start].item())
                        end_frame = int(cumulative_frames[hyp_stop].item())

                        if end_frame > start_frame:
                            label = " ".join(ref_words[ref_start:ref_stop])

                            sample_grid.append(
                                (
                                    start_frame * sec_per_frame,
                                    end_frame * sec_per_frame,
                                    label,
                                )
                            )

                        token_idx = hyp_stop
                        continue

                    token_id = int(batch.x[batch_idx, token_idx].item())
                    if include_space_token_to_grid and token_id == separator_id:
                        start_frame = int(cumulative_frames[token_idx].item())
                        end_frame = int(cumulative_frames[token_idx + 1].item())

                        if end_frame > start_frame:
                            sample_grid.append(
                                (
                                    start_frame * sec_per_frame,
                                    end_frame * sec_per_frame,
                                    " ",
                                )
                            )
                    token_idx += 1

                word_grid.append(sample_grid)

        return AlignerInference(
            token_ids=batch.x,
            token_lengths=batch.x_lengths,
            attn=attn,
            durations=durations,
            texts=batch.texts,
            phones=phone_strings,
            words=word_strings,
            phoneme_grid=phoneme_grid,
            word_grid=word_grid,
        )


class NDAlignerTrainingModuleForward(NamedTuple):
    aligner_output: AlignerForward
    loss: torch.Tensor


class NDAlignerLossWeights(NamedTuple):
    crf_loss_weight: float
    diag_loss_weight: float = 0.0
    recon_loss_weight: float = 0.0


class NDAlignerTrainingModule(BaseModel):
    """
    Training/evaluation wrapper for NDAligner.

    This module delegates alignment computation to the inner NDAligner and adds:
        - weighted loss aggregation from AlignerForwardOutput

    The wrapped NDAligner remains responsible for monotonic CRF alignment,
    posterior computation, Viterbi decoding, and auxiliary loss terms.
    """

    def __init__(
        self,
        nd_aligner: NDAligner,
    ):
        super().__init__()

        self.nd_aligner = nd_aligner

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
        compute_hard_path: bool = False,
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
                compute_hard_path=compute_hard_path,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        total_loss = out.crf_loss.new_zeros(())
        for name, weight in loss_weights._asdict().items():
            if weight == 0.0:
                continue
            loss_name = name.removesuffix("_weight")
            loss = getattr(out, loss_name)
            total_loss = total_loss + weight * loss

        return NDAlignerTrainingModuleForward(
            aligner_output=out,
            loss=total_loss,
        )
