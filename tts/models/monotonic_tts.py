from typing import NamedTuple, override

import torch
import torch.nn.functional as F

from tts.models.modules.spec_decoder import SpecDecoder
from tts.models.modules.duration_predictor import StochasticDurationPredictor
from tts.models.modules.text_embedder import TextEmbedder
from tts.models.modules.hifigan_vocoder import Generator
from tts.models.modules.monotonic_aligner import MonotonicCRFAligner
from tts.models.modules.spec_encoder import SpecEncoder
from tts.models.modules.spk_encoder import SpeakerEncoder
from tts.models.utils.sequence_mask import sequence_mask
from tts.models.utils.losses import (
    compute_viterbi_kl_loss,
    compute_viterbi_ot_loss,
)
from tts.models.utils.beta_diagonal_loss import compute_beta_binomial_loss
from .base_model import BaseModel


class SynthesizerForwardOutput(NamedTuple):
    # alignment / duration
    dur: torch.Tensor
    attn: torch.Tensor
    hard_gamma: torch.Tensor | None
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
    raw_emission: torch.Tensor
    masked_raw_emission: torch.Tensor
    log_b: torch.Tensor
    log_a_stay: torch.Tensor
    log_a_adv: torch.Tensor

    viterbi_path: torch.Tensor | None
    viterbi_logp: torch.Tensor | None

    # reconstruction
    mel_recon: torch.Tensor

    # losses
    dur_loss: torch.Tensor
    mel_recon_loss: torch.Tensor
    align_forward_loss: torch.Tensor
    align_diag_loss: torch.Tensor
    align_viterbi_kl_loss: torch.Tensor
    align_viterbi_ot_loss: torch.Tensor


class SynthesizerInferenceOutput(NamedTuple):
    dur: torch.Tensor
    attn: torch.Tensor
    mel_hat: torch.Tensor
    wav_hat: torch.Tensor | None


class AlignedFeaturesOutput(NamedTuple):
    aligned_feats: torch.Tensor
    dur: torch.Tensor
    spec_mask: torch.Tensor
    y_lengths: torch.Tensor

    h_text: torch.Tensor

    # training-only
    h_spec: torch.Tensor | None = None

    soft_gamma: torch.Tensor | None = None
    hard_gamma: torch.Tensor | None = None
    hard_dur: torch.Tensor | None = None

    log_alpha: torch.Tensor | None = None
    log_beta: torch.Tensor | None = None
    log_gamma: torch.Tensor | None = None
    raw_log_z: torch.Tensor | None = None
    norm_log_z: torch.Tensor | None = None

    raw_evidence: torch.Tensor | None = None
    masked_raw_evidence: torch.Tensor | None = None
    log_b: torch.Tensor | None = None
    log_a_stay: torch.Tensor | None = None
    log_a_adv: torch.Tensor | None = None

    viterbi_path: torch.Tensor | None = None
    viterbi_logp: torch.Tensor | None = None


class MonotonicTTSSynthesizer(BaseModel):
    """
    TTS synthesizer using a pairwise monotonic latent-path CRF aligner.

    Main change from the categorical-emission version:
        - SpecEncoder no longer needs to predict 179-way phoneme logits for the
          alignment emission.
        - Alignment evidence is produced inside MonotonicCRFAligner from
          h_spec[t] and h_text[j].
        - The forward alignment objective should be interpreted as a monotone
          path energy / forward-sum objective, not a literal acoustic HMM NLL.
    """

    def __init__(
        self,
        text_encoder: TextEmbedder,
        dur_predictor: StochasticDurationPredictor,
        spec_decoder: SpecDecoder,
        aligner: MonotonicCRFAligner | None = None,
        spec_encoder: SpecEncoder | None = None,
        vocoder: Generator | None = None,
        speaker_encoder: SpeakerEncoder | None = None,
    ):
        super().__init__()

        self.text_embedder = text_encoder
        self.crf_aligner = aligner

        self.spec_encoder = spec_encoder
        self.spec_decoder = spec_decoder

        self.dur_predictor = dur_predictor
        self.vocoder = vocoder
        self.speaker_encoder = speaker_encoder

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
    ) -> SynthesizerForwardOutput:
        assert self.spec_encoder is not None
        assert self.crf_aligner is not None

        _, n_mels, T_mel = y.shape

        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).to(dtype=y.dtype)

        idx = torch.arange(T_mel, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        # ---------------------------
        # 1. Text / speech representations
        # ---------------------------
        h_text = self.text_embedder(
            text_token_ids=x,
            text_mask=text_mask,
        )  # (B, C_text, T_text)

        h_spec = self._encode_speech(
            y=y,
            spec_mask=spec_mask,
            cond=cond,
        )  # (B, C_spec, T_mel)

        # ---------------------------
        # 2. Pairwise CRF alignment
        # ---------------------------
        out = self.compute_aligned_feats(
            h_text=h_text,
            text_mask=text_mask,
            cond=cond,
            h_spec=h_spec,
            spec_mask=spec_mask,
            noise_scale=0.667,
            compute_hard_path=compute_viterbi_loss,
            to_hard_aligned_feats=False,
        )

        assert out.soft_gamma is not None
        assert out.h_spec is not None

        assert out.log_alpha is not None
        assert out.log_beta is not None
        assert out.log_gamma is not None
        assert out.raw_log_z is not None
        assert out.norm_log_z is not None

        assert out.raw_evidence is not None
        assert out.masked_raw_evidence is not None
        assert out.log_b is not None
        assert out.log_a_stay is not None
        assert out.log_a_adv is not None

        # ---------------------------
        # 3. Text-aligned reconstruction loss
        # ---------------------------
        y_t = y.transpose(1, 2)  # (B, T_mel, n_mels)

        mel_recon = self.spec_decoder(
            out.aligned_feats,
            cond,
            spec_mask,
        )

        mel_recon_loss = F.l1_loss(
            mel_recon * spec_mask.unsqueeze(-1),
            y_t * spec_mask.unsqueeze(-1),
            reduction="sum",
        )
        mel_recon_loss = mel_recon_loss / (spec_mask.sum() * n_mels).clamp_min(1.0)

        # ---------------------------
        # 4. Duration loss
        # ---------------------------
        loss_dur = self.dur_predictor(
            x=out.h_text,
            x_mask=text_mask,
            dur_target=out.dur.unsqueeze(1).float(),
            cond=cond.unsqueeze(-1),
            reverse=False,
        )
        dur_loss = loss_dur.sum() / text_mask.sum().clamp_min(1.0)

        # ---------------------------
        # 5. Forward alignment energy loss
        # ---------------------------
        # With locally normalized / pairwise unary potentials, this is not a
        # literal acoustic HMM negative log-likelihood. It is a forward-sum
        # energy objective over valid monotone paths.
        align_forward_loss = -out.norm_log_z.mean()

        # ---------------------------
        # 6. Optional diagonal prior loss
        # ---------------------------
        align_diag_loss = torch.zeros((), device=x.device)

        if compute_diagonal_loss:
            align_diag_loss = compute_beta_binomial_loss(
                gamma_posterior=out.soft_gamma,
                y_lengths=y_lengths,
                x_lengths=x_lengths,
                omega=1.0,
            )

        # ---------------------------
        # 7. Optional Viterbi hardening losses
        # ---------------------------
        align_viterbi_kl_loss = torch.zeros((), device=x.device)
        align_viterbi_ot_loss = torch.zeros((), device=x.device)

        if compute_viterbi_loss and out.hard_gamma is not None:
            align_viterbi_kl_loss = compute_viterbi_kl_loss(
                log_gamma=out.log_gamma,
                viterbi_attn=out.hard_gamma,
                spec_mask=out.spec_mask,
            )

            align_viterbi_ot_loss = compute_viterbi_ot_loss(
                log_gamma=out.log_gamma,
                viterbi_attn=out.hard_gamma,
                text_mask=text_mask_bool,
                spec_mask=out.spec_mask,
            )

        return SynthesizerForwardOutput(
            dur=out.dur,
            attn=out.soft_gamma,
            hard_gamma=out.hard_gamma,
            hard_dur=out.hard_dur,
            h_text=out.h_text,
            h_spec=out.h_spec,
            log_alpha=out.log_alpha,
            log_beta=out.log_beta,
            log_gamma=out.log_gamma,
            raw_log_z=out.raw_log_z,
            norm_log_z=out.norm_log_z,
            raw_emission=out.raw_evidence,
            masked_raw_emission=out.masked_raw_evidence,
            log_b=out.log_b,
            log_a_stay=out.log_a_stay,
            log_a_adv=out.log_a_adv,
            viterbi_path=out.viterbi_path,
            viterbi_logp=out.viterbi_logp,
            mel_recon=mel_recon,
            dur_loss=dur_loss,
            mel_recon_loss=mel_recon_loss,
            align_forward_loss=align_forward_loss,
            align_diag_loss=align_diag_loss,
            align_viterbi_kl_loss=align_viterbi_kl_loss,
            align_viterbi_ot_loss=align_viterbi_ot_loss,
        )

    @override
    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,  # (B, T_text)
        x_lengths: torch.Tensor,  # (B,)
        cond: torch.Tensor,  # (B, D_cond)
        noise_scale: float = 0.667,
    ) -> SynthesizerInferenceOutput:
        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).float()

        h_text = self.text_embedder(
            text_token_ids=x,
            text_mask=text_mask,
        )

        out = self.compute_aligned_feats(
            h_text=h_text,
            text_mask=text_mask,
            cond=cond,
            h_spec=None,
            spec_mask=None,
            noise_scale=noise_scale,
            compute_hard_path=False,
            to_hard_aligned_feats=False,
        )

        assert out.hard_gamma is not None

        mel_hat = self.spec_decoder(
            out.aligned_feats,
            cond,
            out.spec_mask,
        )

        wav_hat = None
        if self.vocoder is not None:
            wav_hat = self.mel2wav(mel_hat)

        return SynthesizerInferenceOutput(
            dur=out.dur,
            attn=out.hard_gamma,
            mel_hat=mel_hat,
            wav_hat=wav_hat,
        )

    def compute_aligned_feats(
        self,
        h_text: torch.Tensor,
        text_mask: torch.Tensor,
        cond: torch.Tensor,
        h_spec: torch.Tensor | None = None,
        spec_mask: torch.Tensor | None = None,
        noise_scale: float = 0.667,
        compute_hard_path: bool = False,
        to_hard_aligned_feats: bool = False,
    ) -> AlignedFeaturesOutput:
        """
        Training:
            h_spec is provided.
            Use pairwise CRF aligner to compute soft/hard monotonic alignment.

        Inference:
            h_spec is None.
            Use duration predictor to expand h_text.
        """
        B, _, T_text_max = h_text.shape
        device = h_text.device
        dtype = h_text.dtype

        if text_mask.dim() == 2:
            text_mask = text_mask.unsqueeze(1)
        text_mask = text_mask.to(device=device, dtype=dtype)
        text_mask_bool = text_mask.bool()

        log_alpha = log_beta = log_gamma = raw_log_z = norm_log_z = None
        raw_emmision = masked_raw_emission = None
        log_b = log_a_stay = log_a_adv = None
        viterbi_path = viterbi_logp = None
        soft_gamma = hard_attn = hard_dur = None

        # ------------------------------------------------------------------
        # Training path: use speech encoder outputs and pairwise CRF aligner.
        # ------------------------------------------------------------------
        if h_spec is not None:
            assert spec_mask is not None
            assert self.crf_aligner is not None

            if spec_mask.dim() == 3:
                spec_mask_2d = spec_mask.squeeze(1)
            else:
                spec_mask_2d = spec_mask

            spec_mask_2d = spec_mask_2d.to(device=device, dtype=dtype)
            y_lengths = spec_mask_2d.sum(dim=1).long()

            return_hard = compute_hard_path or to_hard_aligned_feats

            if return_hard:
                (
                    soft_gamma,
                    log_alpha,
                    log_beta,
                    log_gamma,
                    raw_log_z,
                    norm_log_z,
                    raw_emmision,
                    masked_raw_emission,
                    log_b,
                    log_a_stay,
                    log_a_adv,
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
                    raw_emmision,
                    masked_raw_emission,
                    log_b,
                    log_a_stay,
                    log_a_adv,
                    soft_dur,
                ) = self.crf_aligner(
                    h_spec=h_spec,
                    spec_mask=spec_mask_2d.unsqueeze(1),
                    h_text=h_text,
                    text_mask=text_mask,
                    cond=cond,
                    return_hard=False,
                )

            dur = soft_dur

            if to_hard_aligned_feats:
                assert hard_attn is not None
                text_attn = hard_attn
            else:
                assert soft_gamma is not None
                text_attn = soft_gamma

            aligned_feats = torch.bmm(
                text_attn,
                # text_attn.detach(),
                h_text.transpose(1, 2),
            )
            aligned_feats = aligned_feats * spec_mask_2d.unsqueeze(-1)

            spec_mask_out = spec_mask_2d

        # ------------------------------------------------------------------
        # Inference path: use duration predictor.
        # ------------------------------------------------------------------
        else:
            logw = self.dur_predictor(
                x=h_text,
                x_mask=text_mask,
                dur_target=None,
                cond=cond.unsqueeze(-1),
                reverse=True,
                noise_scale=noise_scale,
            )

            w = torch.exp(logw) * text_mask
            dur = torch.ceil(w).squeeze(1).long()

            dur = dur.masked_fill(~text_mask_bool.squeeze(1), 0)

            y_lengths = dur.sum(dim=1)

            empty = y_lengths == 0
            if empty.any():
                dur[empty, 0] = 1
                y_lengths = dur.sum(dim=1)

            hard_dur = dur

            T_mel_max = max(1, int(y_lengths.max().item()))
            hard_attn = torch.zeros(
                B,
                T_mel_max,
                T_text_max,
                device=device,
                dtype=dtype,
            )

            aligned_feats_list: list[torch.Tensor] = []
            h_text_t = h_text.transpose(1, 2)  # (B, T_text, C)

            for b in range(B):
                feat_b = torch.repeat_interleave(
                    h_text_t[b],
                    dur[b],
                    dim=0,
                )

                T_b = int(y_lengths[b].item())
                if T_b > 0:
                    m_idx = torch.arange(T_b, device=device)
                    cum_dur = torch.cumsum(dur[b], dim=0)
                    t_idx = torch.bucketize(m_idx, cum_dur, right=True)
                    t_idx = t_idx.clamp_max(T_text_max - 1)
                    hard_attn[b, m_idx, t_idx] = 1.0

                pad_len = T_mel_max - feat_b.shape[0]
                if pad_len > 0:
                    feat_b = F.pad(feat_b, (0, 0, 0, pad_len))

                aligned_feats_list.append(feat_b)

            aligned_feats = torch.stack(aligned_feats_list, dim=0)

            idx = torch.arange(T_mel_max, device=device).unsqueeze(0)
            spec_mask_out = (idx < y_lengths.unsqueeze(1)).to(dtype=dtype)
            aligned_feats = aligned_feats * spec_mask_out.unsqueeze(-1)

        return AlignedFeaturesOutput(
            aligned_feats=aligned_feats,
            dur=dur,
            spec_mask=spec_mask_out,
            y_lengths=y_lengths,
            h_text=h_text,
            h_spec=h_spec,
            soft_gamma=soft_gamma,
            hard_gamma=hard_attn,
            hard_dur=hard_dur,
            log_alpha=log_alpha,
            log_beta=log_beta,
            log_gamma=log_gamma,
            raw_log_z=raw_log_z,
            norm_log_z=norm_log_z,
            raw_evidence=raw_emmision,
            masked_raw_evidence=masked_raw_emission,
            log_b=log_b,
            log_a_stay=log_a_stay,
            log_a_adv=log_a_adv,
            viterbi_path=viterbi_path,
            viterbi_logp=viterbi_logp,
        )

    def _encode_speech(
        self,
        y: torch.Tensor,
        spec_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility wrapper for SpecEncoder.

        Preferred new SpecEncoder API:
            h_spec = spec_encoder(y, mask, cond)

        Backward-compatible old API:
            spec_logits, spec_probs, h_spec = spec_encoder(y, mask, cond)

        The pairwise aligner uses only h_spec.
        """
        assert self.spec_encoder is not None

        enc_out = self.spec_encoder(
            y,
            spec_mask.unsqueeze(1),
            cond=cond,
        )

        if isinstance(enc_out, tuple):
            # Old SpecEncoder returned (logits, probs, h_spec).
            return enc_out[-1]

        return enc_out

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

    @torch.no_grad()
    def inference_from_ref_audio(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        ref_waveform: torch.Tensor,
        noise_scale: float = 0.667,
    ) -> SynthesizerInferenceOutput:
        cond = self.compute_speaker_embeddings(ref_waveform).squeeze(1)
        return self.inference(
            x=x,
            x_lengths=x_lengths,
            cond=cond,
            noise_scale=noise_scale,
        )
