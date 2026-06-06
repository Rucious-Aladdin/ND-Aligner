from __future__ import annotations

from typing import Any, NamedTuple, override, cast

import torch

from .base_model import BaseModel
from .diffusion.karras_diffusion import KarrasDiffusionModel

from .monotonic_tts import MonotonicTTSSynthesizer, SynthesizerInferenceOutput
from tts.models.utils.sequence_mask import sequence_mask


class DiffusionForwardOutput(NamedTuple):
    mel_hat: torch.Tensor  # (B, n_mels, T_mel) - Denoised prediction
    mel_gt: torch.Tensor  # (B, n_mels, T_mel) - Ground truth
    sigma: torch.Tensor  # (B,) - Sampled noise levels
    mask: torch.Tensor  # (B, 1, T_mel) - Sequence mask
    # The value just before loss_weight multiplication: (denoised - y_gt)^2 * mask
    mel_loss_unweighted: torch.Tensor  # (B, n_mels, T_mel)

    soft_attn: torch.Tensor | None
    hard_attn: torch.Tensor | None
    aligned_texts: torch.Tensor | None


class KarrasTTSSynthesizer(BaseModel):
    """
    Integrated TTS model that combines Stage 1 (Alignment & Coarse Mel)
    with Stage 2 (EDM Diffusion Refinement).
    """

    def __init__(
        self,
        syn: MonotonicTTSSynthesizer,
        diffusion_model: KarrasDiffusionModel,
        stochastic_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.syn_backbone = syn
        self.diffusion_model = diffusion_model
        self.stochastic_sampling = stochastic_sampling

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        cond: torch.Tensor,
        text_cond_drop_prob: float = 0.0,
        spk_cond_drop_prob: float = 0.0,
        text_cond_mask_ratio: float = 0.0,
        spk_cond_mask_ratio: float = 0.0,
    ) -> DiffusionForwardOutput:
        """
        Compute Stage 2 Training Path.
        Stage 1 weights are assumed to be fixed (or detached).
        Loss weighting is delegated to the training script.
        """

        _, _, T_mel = y.shape

        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).to(dtype=y.dtype)

        idx = torch.arange(T_mel, device=y.device).unsqueeze(0)
        spec_mask = (idx < y_lengths.unsqueeze(1)).to(dtype=y.dtype)

        # 1. Get Aligned Features from Stage 1 (detached)
        with torch.no_grad():
            h_text = cast(
                torch.Tensor,
                self.syn_backbone.text_encoder_align(
                    x=x,
                    x_mask=text_mask,
                ),
            )  # (B, C_text, T_text)

            h_spec = self.syn_backbone.spec_encoder(  # type: ignore
                x=y,
                mask=spec_mask,
                cond=cond,
            )  # (B, C_spec, T_mel)

            out = self.syn_backbone.compute_aligned_feats(
                h_text=h_text,
                text_mask=text_mask,
                cond=cond,
                h_spec=h_spec,
                spec_mask=spec_mask,
                compute_hard_path=True,
                to_hard_aligned_feats=True,
            )

            # aligned_feats: (B, T_mel, C) -> (B, C, T_mel)
            aligned_texts = out.aligned_feats.transpose(1, 2)

            # spec_mask: (B, T_mel) -> (B, 1, T_mel)
            mask = out.spec_mask.unsqueeze(1)

        assert out.hard_gamma is not None

        # >> Diffusion Training Step (Sigma sampling is handled internally)
        y_noisy, sigma = self.diffusion_model.forward_diffusion(
            x=y,
            sigma=None,
        )

        # >> Predict Denoised Mel
        # Note: loss_weight calculation is delegated to the training script

        x0_hat = self.diffusion_model(
            x=y_noisy,
            sigma=sigma,
            mask=mask,
            text=aligned_texts,
            spk=cond,
            text_cond_drop_prob=text_cond_drop_prob,
            spk_cond_drop_prob=spk_cond_drop_prob,
            text_cond_mask_ratio=text_cond_mask_ratio,
            spk_cond_mask_ratio=spk_cond_mask_ratio,
            h_text=h_text,
            hard_gamma=out.hard_gamma,
            text_mask=text_mask,
        )

        # Calculate unweighted squared error (just before loss_weight)
        # Apply mask to ensure padding doesn't contribute to loss
        mel_loss_unweighted = ((x0_hat - y) ** 2) * mask

        return DiffusionForwardOutput(
            mel_hat=x0_hat,
            mel_gt=y,
            sigma=sigma,
            mask=mask,
            mel_loss_unweighted=mel_loss_unweighted,
            soft_attn=out.soft_gamma,
            hard_attn=out.hard_gamma,
            aligned_texts=aligned_texts,
        )

    @override
    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        cond: torch.Tensor,
        n_steps: int = 35,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
        noise_scale: float = 0.667,
        **sampler_kwargs: Any,
    ) -> SynthesizerInferenceOutput:
        """
        Full TTS Inference: Stage 1 Alignment -> Stage 2 Diffusion Refinement -> Vocoding.
        """
        # 1. Stage 1 Inference (Predict Duration & Alignment)
        text_mask_bool = sequence_mask(x_lengths, x.size(1))
        text_mask = text_mask_bool.unsqueeze(1).to(dtype=x.dtype)

        # 1. Get Aligned Features from Stage 1 (detached)
        h_text = self.syn_backbone.text_encoder_align(
            x=x,
            x_mask=text_mask,
        )  # (B, C_text, T_text)

        out = self.syn_backbone.compute_aligned_feats(
            h_text=h_text,
            text_mask=text_mask,
            cond=cond,
            noise_scale=noise_scale,
            compute_hard_path=True,
        )
        assert out.hard_gamma is not None

        # aligned_feats: (B, T_mel, C) -> (B, C, T_mel)
        aligned_texts = out.aligned_feats.transpose(1, 2)

        # spec_mask: (B, T_mel) -> (B, 1, T_mel)
        mask = out.spec_mask.unsqueeze(1)

        # 2. Stage 2 Diffusion Refinement
        refined_mel = self.diffusion_model.sample(
            text=aligned_texts,
            mask=mask,
            spk=cond,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            stochastic=self.stochastic_sampling,
            h_text=h_text,
            hard_gamma=out.hard_gamma,
            text_mask=text_mask,
            **sampler_kwargs,
        )  # (B, n_mels, T_mel)

        # 3. Vocoding (using Stage 1's vocoder)
        wav_hat = self.syn_backbone.mel2wav(refined_mel)

        return SynthesizerInferenceOutput(
            dur=out.dur,
            attn=out.hard_gamma,
            mel_hat=refined_mel.transpose(1, 2),  # (B, T_mel, n_mels)
            wav_hat=wav_hat,
        )

    @torch.no_grad()
    def inference_from_ref_audio(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        ref_waveform: torch.Tensor,
        n_steps: int = 35,
        guidance_scale: float | tuple[float, float] = 1.0,
        cfg_mode: str = "base",
        noise_scale: float = 0.667,
        **sampler_kwargs: Any,
    ) -> SynthesizerInferenceOutput:
        """
        Synthesizes speech using a reference audio waveform for speaker conditioning.
        """
        cond = self.syn_backbone.compute_speaker_embeddings(ref_waveform).squeeze(1)
        return self.inference(
            x=x,
            x_lengths=x_lengths,
            cond=cond,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            noise_scale=noise_scale,
            **sampler_kwargs,
        )

    def set_sample_mode(self, stochastic: bool = True):
        self.stochastic_sampling = stochastic
