from __future__ import annotations

from typing import Any, NamedTuple, override

import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .diffusion.sampler import KarrasSampler
from .diffusion.score_estimator import KarrasScoreEstimator
from .monotonic_tts import MonotonicTTSSynthesizer, SynthesizerInferenceOutput
from .utils.fix_len_compatibility import fix_len_compatibility


class DiffusionForwardOutput(NamedTuple):
    mel_hat: torch.Tensor  # (B, n_mels, T_mel) - Denoised prediction
    mel_gt: torch.Tensor  # (B, n_mels, T_mel) - Ground truth
    sigma: torch.Tensor  # (B,) - Sampled noise levels
    mask: torch.Tensor  # (B, 1, T_mel) - Sequence mask
    # The value just before loss_weight multiplication: (denoised - y_gt)^2 * mask
    mel_loss_unweighted: torch.Tensor  # (B, n_mels, T_mel)


class KarrasTTSSynthesizer(BaseModel):
    """
    Integrated TTS model that combines Stage 1 (Alignment & Coarse Mel)
    with Stage 2 (EDM Diffusion Refinement).
    """

    def __init__(
        self,
        syn: MonotonicTTSSynthesizer,
        estimator: KarrasScoreEstimator,
        stochastic_sampling: bool = True,
        num_unet_downsample: int = 2,
        unet_out_size: int = 172,
    ) -> None:
        super().__init__()
        self.syn_backbone = syn
        self.estimator = estimator
        self.stochastic_sampling = stochastic_sampling
        self.num_unet_downsample = num_unet_downsample
        self.unet_out_size = unet_out_size

        self.sampler = KarrasSampler(estimator)

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
        # 1. Get Aligned Features from Stage 1 (detached)
        with torch.no_grad():
            out = self.syn_backbone.compute_aligned_feats(
                x=x,
                x_lengths=x_lengths,
                cond=cond,
                y=y,
                y_lengths=y_lengths,
                to_hard_aligened_feats=True,
                text_only_aligned_feats=True,
            )

            # aligned_feats: (B, T_mel, C) -> (B, C, T_mel)
            aligned_feats = out.aligned_feats.transpose(1, 2)

            # spec_mask: (B, T_mel) -> (B, 1, T_mel)
            mask = out.spec_mask.unsqueeze(1)

            # y: (B, n_mels, T_mel)
            y_gt = out.y
            assert y_gt is not None

        y_gt, aligned_feats, mask = self._get_random_segments(
            y_gt=y_gt,
            mask=mask,
            aligned_feats=aligned_feats,
        )

        # >> Diffusion Training Step (Sigma sampling is handled internally)
        y_noisy, sigma = self.estimator.forward_diffusion(
            x=y_gt,
            sigma=None,
        )

        # >> Predict Denoised Mel
        # Note: loss_weight calculation is delegated to the training script

        denoised = self.estimator.reverse_diffusion(
            x=y_noisy,
            sigma=sigma,
            mask=mask,
            text=aligned_feats,
            spk=cond,
            guidance_scale=1.0,
            text_cond_drop_prob=text_cond_drop_prob,
            spk_cond_drop_prob=spk_cond_drop_prob,
            text_cond_mask_ratio=text_cond_mask_ratio,
            spk_cond_mask_ratio=spk_cond_mask_ratio,
        )

        # Calculate unweighted squared error (just before loss_weight)
        # Apply mask to ensure padding doesn't contribute to loss
        mel_loss_unweighted = ((denoised - y_gt) ** 2) * mask

        return DiffusionForwardOutput(
            mel_hat=denoised,
            mel_gt=y_gt,
            sigma=sigma,
            mask=mask,
            mel_loss_unweighted=mel_loss_unweighted,
        )

    @override
    @torch.no_grad()
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
        out = self.syn_backbone.compute_aligned_feats(
            x,
            x_lengths,
            cond,
            noise_scale=noise_scale,
            text_only_aligned_feats=True,
        )
        assert out.hard_attn is not None

        # aligned_feats: (B, T_mel, C) -> (B, C, T_mel)
        aligned_feats = out.aligned_feats.transpose(1, 2)

        # spec_mask: (B, T_mel) -> (B, 1, T_mel)
        mask = out.spec_mask.unsqueeze(1)

        aligned_feats, mask, _ = self._pad_length_compatible(
            aligned_feats=aligned_feats,
            mask=mask,
        )

        # 2. Stage 2 Diffusion Refinement
        refined_mel = self.sampler.sample(
            text=aligned_feats,
            mask=mask,
            spk=cond,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            cfg_mode=cfg_mode,
            stochastic=self.stochastic_sampling,
            **sampler_kwargs,
        )  # (B, n_mels, T_mel)

        # 3. Vocoding (using Stage 1's vocoder)
        wav_hat = self.syn_backbone.mel2wav(refined_mel)

        return SynthesizerInferenceOutput(
            dur=out.dur,
            attn=out.hard_attn,
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

    def _get_random_segments(
        self,
        y_gt: torch.Tensor,
        mask: torch.Tensor,
        aligned_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_size = self.unet_out_size
        batch_size = y_gt.size(0)
        device = y_gt.device

        lengths = mask.squeeze(1).long().sum(dim=-1)  # (B,)
        max_offset = (lengths - out_size).clamp(min=0)  # (B,)

        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            max_off = int(max_offset[i].item())
            offsets[i] = torch.randint(0, max_off + 1, (1,), device=device) if max_off > 0 else 0

        x_cut = y_gt.new_zeros(batch_size, y_gt.size(1), out_size)
        text_cut = aligned_feats.new_zeros(batch_size, aligned_feats.size(1), out_size)
        cut_lengths: list[int] = []

        for i in range(batch_size):
            seq_len = int(lengths[i].item())
            start = int(offsets[i].item())

            cut_len = min(seq_len, out_size)
            end = start + cut_len

            x_cut[i, :, :cut_len] = y_gt[i, :, start:end]
            text_cut[i, :, :cut_len] = aligned_feats[i, :, start:end]
            cut_lengths.append(cut_len)

        cut_lengths_t = torch.tensor(cut_lengths, dtype=torch.long, device=device)
        time_idx = torch.arange(out_size, device=device).unsqueeze(0)
        mask_cut = (time_idx < cut_lengths_t.unsqueeze(1)).unsqueeze(1).to(mask.dtype)

        return x_cut, text_cut, mask_cut

    def _pad_length_compatible(
        self,
        aligned_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        orig_len = aligned_feats.size(-1)
        compat_len = fix_len_compatibility(orig_len, self.num_unet_downsample)

        if compat_len == orig_len:
            return aligned_feats, mask, orig_len

        pad_len = compat_len - orig_len

        aligned_feats = F.pad(aligned_feats, (0, pad_len))
        mask = F.pad(mask, (0, pad_len))

        return aligned_feats, mask, orig_len
