from __future__ import annotations

from typing import override

import numpy as np
import torch
import torch.nn as nn
from resemblyzer import VoiceEncoder, preprocess_wav


class ResemblyzerSpeakerEncoder(nn.Module):
    def __init__(
        self,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.embedding_dim = 256

        self.spk_encoder = VoiceEncoder(
            device=device,  # type: ignore
            verbose=False,
        )
        self.spk_encoder.eval()

        for parameter in self.spk_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    @override
    def forward(
        self,
        waveform_16k: torch.Tensor,
        wav_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            waveforms:
                (T,), (B, T), or (B, 1, T)

            lengths:
                Valid waveform lengths in samples, shape (B,).
                Recommended when waveforms are zero-padded.

        Returns:
            Speaker embeddings of shape (B, 256).
        """
        output_device = waveform_16k.device

        if waveform_16k.ndim == 1:
            waveform_16k = waveform_16k.unsqueeze(0)

        elif waveform_16k.ndim == 3:
            waveform_16k = waveform_16k.squeeze(1)

        batch_size, max_length = waveform_16k.shape

        if wav_lens is None:
            wav_lens = torch.full(
                (batch_size,),
                max_length,
                dtype=torch.long,
                device=waveform_16k.device,
            )
        else:
            wav_lens = wav_lens.reshape(-1)

        # VoiceEncoder stores its inference device separately.
        self.spk_encoder.device = next(self.spk_encoder.parameters()).device

        embeddings: list[np.ndarray] = []

        for batch_idx in range(batch_size):
            wav_length = int(wav_lens[batch_idx].item())
            wav_np = waveform_16k[batch_idx, :wav_length].detach().float().cpu().numpy()

            wav_np = preprocess_wav(wav_np)
            embedding = self.spk_encoder.embed_utterance(wav_np)
            embeddings.append(embedding)  # type: ignore

        return torch.from_numpy(np.stack(embeddings, axis=0)).to(
            device=output_device,
            dtype=torch.float32,
        )
