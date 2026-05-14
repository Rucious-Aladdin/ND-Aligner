import os
from typing import override

import torch
import torch.nn as nn
import torchaudio

# Set audio backend for torchaudio if needed by SpeechBrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # pyright: ignore

# Import the actual SpeakerEncoder from speechbrain
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEncoder(nn.Module):
    """
    Wrapper around SpeechBrain's ECAPA-TDNN for extracting speaker embeddings.
    Expects 16kHz mono audio waveform as input.
    Outputs a 192-dimensional speaker embedding.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Ensure that the speechbrain model is only loaded once
        # and moved to the correct device
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "pretrained_ecapa"
                ),
                run_opts={"device": self.device},
            )
            assert self.classifier is not None
            self.classifier.eval().to(self.device)
            print(f"SpeechBrain ECAPA-TDNN SpeakerEncoder loaded on {self.device}.")
        except Exception as e:
            print(f"Error loading SpeechBrain ECAPA-TDNN: {e}")
            raise RuntimeError("Failed to load SpeechBrain SpeakerEncoder.")

    @override
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts speaker embedding from a batch of waveforms.
        Args:
            waveform (torch.Tensor): Input waveform tensor (B, T). Must be 16kHz.
        Returns:
            torch.Tensor: Speaker embedding tensor (B, 192).
        """
        # SpeechBrain expects waveform in (B, T)
        # It also returns embeddings as (B, embedding_dim, 1), so squeeze the last dim
        with torch.no_grad():
            assert self.classifier is not None
            embeddings = self.classifier.encode_batch(waveform.to(self.device))
        return embeddings.squeeze(2)  # (B, 192, 1) -> (B, 192)
