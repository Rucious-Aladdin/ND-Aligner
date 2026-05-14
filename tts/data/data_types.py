from typing import NamedTuple

import torch


class LossWeights(NamedTuple):
    dur: float
    mel_recon: float
    align_forward: float
    align_diag: float
    align_viterbi_kl: float
    align_viterbi_ot: float


class LossValues(NamedTuple):
    dur: float
    mel_recon: float
    align_forward: float
    align_diag: float
    align_viterbi_kl: float
    align_viterbi_ot: float


class TTSItem(NamedTuple):
    audio_path: str  # preprocessed wav file path
    spk_path: str  # spk-embedding pt file path
    text: str  # script file


class TTSBatch(NamedTuple):
    text: torch.Tensor  # Text token indices (B, T_text)
    text_lengths: torch.Tensor  # Lengths of text (B,)

    spec: torch.Tensor  # Ground-truth spectrogram (B, n_mels, T_mel)
    spec_lengths: torch.Tensor  # Lengths of speech (B,)

    cond: torch.Tensor  # Condition vector (B, cond_dim)
    scripts: list[str]  # Raw text scripts for logging
