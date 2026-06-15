from typing import NamedTuple

import torch


class LossValues(NamedTuple):
    recon: float
    crf: float
    diag: float
    viterbi_kl: float
    viterbi_ot: float


class TTSItem(NamedTuple):
    audio_path: str  # preprocessed wav file path
    spk_path: str  # spk-embedding pt file path
    text: str  # script file

    # metadata for eval/logging
    utt_id: str = ""
    spk_id: str = ""
    dataset: str = ""


class TTSDatasetInstance(NamedTuple):
    text: torch.Tensor  # (T_text,)
    spec: torch.Tensor  # (n_mels, T_mel)
    cond: torch.Tensor  # (cond_dim,)

    # metadata
    wav_path: str
    script: str
    utt_id: str
    spk_id: str
    dataset: str


class TTSBatch(NamedTuple):
    text: torch.Tensor  # Text token indices (B, T_text)
    text_lengths: torch.Tensor  # Lengths of text (B,)

    spec: torch.Tensor  # Ground-truth spectrogram (B, n_mels, T_mel)
    spec_lengths: torch.Tensor  # Lengths of speech (B,)

    cond: torch.Tensor  # Condition vector (B, cond_dim)

    # Metadata for logging/eval
    scripts: list[str]  # Raw text scripts for logging
    wav_paths: list[str]
    utt_ids: list[str]
    spk_ids: list[str]
    datasets: list[str]
