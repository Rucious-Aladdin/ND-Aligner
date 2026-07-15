from typing import NamedTuple

import torch


class LossValues(NamedTuple):
    recon: float
    crf: float
    diag: float
    viterbi_kl: float


class TrainItem(NamedTuple):
    audio_path: str  # preprocessed wav file path
    spk_path: str  # spk-embedding pt file path
    text: str  # script file

    # metadata for eval/logging
    utt_id: str = ""
    spk_id: str = ""
    dataset: str = ""


class TrainDatasetInstance(NamedTuple):
    text: torch.Tensor  # (T_text,)

    # Alignment input feature.
    #   mel:     (n_mels, T)
    #   linspec: (n_fft // 2 + 1, T)
    spec: torch.Tensor

    # Reconstruction target feature. Always mel.
    #   (n_mels, T)
    recon_spec: torch.Tensor

    cond: torch.Tensor  # (cond_dim,)

    # metadata
    wav_path: str
    script: str
    utt_id: str
    spk_id: str
    dataset: str


class TrainBatch(NamedTuple):
    text: torch.Tensor  # (B, T_text)
    text_lengths: torch.Tensor  # (B,)

    # Alignment input feature.
    #   mel:     (B, n_mels, T)
    #   linspec: (B, n_fft // 2 + 1, T)
    spec: torch.Tensor
    spec_lengths: torch.Tensor

    # Reconstruction target. Always mel.
    #   (B, n_mels, T)
    recon_spec: torch.Tensor
    recon_spec_lengths: torch.Tensor

    cond: torch.Tensor  # (B, cond_dim)

    scripts: list[str]
    wav_paths: list[str]
    utt_ids: list[str]
    spk_ids: list[str]
    datasets: list[str]
