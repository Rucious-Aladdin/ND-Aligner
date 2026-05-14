# refactoring https://github.com/jik876/hifi-gan/blob/master/meldataset.py

import warnings
from functools import cached_property
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from tts.config.stage1.data_config import AudioConfig


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: int = 1,
    clip_val: float = 1e-5,
):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class MelSpecExtractor(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_size = config.hop_length
        self.win_size = config.win_length
        self.fmin = config.f_min
        self.fmax = config.f_max

        mel = librosa_mel_fn(
            sr=config.sr,
            n_fft=self.n_fft,
            n_mels=config.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("hann_window", torch.hann_window(self.win_size))
        self.mel_basis: torch.Tensor
        self.hann_window: torch.Tensor

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(0)  # (Time,) -> (1, Time) 으로 배치 차원 추가

        if torch.min(y) < -1.0:
            print("Warning: min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("Warning: max value is ", torch.max(y))

        pad_size = int((self.n_fft - self.hop_size) / 2)
        y = F.pad(y.unsqueeze(1), (pad_size, pad_size), mode="reflect")
        y = y.squeeze(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=False,
            )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_torch(spec)
        return spec

    @cached_property
    def pad_value(self) -> float:
        dummy_zero = torch.zeros(1)
        return dynamic_range_compression_torch(dummy_zero).item()
