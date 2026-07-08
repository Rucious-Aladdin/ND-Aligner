from functools import cached_property
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSpecExtractor(nn.Module):
    """
    Linear-frequency magnitude spectrogram extractor.

    This is intended to match the behavior of:

        spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False)

    from VITS-style preprocessing.

    Output:
        spec: (B, n_fft // 2 + 1, T_frame)

    No mel projection.
    Optional log1p compression.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        fmax: float | None = None,
        log_compress: bool = True,
        log_scale: float = 1.0,
        magnitude_eps: float = 1e-6,
    ):
        super().__init__()

        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_length)
        self.win_length = int(win_length)
        self.fmax = fmax

        self.log_compress = bool(log_compress)
        self.log_scale = float(log_scale)
        self.magnitude_eps = float(magnitude_eps)

        if self.log_scale <= 0.0:
            raise ValueError(f"log_scale must be positive, got {self.log_scale}")

        if self.magnitude_eps < 0.0:
            raise ValueError(f"magnitude_eps must be non-negative, got {self.magnitude_eps}")

        self.register_buffer("hann_window", torch.hann_window(self.win_length))
        self.hann_window: torch.Tensor

        freqs = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32) * (
            float(self.sr) / float(self.n_fft)
        )

        if self.fmax is None:
            freq_mask = torch.ones_like(freqs, dtype=torch.bool)
        else:
            if self.fmax < 0:
                raise ValueError(f"fmax must be non-negative or None, got {self.fmax}")
            freq_mask = freqs <= float(self.fmax)

        self.register_buffer("freq_mask", freq_mask)
        self.freq_mask: torch.Tensor

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y:
                waveform, (T,) or (B, T)

        Returns:
            spec:
                linear magnitude spectrogram, (B, n_fft // 2 + 1, T_frame)
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)

        if y.dim() != 2:
            raise ValueError(f"y must have shape (T,) or (B, T), got {tuple(y.shape)}")

        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        window = self.hann_window.to(dtype=y.dtype, device=y.device)

        pad_size = int((self.n_fft - self.hop_size) / 2)
        y = F.pad(
            y.unsqueeze(1),
            (pad_size, pad_size),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec_complex = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B, F, T), complex

        spec = torch.sqrt(
            spec_complex.real.pow(2) + spec_complex.imag.pow(2) + self.magnitude_eps
        )  # (B, F, T)

        if self.log_compress:
            spec = torch.log1p(self.log_scale * spec)

        freq_mask = self.freq_mask.to(device=spec.device)
        spec = spec.masked_fill(~freq_mask.view(1, -1, 1), 0.0)

        return spec

    @cached_property
    def pad_value(self) -> float:
        mag_floor = self.magnitude_eps**0.5

        if self.log_compress:
            return float(torch.log1p(torch.tensor(self.log_scale * mag_floor)).item())

        return float(mag_floor)
