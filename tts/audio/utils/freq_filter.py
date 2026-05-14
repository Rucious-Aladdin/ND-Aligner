import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    x: NDArray[np.float32],
    sr: int,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> NDArray[np.float32]:
    if x.ndim == 2:
        x = x.squeeze()

    nyq = sr * 0.5
    if not (0 < low_freq < high_freq < nyq):
        raise ValueError(f"Require 0 < low_freq < high_freq < Nyquist({nyq})")

    sos = butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sr,
        output="sos",
    )
    return sosfiltfilt(sos, x.astype(np.float32)).astype(np.float32)


def lowpass_filter(
    x: NDArray[np.float32],
    sr: int,
    cutoff_freq: float,
    order: int = 4,
) -> NDArray[np.float32]:
    if x.ndim == 2:
        x = x.squeeze()

    nyq = sr * 0.5
    if not (0 < cutoff_freq < nyq):
        raise ValueError(f"Require 0 < cutoff_freq < Nyquist({nyq})")

    sos = butter(
        order,
        cutoff_freq,
        btype="lowpass",
        fs=sr,
        output="sos",
    )
    return sosfiltfilt(sos, x.astype(np.float32)).astype(np.float32)


def highpass_filter(
    x: NDArray[np.float32],
    sr: int,
    cutoff_freq: float,
    order: int = 4,
) -> NDArray[np.float32]:
    if x.ndim == 2:
        x = x.squeeze()

    nyq = sr * 0.5
    if not (0 < cutoff_freq < nyq):
        raise ValueError(f"Require 0 < cutoff_freq < Nyquist({nyq})")

    sos = butter(
        order,
        cutoff_freq,
        btype="highpass",
        fs=sr,
        output="sos",
    )
    return sosfiltfilt(sos, x.astype(np.float32)).astype(np.float32)
