import matplotlib.pyplot as plt
import torch
from einops import asnumpy
from matplotlib.figure import Figure


def plot_spectrogram(spectrogram: torch.Tensor) -> Figure:
    """Plots a mel-spectrogram with width proportional to its aspect ratio."""
    # spectrogram: (n_mels, T)
    n_mels, T = spectrogram.shape

    # Fixed height of 3 inches
    height = 3
    # Width is calculated to maintain the data's natural aspect ratio
    # We add a small constant for the colorbar space
    width = max(6, (T / n_mels) * height + 1.5)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(asnumpy(spectrogram), aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
