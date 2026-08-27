import matplotlib.pyplot as plt
import torch
from einops import asnumpy
from matplotlib.figure import Figure


def plot_alignment(
    alignment: torch.Tensor,
    tokens: list[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Plots an alignment matrix with optional phoneme/token labels on the y-axis.

    Args:
        alignment: (T_mel, T_text)
        tokens: Optional list of length T_text. If provided, shown as y-axis labels.
        vmin: Minimum color scale value.
        vmax: Maximum color scale value.

    Returns:
        Matplotlib Figure.
    """
    T_mel, T_text = alignment.shape

    if tokens is not None and len(tokens) != T_text:
        raise ValueError(
            f"tokens length must match T_text={T_text}, but got len(tokens)={len(tokens)}."
        )

    height = max(5, T_text * 0.25 if tokens is not None else 5)
    width = max(6, (T_mel / T_text) * height + 1.5)

    fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(
        asnumpy(alignment.transpose(0, 1)),  # (T_text, T_mel)
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Mel Frame Index")
    ax.set_ylabel("Phoneme Index" if tokens is None else "Phoneme")

    if tokens is not None:
        ax.set_yticks(range(T_text))
        ax.set_yticklabels(tokens)
    else:
        ax.set_yticks(range(T_text))

    fig.tight_layout()
    return fig
