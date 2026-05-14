import matplotlib.pyplot as plt
import torch


def visualize_2d_map(
    attn: torch.Tensor,
    title: str = "",
    save_path: str = "temp_map.png",
) -> None:
    """
    attn: (T_speech, T_text)
    visualize with speech on x-axis
    """

    if attn.dim() != 2:
        raise ValueError(f"Expected attn shape (T_speech, T_text), got {tuple(attn.shape)}")

    # (T_speech, T_text) -> (T_text, T_speech)
    attn_map = attn.transpose(0, 1).detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(attn_map, aspect="auto", origin="lower")
    plt.colorbar()

    plt.xlabel("Speech frame")
    plt.ylabel("Text token")
    plt.title(f"{title}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved at: {save_path}")
    plt.close()
