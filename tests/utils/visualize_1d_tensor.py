import matplotlib.pyplot as plt
import torch


def visualize_1d_tensor(
    tensor: torch.Tensor,
    batch_idx: int = 0,
    title: str = "1D Visualization",
    xlabel: str = "X-axis (Time/Frames)",
    ylabel: str = "Amplitude",
    color: str = "blue",
    save_path: str = "temp_tensor_1d.png",
) -> None:
    """
    임의의 1D, 2D 또는 3D 텐서(데이터가 1D인 경우)를 시각화합니다.

    Args:
        tensor (torch.Tensor): (T,), (B, T) 또는 (B, 1, T) 형태의 텐서
        batch_idx (int): 텐서가 2D 또는 3D일 경우 시각화할 배치의 인덱스
        title (str): 그래프 제목
        xlabel (str): X축 라벨
        ylabel (str): Y축 라벨
        color (str): 선 색상
        save_path (str): 저장할 파일 경로
    """
    if tensor.dim() not in [1, 2, 3]:
        raise ValueError(
            f"Expected 1D, 2D or 3D tensor, got {tensor.dim()}D tensor with shape {tuple(tensor.shape)}"
        )

    if tensor.dim() == 3:
        if tensor.size(1) != 1:
            raise ValueError(
                f"For 1D visualization of 3D tensor, dim 1 must be 1, got {tensor.size(1)}"
            )
        B = tensor.size(0)
        if not (0 <= batch_idx < B):
            raise ValueError(f"batch_idx must be in [0, {B-1}], got {batch_idx}")
        data_1d = tensor[batch_idx, 0].detach().cpu().numpy()
        title_suffix = f" (batch_idx={batch_idx})"
    elif tensor.dim() == 2:
        B = tensor.size(0) # pyright: ignore[reportConstantRedefinition]
        if not (0 <= batch_idx < B):
            raise ValueError(f"batch_idx must be in [0, {B-1}], got {batch_idx}")
        data_1d = tensor[batch_idx].detach().cpu().numpy()
        title_suffix = f" (batch_idx={batch_idx})"
    else:
        data_1d = tensor.detach().cpu().numpy()
        title_suffix = ""

    plt.figure(figsize=(12, 4))
    plt.plot(data_1d, color=color, linewidth=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}{title_suffix}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved at: {save_path}")
    plt.close()
