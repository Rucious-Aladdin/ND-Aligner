import matplotlib.pyplot as plt
import torch


def visualize_2d_tensor(
    tensor: torch.Tensor,
    batch_idx: int = 0,
    title: str = "2D Visualization",
    xlabel: str = "X-axis (Time/Frames)",
    ylabel: str = "Y-axis (Channels/Tokens)",
    cmap: str = "viridis",
    origin: str = "lower",
    save_path: str = "temp_tensor.png",
) -> None:
    """
    임의의 2D 또는 3D 텐서를 시각화합니다.

    Args:
        tensor (torch.Tensor): (H, W) 또는 (B, H, W) 형태의 텐서
        batch_idx (int): 텐서가 3D일 경우 시각화할 배치의 인덱스
        title (str): 그래프 제목
        xlabel (str): X축 라벨
        ylabel (str): Y축 라벨
        cmap (str): matplotlib 컬러맵 (예: 'viridis', 'magma', 'gray')
        origin (str): 'lower' (Mel, 음성 등 y축이 아래에서 시작) 또는 'upper' (일반 이미지)
        save_path (str): 저장할 파일 경로
    """
    if tensor.dim() not in [2, 3]:
        raise ValueError(
            f"Expected 2D or 3D tensor, got {tensor.dim()}D tensor with shape {tuple(tensor.shape)}"
        )

    if tensor.dim() == 3:
        B = tensor.size(0)
        if not (0 <= batch_idx < B):
            raise ValueError(f"batch_idx must be in [0, {B-1}], got {batch_idx}")
        map_2d = tensor[batch_idx].detach().cpu().numpy()
        title_suffix = f" (batch_idx={batch_idx})"
    else:
        map_2d = tensor.detach().cpu().numpy()
        title_suffix = ""

    # 2. 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(
        map_2d,
        aspect="auto",
        origin=origin,  # pyright: ignore[reportArgumentType]
        cmap=cmap,
    )
    plt.colorbar()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}{title_suffix}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved at: {save_path}")
    plt.close()
