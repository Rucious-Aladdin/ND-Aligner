import torch


def compute_duration_mae(
    pred_duration: torch.Tensor,
    target_duration: torch.Tensor,
    text_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pred_duration:   (B, T_text)
        target_duration: (B, T_text)
        text_mask:       (B, T_text)

    Returns:
        mae: (B,)
    """
    if text_mask.dim() == 3:
        text_mask = text_mask.squeeze(1)

    text_mask = text_mask.to(dtype=pred_duration.dtype, device=pred_duration.device)

    abs_err = (pred_duration - target_duration).abs() * text_mask
    denom = text_mask.sum(dim=-1).clamp_min(1.0)

    return abs_err.sum(dim=-1) / denom
