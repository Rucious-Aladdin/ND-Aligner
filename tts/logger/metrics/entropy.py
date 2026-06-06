import torch


def compute_framewise_entropy(
    attn: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-sample frame-wise posterior entropy.

    Args:
        attn: (B, T_mel, T_text)
            Posterior alignment probability over text tokens.
            Usually sums to 1 over T_text for each valid mel frame.

        mask: (B, T_mel)
            1 for valid mel frames, 0 for padding.

        eps:
            Numerical stability constant for log.

    Returns:
        avg_entropy: (B,)
            Per-sample average entropy over valid mel frames.
            Lower means sharper posterior.
    """
    if attn.dim() != 3:
        raise ValueError(f"attn must have shape (B, T_mel, T_text), got {tuple(attn.shape)}.")

    B, T_mel, _T_text = attn.shape

    if mask.shape != (B, T_mel):
        raise ValueError(f"mask must have shape (B, T_mel), got {tuple(mask.shape)}.")

    mask = mask.to(device=attn.device, dtype=attn.dtype)

    # Frame-wise entropy:
    # H_t = - sum_j gamma[t, j] log gamma[t, j]
    entropy_per_frame = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1)
    # (B, T_mel)

    numerator = (entropy_per_frame * mask).sum(dim=-1)  # (B,)
    denominator = mask.sum(dim=-1).clamp_min(1.0)  # (B,)

    return numerator / denominator
