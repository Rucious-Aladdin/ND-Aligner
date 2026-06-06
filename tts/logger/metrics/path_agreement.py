import torch


def compute_path_agreement_score(
    attn_posterior: torch.Tensor,
    attn_viterbi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        attn_posterior: (B, T_mel, T_text)
            Posterior alignment probability gamma.
        attn_viterbi: (B, T_mel, T_text)
            Binary hard Viterbi path. Usually one-hot over T_text for each valid mel frame.
        mask: (B, T_mel)
            1 for valid mel frames, 0 for padding.

    Returns:
        path_agreement_score: (B,)
            Per-sample average posterior mass on the Viterbi path over valid mel frames.
            Higher is better.
    """
    if attn_posterior.shape != attn_viterbi.shape:
        raise ValueError(
            f"attn_posterior and attn_viterbi must have the same shape, "
            + f"got {tuple(attn_posterior.shape)} and {tuple(attn_viterbi.shape)}."
        )

    if attn_posterior.dim() != 3:
        raise ValueError(
            f"attn_posterior must have shape (B, T_mel, T_text), "
            + f"got {tuple(attn_posterior.shape)}."
        )

    B, T_mel, _T_text = attn_posterior.shape

    if mask.shape != (B, T_mel):
        raise ValueError(f"mask must have shape (B, T_mel), got {tuple(mask.shape)}.")

    mask = mask.to(device=attn_posterior.device, dtype=attn_posterior.dtype)

    # Posterior probability assigned to the Viterbi-selected text token.
    # (B, T_mel, T_text) -> (B, T_mel)
    path_posterior = (attn_posterior * attn_viterbi).sum(dim=-1)

    numerator = (path_posterior * mask).sum(dim=-1)  # (B,)
    denominator = mask.sum(dim=-1).clamp_min(1.0)  # (B,)

    return numerator / denominator
