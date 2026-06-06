import torch


def compute_viterbi_kl_loss(
    log_gamma: torch.Tensor,
    viterbi_attn: torch.Tensor,
    spec_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the negative log-likelihood (NLL) of the optimal Viterbi path.
    This effectively acts as a KL divergence loss between the hard Viterbi
    distribution (from MAS) and the aligner's likelihood distribution.

    Args:
        log_gamma (torch.Tensor): Log-likelihood matrix of shape (B, T_mel, T_text).
            Typically computed as log-probabilities of mel frames given text features.
        viterbi_attn (torch.Tensor): Binary hard alignment path of shape (B, T_mel, T_text)
            obtained from `find_maximum_path`.
        spec_mask (torch.Tensor): Binary mask for spectrogram sequences of shape (B, T_mel).
            1.0 for valid frames, 0.0 for padding.

    Returns:
        torch.Tensor: Scalar tensor representing the averaged NLL loss.
            The loss is normalized by the total number of valid frames in the batch.
    """
    # 1. Extract log-likelihoods along the Viterbi path
    # Since viterbi_attn is one-hot (B, T_mel, T_text),
    # summing over T_text gives the log-prob at the selected index.
    # Shape: (B, T_mel)
    viterbi_log_probs = (viterbi_attn * log_gamma).sum(dim=-1)

    # 2. Calculate Negative Log-Likelihood (KL Divergence for one-hot target)
    kl_per_frame = -viterbi_log_probs

    # 3. Apply spectrogram mask to ignore padded frames
    masked_kl = kl_per_frame * spec_mask

    # 4. Compute the final average loss (Frame-level average)
    total_kl = masked_kl.sum()
    total_frames = spec_mask.sum() + 1e-8

    loss = total_kl / total_frames

    return loss


def compute_viterbi_ot_loss(
    log_gamma: torch.Tensor,
    viterbi_attn: torch.Tensor,
    text_mask: torch.Tensor,
    spec_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a frame-wise 1D Optimal Transport loss between the predicted
    alignment distribution and the hard Viterbi alignment path.

    This uses the discrete 1D Wasserstein-1 distance on the text-token axis:
        W1(p, q) = sum_j |CDF_p(j) - CDF_q(j)|

    Args:
        log_gamma (torch.Tensor): Log-likelihood / logit matrix of shape (B, T_mel, T_text).
        viterbi_attn (torch.Tensor): Binary hard alignment path (B, T_mel, T_text).
        text_mask (torch.Tensor): Binary mask for text tokens (B, T_text) or (B, 1, T_text).
        spec_mask (torch.Tensor): Binary mask for spectrogram (B, T_mel).

    Returns:
        torch.Tensor: Scalar tensor representing the averaged frame-wise OT loss.
    """
    if text_mask.dim() == 2:
        text_mask = text_mask.unsqueeze(1)  # (B, 1, T_text)

    # 1. Mask log_gamma before softmax to ensure no probability is leaked to padded tokens.
    # Use a safe value for masking (avoiding too large values that might cause fp16 issues).
    masked_log_gamma = log_gamma.masked_fill(text_mask == 0, -1e4)
    pred_attn = torch.softmax(masked_log_gamma, dim=-1)

    # 2. Compute CDFs difference directly.
    # target_attn is viterbi_attn (one-hot, so CDF is step function).
    # cumsum(dim=-1) is efficient.
    # Wasserstein-1 distance: sum |CDF_p - CDF_q|
    # Shape: (B, T_mel, T_text)
    ot_diff = (pred_attn.cumsum(dim=-1) - viterbi_attn.cumsum(dim=-1)).abs()

    # 3. Apply text mask and sum over text dimension
    ot_per_frame = (ot_diff * text_mask).sum(dim=-1)

    # 4. Mask out padded spectrogram frames and average over valid frames.
    masked_ot = ot_per_frame * spec_mask
    loss = masked_ot.sum() / (spec_mask.sum() + 1e-8)

    return loss
