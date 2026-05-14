import time

import torch
import torch.nn.functional as F


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


def compute_ahlsc_loss(
    evidence: torch.Tensor,
    soft_attn: torch.Tensor,
    hard_attn: torch.Tensor | None,
    text_mask: torch.Tensor,
    spec_mask: torch.Tensor,
    eta: float | torch.Tensor = 0.0,
    contrastive_temperature: float = 0.1,
    neighborhood_radius: int = 1,
    margin: float = 0.0,
    eps: float = 1e-6,
    reduction: str = "mean",
    detach_soft_region: bool = True,
    detach_hard_region: bool = True,
    sanitize_large_negative_evidence: bool = True,
) -> torch.Tensor:
    """
    Annealed Hard Local Support Contrastive loss.

    Implements

        r_{t,j}^{(eta)}
        =
        (1 - eta) * sg(gamma_{t,j}) / (d_j^soft + eps)
        +
        eta * hard_gamma_{t,j} / (d_j^hard + eps)

        u_{j -> k}^{(eta)}
        =
        sum_t r_{t,j}^{(eta)} e_{t,k}

        L_AHLSC
        =
        - 1/N sum_j log
            exp(u_{j -> j} / tau_c)
            /
            sum_{k in N(j)} exp(u_{j -> k} / tau_c)

    Args:
        evidence:
            Unary evidence field e_{t,k}, shape (B, T_mel, T_text).
            Ideally this should be raw unary evidence, not log_emission and not log_gamma.

        soft_attn:
            Posterior occupancy gamma, shape (B, T_mel, T_text).

        hard_attn:
            Viterbi hard alignment, shape (B, T_mel, T_text).
            If None, the loss uses only the soft component regardless of eta.

        text_mask:
            Valid text-token mask, shape (B, T_text) or (B, 1, T_text).

        spec_mask:
            Valid spectrogram-frame mask, shape (B, T_mel) or (B, 1, T_mel).

        eta:
            Annealing coefficient in [0, 1].
            eta=0: purely soft region pooling.
            eta=1: purely hard Viterbi region pooling.

        contrastive_temperature:
            Local Gibbs / NCE temperature tau_c.

        neighborhood_radius:
            Radius for local neighborhood N(j).
            neighborhood_radius=1 means {j-1, j, j+1} clipped to valid tokens.

        margin:
            Optional positive margin rho.
            margin=0.0 gives the main AHLSC objective.
            margin>0 gives the margin-shifted local NCE variant.

        reduction:
            "mean", "sum", or "none".
            "none" returns per-batch loss, shape (B,).

        detach_soft_region:
            If True, stop gradient through soft_attn in r_{t,j}^{(eta)}.
            This matches sg(gamma) in the paper.

        detach_hard_region:
            Usually True. Hard path should not carry gradient.

        sanitize_large_negative_evidence:
            If True, values smaller than -1e4 are replaced by 0 before pooling.
            This is a safety guard if the caller accidentally passes masked evidence
            containing -1e9. The mathematically clean input is raw evidence.

    Returns:
        Scalar loss if reduction is "mean" or "sum"; otherwise shape (B,).
    """
    if evidence.dim() != 3:
        raise ValueError(f"evidence must be 3D (B, T, N), got {tuple(evidence.shape)}")
    if soft_attn.dim() != 3:
        raise ValueError(f"soft_attn must be 3D (B, T, N), got {tuple(soft_attn.shape)}")

    B, T_mel, T_text = evidence.shape

    if soft_attn.shape != evidence.shape:
        raise ValueError(
            f"soft_attn shape must match evidence shape. "
            + f"got soft_attn={tuple(soft_attn.shape)}, evidence={tuple(evidence.shape)}"
        )

    if hard_attn is not None and hard_attn.shape != evidence.shape:
        raise ValueError(
            f"hard_attn shape must match evidence shape. "
            + f"got hard_attn={tuple(hard_attn.shape)}, evidence={tuple(evidence.shape)}"
        )

    if text_mask.dim() == 3:
        text_mask = text_mask.squeeze(1)
    if spec_mask.dim() == 3:
        spec_mask = spec_mask.squeeze(1)

    text_mask = text_mask.bool()
    spec_mask = spec_mask.bool()

    if text_mask.shape != (B, T_text):
        raise ValueError(
            f"text_mask must have shape (B, T_text). "
            + f"got {tuple(text_mask.shape)}, expected {(B, T_text)}"
        )
    if spec_mask.shape != (B, T_mel):
        raise ValueError(
            f"spec_mask must have shape (B, T_mel). "
            + f"got {tuple(spec_mask.shape)}, expected {(B, T_mel)}"
        )

    if contrastive_temperature <= 0:
        raise ValueError("contrastive_temperature must be positive.")
    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown reduction: {reduction}")

    dtype = evidence.dtype
    device = evidence.device

    # ---------------------------------------------------------------------
    # 1. Prepare evidence e_{t,k}
    # ---------------------------------------------------------------------
    # AHLSC should ideally use raw unary evidence.
    # If the caller passes masked_evidence with -1e9 entries, pooling those
    # entries can create artificial huge negative supports. This guard avoids
    # catastrophic values, but the recommended fix is to pass raw evidence.
    if sanitize_large_negative_evidence:
        evidence_for_pool = torch.where(
            evidence < -1e4,
            torch.zeros_like(evidence),
            evidence,
        )
    else:
        evidence_for_pool = evidence

    # Remove padded frames and padded text tokens from the evidence surface.
    evidence_valid_mask = spec_mask.unsqueeze(-1) & text_mask.unsqueeze(1)
    evidence_for_pool = evidence_for_pool.masked_fill(~evidence_valid_mask, 0.0)

    # ---------------------------------------------------------------------
    # 2. Build annealed region weights r_{t,j}^{(eta)}
    # ---------------------------------------------------------------------
    soft_region = soft_attn
    if detach_soft_region:
        soft_region = soft_region.detach()

    soft_region = soft_region.masked_fill(~evidence_valid_mask, 0.0)
    soft_dur = soft_region.sum(dim=1)  # (B, N)
    soft_region = soft_region / soft_dur.clamp_min(eps).unsqueeze(1)

    if hard_attn is None:
        hard_region = torch.zeros_like(soft_region)
        eta_value = torch.as_tensor(0.0, device=device, dtype=dtype)
    else:
        hard_region = hard_attn
        if detach_hard_region:
            hard_region = hard_region.detach()

        hard_region = hard_region.masked_fill(~evidence_valid_mask, 0.0)
        hard_dur = hard_region.sum(dim=1)  # (B, N)
        hard_region = hard_region / hard_dur.clamp_min(eps).unsqueeze(1)

        eta_value = torch.as_tensor(eta, device=device, dtype=dtype).clamp(0.0, 1.0)

    region_weight = (1.0 - eta_value) * soft_region + eta_value * hard_region
    region_weight = region_weight.masked_fill(~evidence_valid_mask, 0.0)

    # ---------------------------------------------------------------------
    # 3. Pooled support scores u_{j -> k}
    # ---------------------------------------------------------------------
    # region_weight:      (B, T, J)
    # evidence_for_pool:  (B, T, K)
    # support:            (B, J, K)
    support = torch.bmm(region_weight.transpose(1, 2), evidence_for_pool)

    # ---------------------------------------------------------------------
    # 4. Local neighborhood mask N(j)
    # ---------------------------------------------------------------------
    j_idx = torch.arange(T_text, device=device).view(1, T_text, 1)
    k_idx = torch.arange(T_text, device=device).view(1, 1, T_text)

    local_mask = (k_idx - j_idx).abs() <= neighborhood_radius  # (1, J, K)

    # Positive item j must be valid, candidate k must also be valid.
    query_valid = text_mask.unsqueeze(2)  # (B, J, 1)
    candidate_valid = text_mask.unsqueeze(1)  # (B, 1, K)

    local_mask = local_mask & query_valid & candidate_valid  # (B, J, K)

    # ---------------------------------------------------------------------
    # 5. Local NCE / margin-shifted local NCE
    # ---------------------------------------------------------------------
    logits = support / contrastive_temperature

    # Optional margin-shifted variant:
    # positive logit gets -margin / tau.
    if margin != 0.0:
        eye = torch.eye(T_text, device=device, dtype=torch.bool).unsqueeze(0)
        positive_mask = eye & query_valid & candidate_valid
        logits = logits.masked_fill(
            positive_mask,
            logits.masked_select(positive_mask) - (margin / contrastive_temperature),
        )

    neg_large = -1e9 if dtype in (torch.float32, torch.float64) else -1e4
    logits = logits.masked_fill(~local_mask, neg_large)

    # Target class for query j is k=j.
    target = torch.arange(T_text, device=device).view(1, T_text).expand(B, T_text)

    # Cross entropy over K for every query j.
    token_loss = F.cross_entropy(
        logits.reshape(B * T_text, T_text),
        target.reshape(B * T_text),
        reduction="none",
    ).view(B, T_text)

    token_loss = token_loss.masked_fill(~text_mask, 0.0)

    # Average over valid tokens per batch.
    denom = text_mask.sum(dim=1).clamp_min(1).to(dtype)
    batch_loss = token_loss.sum(dim=1) / denom

    if reduction == "none":
        return batch_loss
    if reduction == "sum":
        return batch_loss.sum()
    return batch_loss.mean()


# def compute_alignment_kl_loss(
#     mu_spec: torch.Tensor,  # (B, C, T_s)
#     logvar_spec: torch.Tensor,  # (B, C, T_s)
#     mu_text: torch.Tensor,  # (B, C, T_t)
#     logvar_text: torch.Tensor,  # (B, C, T_t)
#     attn: torch.Tensor,  # gamma, (B, T_s, T_t)
#     spec_mask: torch.Tensor,  # (B, T_s) or (B, 1, T_s)
#     text_mask: torch.Tensor,  # (B, T_t) or (B, 1, T_t)
#     *,
#     detach_attn: bool = False,
# ) -> torch.Tensor:
#     """
#     Alignment-weighted diagonal Gaussian KL loss.

#         L = (1 / T_s) sum_{t,j} gamma_{t,j}
#             KL(q_spec_t || p_text_j)

#     where:
#         q_spec_t = N(mu_spec_t, exp(logvar_spec_t))
#         p_text_j = N(mu_text_j, exp(logvar_text_j))

#     Args:
#         mu_spec:      (B, C, T_s)
#         logvar_spec:  (B, C, T_s)
#         mu_text:      (B, C, T_t)
#         logvar_text:  (B, C, T_t)
#         attn:         posterior alignment gamma, (B, T_s, T_t)
#         spec_mask:    (B, T_s) or (B, 1, T_s)
#         text_mask:    (B, T_t) or (B, 1, T_t)
#         detach_attn:  if True, treats gamma as fixed responsibility.

#     Returns:
#         scalar loss
#     """
#     if spec_mask.dim() == 3:
#         spec_mask = spec_mask.squeeze(1)
#     if text_mask.dim() == 3:
#         text_mask = text_mask.squeeze(1)

#     spec_mask = spec_mask.bool()
#     text_mask = text_mask.bool()

#     weight = attn.detach() if detach_attn else attn
#     pair_mask = spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)
#     weight = weight.masked_fill(~pair_mask, 0.0)

#     mu_s = mu_spec.unsqueeze(-1)  # (B, C, T_s, 1)
#     logvar_s = logvar_spec.unsqueeze(-1)  # (B, C, T_s, 1)
#     var_s = torch.exp(logvar_s)

#     mu_t = mu_text.unsqueeze(2)  # (B, C, 1, T_t)
#     logvar_t = logvar_text.unsqueeze(2)  # (B, C, 1, T_t)
#     inv_var_t = torch.exp(-logvar_t)

#     kl = 0.5 * (logvar_t - logvar_s + (var_s + (mu_s - mu_t).pow(2)) * inv_var_t - 1.0)

#     kl = kl.sum(dim=1)  # (B, T_s, T_t)

#     loss_per_batch = (kl * weight).sum(dim=(1, 2))
#     denom = spec_mask.sum(dim=-1).float().clamp_min(1.0)

#     return (loss_per_batch / denom).mean()


def compute_alignment_kl_loss(
    mu_spec: torch.Tensor,  # (B, C, T_s)
    logvar_spec: torch.Tensor,  # (B, C, T_s)
    mu_text: torch.Tensor,  # (B, C, T_t)
    logvar_text: torch.Tensor,  # (B, C, T_t)
    attn: torch.Tensor,  # (B, T_s, T_t)
    spec_mask: torch.Tensor,  # (B, T_s) or (B, 1, T_s)
    text_mask: torch.Tensor,  # (B, T_t) or (B, 1, T_t)
    *,
    detach_attn: bool = False,
) -> torch.Tensor:
    """
    Same loss as compute_alignment_kl_loss, but avoids constructing
    the huge (B, C, T_s, T_t) KL tensor.

    Computes:

        L = (1 / T_s) sum_{t,j} gamma_{t,j}
            KL(q_spec_t || p_text_j)
    """
    if spec_mask.dim() == 3:
        spec_mask = spec_mask.squeeze(1)
    if text_mask.dim() == 3:
        text_mask = text_mask.squeeze(1)

    spec_mask = spec_mask.bool()
    text_mask = text_mask.bool()

    B, C, T_s = mu_spec.shape

    weight = attn.detach() if detach_attn else attn

    # Remove invalid spec rows without constructing pair_mask of shape (B, T_s, T_t)
    weight = weight * spec_mask.to(weight.dtype).unsqueeze(-1)

    text_mask_f = text_mask.to(weight.dtype)
    spec_mask_c = spec_mask.unsqueeze(1)  # (B, 1, T_s)
    text_mask_c = text_mask.unsqueeze(1)  # (B, 1, T_t)

    # Sanitize invalid positions. This also avoids possible 0 * NaN issues.
    mu_s = mu_spec.masked_fill(~spec_mask_c, 0.0)
    logvar_s = logvar_spec.masked_fill(~spec_mask_c, 0.0)

    mu_t = mu_text.masked_fill(~text_mask_c, 0.0)
    logvar_t = logvar_text.masked_fill(~text_mask_c, 0.0)

    tmask = text_mask_f.unsqueeze(1)  # (B, 1, T_t)

    inv_var_t = torch.exp(-logvar_t) * tmask
    mu_inv_var_t = mu_t * inv_var_t
    text_const_t = logvar_t * tmask + mu_t.square() * inv_var_t

    # Pack three text-side terms and do one batched matmul.
    #
    # weight:              (B, T_s, T_t)
    # text_terms.T:        (B, T_t, 3C)
    # weighted_terms:      (B, 3C, T_s)
    text_terms = torch.cat(
        [inv_var_t, mu_inv_var_t, text_const_t],
        dim=1,
    )  # (B, 3C, T_t)

    weighted_terms = torch.bmm(
        weight,
        text_terms.transpose(1, 2),
    ).transpose(
        1, 2
    )  # (B, 3C, T_s)

    w_inv_var_t, w_mu_inv_var_t, w_text_const_t = weighted_terms.split(C, dim=1)

    # row_sum[t] = sum_j gamma[t, j], excluding invalid text positions
    row_sum = torch.bmm(
        weight,
        text_mask_f.unsqueeze(-1),
    ).squeeze(
        -1
    )  # (B, T_s)

    var_s = torch.exp(logvar_s)

    # KL weighted over text positions, still before summing channel/spec time.
    kl_weighted = (
        w_text_const_t
        + (var_s + mu_s.square()) * w_inv_var_t
        - 2.0 * mu_s * w_mu_inv_var_t
        - (logvar_s + 1.0) * row_sum.unsqueeze(1)
    )

    loss_per_batch = 0.5 * kl_weighted.sum(dim=(1, 2))

    denom = spec_mask.sum(dim=-1).float().clamp_min(1.0)

    return (loss_per_batch / denom).mean()
