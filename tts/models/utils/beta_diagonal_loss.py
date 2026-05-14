import torch


def compute_beta_binomial_loss(
    gamma_posterior: torch.Tensor,  # (B, T_s, T_t)
    y_lengths: torch.Tensor,  # (B,)
    x_lengths: torch.Tensor,  # (B,)
    *,
    omega: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Beta-binomial diagonal prior loss for monotonic alignment.

    Computes:

        L = -(1 / sum_b y_lengths[b]) sum_{b,t,j}
                gamma[b,t,j] * log prior_bb[b,t,j]

    where:

        prior_bb[t, j]
            = BetaBinomial(
                j; n=x_lengths[b]-1,
                alpha=omega * (t + 1),
                beta =omega * (T - t)
              )

    Args:
        gamma_posterior:
            Soft alignment posterior, (B, T_s, T_t).

        y_lengths:
            Valid speech/mel lengths, (B,).

        x_lengths:
            Valid text/token lengths, (B,).

        omega:
            Beta-binomial concentration scale.
            Lower omega => wider prior.
            Higher omega => sharper diagonal prior.

        eps:
            Numerical epsilon.

        detach_gamma:
            If True, the loss does not backprop through gamma_posterior.

    Returns:
        Scalar loss.
    """
    if gamma_posterior.dim() != 3:
        raise ValueError(
            f"gamma_posterior must have shape (B, T_s, T_t), "
            + f"got {tuple(gamma_posterior.shape)}."
        )

    B, T_s, T_t = gamma_posterior.shape
    device = gamma_posterior.device
    dtype = gamma_posterior.dtype

    y_lengths = y_lengths.to(device=device).long()
    x_lengths = x_lengths.to(device=device).long()

    if torch.any(y_lengths <= 0):
        raise ValueError("All y_lengths must be positive.")
    if torch.any(x_lengths <= 0):
        raise ValueError("All x_lengths must be positive.")
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}.")

    gamma = gamma_posterior

    t = torch.arange(T_s, device=device, dtype=dtype).view(1, T_s, 1)
    j = torch.arange(T_t, device=device, dtype=dtype).view(1, 1, T_t)

    T = y_lengths.to(dtype=dtype).view(B, 1, 1)
    N = x_lengths.to(dtype=dtype).view(B, 1, 1)

    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)
    j_idx = torch.arange(T_t, device=device).view(1, 1, T_t)

    valid = (t_idx < y_lengths.view(B, 1, 1)) & (j_idx < x_lengths.view(B, 1, 1))

    # zero-indexed t -> one-indexed t1
    t1 = torch.minimum(t + 1.0, T)

    alpha = omega * t1
    beta = omega * (T - t1 + 1.0)

    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    # Beta-binomial over j = 0, ..., N - 1.
    # Number of trials is n = N - 1.
    n = (N - 1.0).clamp_min(0.0)

    # Avoid invalid lgamma for padded j positions.
    j_eff = torch.minimum(j, n)

    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(j_eff + 1.0) - torch.lgamma(n - j_eff + 1.0)

    log_beta_num = (
        torch.lgamma(j_eff + alpha)
        + torch.lgamma(n - j_eff + beta)
        - torch.lgamma(n + alpha + beta)
    )

    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

    log_prior = log_comb + log_beta_num - log_beta_den  # (B, T_s, T_t)

    neg_large = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9

    log_prior = log_prior.masked_fill(~valid, neg_large)

    # Renormalize over valid text states, because padded positions were clamped.
    log_prior = log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)
    log_prior = log_prior.masked_fill(~valid, neg_large)

    gamma = gamma.masked_fill(~valid, 0.0)

    loss_per_frame = -(gamma * log_prior).sum(dim=-1)  # (B, T_s)

    valid_frame = t_idx.squeeze(-1) < y_lengths.view(B, 1)
    loss_per_frame = loss_per_frame.masked_fill(~valid_frame, 0.0)

    denom = y_lengths.to(dtype=dtype).sum().clamp_min(1.0)

    return loss_per_frame.sum() / denom


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Example lengths
    # ------------------------------------------------------------
    T_s = 60  # y length / mel length
    T_t = 180  # x length / text length
    omega = 1.0

    B = 1
    device = torch.device("cpu")
    dtype = torch.float32

    y_lengths = torch.tensor([T_s], device=device).long()
    x_lengths = torch.tensor([T_t], device=device).long()

    # ------------------------------------------------------------
    # Same beta-binomial prior construction as the loss function
    # ------------------------------------------------------------
    t = torch.arange(T_s, device=device, dtype=dtype).view(1, T_s, 1)
    j = torch.arange(T_t, device=device, dtype=dtype).view(1, 1, T_t)

    T = y_lengths.to(dtype=dtype).view(B, 1, 1)
    N = x_lengths.to(dtype=dtype).view(B, 1, 1)

    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)
    j_idx = torch.arange(T_t, device=device).view(1, 1, T_t)

    valid = (t_idx < y_lengths.view(B, 1, 1)) & (j_idx < x_lengths.view(B, 1, 1))

    eps = 1e-8

    # zero-indexed t -> one-indexed t1
    t1 = torch.minimum(t + 1.0, T)

    alpha = omega * t1
    beta = omega * (T - t1 + 1.0)

    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    # Beta-binomial over j = 0, ..., N - 1
    n = (N - 1.0).clamp_min(0.0)
    j_eff = torch.minimum(j, n)

    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(j_eff + 1.0) - torch.lgamma(n - j_eff + 1.0)

    log_beta_num = (
        torch.lgamma(j_eff + alpha)
        + torch.lgamma(n - j_eff + beta)
        - torch.lgamma(n + alpha + beta)
    )

    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

    log_prior = log_comb + log_beta_num - log_beta_den  # (B, T_s, T_t)

    neg_large = -1e9
    log_prior = log_prior.masked_fill(~valid, neg_large)

    # This is the beta-binomial target gamma.
    log_gamma_beta = log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)
    log_gamma_beta = log_gamma_beta.masked_fill(~valid, neg_large)

    gamma_beta = log_gamma_beta.exp()[0]  # (T_s, T_t)

    # ------------------------------------------------------------
    # Plot beta-binomial target gamma
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.imshow(
        gamma_beta.numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(label="gamma_beta[t, j]")
    plt.xlabel("Text token index j")
    plt.ylabel("Mel frame index t")
    plt.title(f"Beta-binomial target gamma | T_s={T_s}, T_t={T_t}, omega={omega}")
    plt.tight_layout()
    plt.savefig("beta_binomial.png")
