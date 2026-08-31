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
    t1 = torch.minimum(t + 1.0, T)

    alpha = omega * t1
    beta = omega * (T - t1 + 1.0)

    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

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
    neg_large = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9

    log_prior = log_prior.masked_fill(~valid, neg_large)
    log_prior = log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)
    log_prior = log_prior.masked_fill(~valid, neg_large)

    gamma = gamma.masked_fill(~valid, 0.0)
    loss_per_frame = -(gamma * log_prior).sum(dim=-1)  # (B, T_s)

    valid_frame = t_idx.squeeze(-1) < y_lengths.view(B, 1)
    loss_per_frame = loss_per_frame.masked_fill(~valid_frame, 0.0)
    denom = y_lengths.to(dtype=dtype).sum().clamp_min(1.0)
    return loss_per_frame.sum() / denom


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    OUT_DIR = Path(__file__).with_suffix("").parent / "beta_binomial_viz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T_s, T_t = 200, 40
    y_lengths = torch.tensor([T_s])
    x_lengths = torch.tensor([T_t])

    def prior_map(omega: float) -> torch.Tensor:
        """Recompute the normalized log prior used inside the loss."""
        t = torch.arange(T_s, dtype=torch.float32).view(1, T_s, 1)
        j = torch.arange(T_t, dtype=torch.float32).view(1, 1, T_t)
        T = torch.tensor(float(T_s)).view(1, 1, 1)
        N = torch.tensor(float(T_t)).view(1, 1, 1)

        t1 = torch.minimum(t + 1.0, T)
        alpha = (omega * t1).clamp_min(1e-8)
        beta = (omega * (T - t1 + 1.0)).clamp_min(1e-8)
        n = (N - 1.0).clamp_min(0.0)
        j_eff = torch.minimum(j, n)

        log_comb = torch.lgamma(n + 1.0) - torch.lgamma(j_eff + 1.0) - torch.lgamma(n - j_eff + 1.0)
        log_num = (
            torch.lgamma(j_eff + alpha)
            + torch.lgamma(n - j_eff + beta)
            - torch.lgamma(n + alpha + beta)
        )
        log_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_prior = log_comb + log_num - log_den
        return log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)

    def make_gamma(offset: float, width: float) -> torch.Tensor:
        """A diagonal band shifted by `offset` tokens, with Gaussian width."""
        t = torch.arange(T_s, dtype=torch.float32).view(T_s, 1)
        j = torch.arange(T_t, dtype=torch.float32).view(1, T_t)
        center = (t / max(T_s - 1, 1)) * (T_t - 1) + offset
        g = torch.exp(-0.5 * ((j - center) / width) ** 2)
        return (g / g.sum(dim=-1, keepdim=True)).unsqueeze(0)

    # --- 1. what the prior looks like ---------------------------------------
    omegas = [0.1, 1.0, 5.0]
    maps = [prior_map(om)[0] for om in omegas]

    prob_vmax = max(m.exp().max().item() for m in maps)
    log_vmin = min(m.min().item() for m in maps)
    log_vmax = max(m.max().item() for m in maps)

    fig, axes = plt.subplots(2, len(omegas), figsize=(12 * len(omegas), 6.0))
    for col, (om, lp) in enumerate(zip(omegas, maps)):
        im_p = axes[0, col].imshow(
            lp.exp().T.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=prob_vmax,
        )
        axes[0, col].set_title(rf"prior, $\omega={om}$")
        axes[0, col].set_ylabel("text token $j$")

        im_l = axes[1, col].imshow(
            lp.T.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=log_vmin,
            vmax=log_vmax,
        )
        axes[1, col].set_title(rf"$\log$ prior, $\omega={om}$")
        axes[1, col].set_xlabel("speech frame $t$")
        axes[1, col].set_ylabel("text token $j$")

    fig.colorbar(im_p, ax=axes[0, :].tolist(), fraction=0.025)
    fig.colorbar(im_l, ax=axes[1, :].tolist(), fraction=0.025)

    fig.suptitle("Beta-binomial diagonal prior")
    fig.savefig(OUT_DIR / "prior_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2. what the loss costs for a given posterior ------------------------
    cases = {
        "on-diagonal, sharp": make_gamma(0.0, 0.8),
        "on-diagonal, blurred": make_gamma(0.0, 4.0),
        "shifted by 5 tokens": make_gamma(5.0, 0.8),
    }

    fig, axes = plt.subplots(2, len(cases), figsize=(4 * len(cases), 5.5))
    lp = prior_map(1.0)
    for col, (name, gamma) in enumerate(cases.items()):
        loss = compute_beta_binomial_loss(gamma, y_lengths, x_lengths, omega=1.0)

        axes[0, col].imshow(
            gamma[0].T.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, col].set_title(f"{name}\nloss = {loss.item():.3f}")
        axes[0, col].set_ylabel("text token $j$")

        per_frame = -(gamma * lp).sum(dim=-1)[0]
        axes[1, col].plot(per_frame.numpy(), linewidth=1.0)
        axes[1, col].set_xlabel("speech frame $t$")
        axes[1, col].set_ylabel("per-frame loss")
        axes[1, col].grid(alpha=0.3)

    fig.suptitle(r"What $\mathcal{L}_{\mathrm{diag}}$ charges, $\omega=1.0$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_by_posterior.png", dpi=150)
    plt.close(fig)

    # --- 3. omega sweep at a fixed posterior ---------------------------------
    gamma = make_gamma(0.0, 2.0)
    sweep = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    losses = [
        compute_beta_binomial_loss(gamma, y_lengths, x_lengths, omega=om).item() for om in sweep
    ]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sweep, losses, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs prior concentration (fixed posterior)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "omega_sweep.png", dpi=150)
    plt.close(fig)

    print(f"saved to {OUT_DIR}")
