import logging

import pytest
import torch
import math
from tts.models.modules.monotonic_aligner import MonotonicAlignmentNet

logger = logging.getLogger(__name__)


def _log_tensor_stats(
    name: str,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    with torch.no_grad():
        if mask is not None:
            x_view = x[mask]
        else:
            x_view = x.reshape(-1)

        if x_view.numel() == 0:
            logger.info("%s: empty", name)
            return

        logger.info(
            "%s | shape=%s | mean=%.6f | std=%.6f | min=%.6f | max=%.6f | nan=%s",
            name,
            tuple(x.shape),
            x_view.float().mean().item(),
            x_view.float().std(unbiased=False).item(),
            x_view.float().min().item(),
            x_view.float().max().item(),
            torch.isnan(x).any().item(),
        )


def _normalize_log_map_over_states(
    log_map: torch.Tensor,
    valid_mask: torch.Tensor,
    neg_large: float = -1e9,
) -> torch.Tensor:
    """
    Normalize a (T, N) or (B, T, N) log-map over the last dimension
    by subtracting logsumexp over valid states.
    """
    log_map_masked = log_map.masked_fill(~valid_mask, neg_large)
    log_norm = torch.logsumexp(log_map_masked, dim=-1, keepdim=True)
    normalized = log_map_masked - log_norm
    normalized = normalized.masked_fill(~valid_mask, neg_large)
    return normalized


def _prepare_visual_log_map(
    log_map_2d: torch.Tensor,
    valid_mask_2d: torch.Tensor,
    mode: str = "relative",
) -> torch.Tensor:
    """
    mode:
      - 'raw'  : raw log values
      - 'prob' : exp(relative), i.e. framewise normalized probabilities
    """
    if mode == "raw":
        vis = log_map_2d.masked_fill(~valid_mask_2d, float("-inf"))
    elif mode == "prob":
        vis = _normalize_log_map_over_states(
            log_map_2d.unsqueeze(0),
            valid_mask_2d.unsqueeze(0),
        )[0].exp()
        vis = vis.masked_fill(~valid_mask_2d, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return vis


def _assert_no_pad_leakage(
    name: str,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    leakage = x.masked_fill(valid_mask, 0.0).abs().sum().item()
    logger.info("%s leakage=%.8f", name, leakage)
    assert leakage < atol, f"{name} leakage detected: {leakage}"


def _assert_masked_log_emission_consistency(
    log_emission: torch.Tensor,
    masked_log_emission: torch.Tensor,
    state_valid: torch.Tensor,
    neg_large_threshold: float = -1e8,
) -> None:
    valid_error = (masked_log_emission[state_valid] - log_emission[state_valid]).abs().max().item()

    logger.info("masked_log_emission valid_error=%.8f", valid_error)
    assert valid_error < 1e-6, "masked_log_emission differs from log_emission on valid states."

    invalid_values = masked_log_emission[~state_valid]
    if invalid_values.numel() > 0:
        invalid_max = invalid_values.max().item()
        logger.info("masked_log_emission invalid_max=%.6f", invalid_max)
        assert invalid_max <= neg_large_threshold, (
            "masked_log_emission invalid states are not sufficiently negative. "
            f"invalid_max={invalid_max:.6f}"
        )


def _assert_masked_advance_delta_consistency(
    advance_delta: torch.Tensor,
    masked_advance_delta: torch.Tensor,
    state_valid: torch.Tensor,
) -> None:
    valid_error = (
        (masked_advance_delta[state_valid] - advance_delta[state_valid]).abs().max().item()
    )

    logger.info("masked_advance_delta valid_error=%.8f", valid_error)
    assert valid_error < 1e-6, "masked_advance_delta differs from advance_delta on valid states."

    invalid_values = masked_advance_delta[~state_valid]
    if invalid_values.numel() > 0:
        invalid_abs_max = invalid_values.abs().max().item()
        logger.info("masked_advance_delta invalid_abs_max=%.8f", invalid_abs_max)
        assert (
            invalid_abs_max < 1e-6
        ), "masked_advance_delta invalid states should be zero for debugging/logging."


def _assert_transition_consistency(
    log_a_stay: torch.Tensor,
    log_a_adv: torch.Tensor,
    advance_delta: torch.Tensor,
    kappa_adv: torch.Tensor,
    spec_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    neg_large_threshold: float = -1e8,
) -> None:
    """
    Checks contextual transition probabilities.

    For non-last text states:
        log_a_adv  = log sigmoid(kappa_adv + advance_delta)
        log_a_stay = log sigmoid(-(kappa_adv + advance_delta))
        logaddexp(log_a_stay, log_a_adv) = 0

    For last text state:
        stay = 0, advance = -inf
    """
    B, T_s, T_t = log_a_stay.shape
    device = log_a_stay.device

    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)
    j_idx = torch.arange(T_t, device=device).view(1, 1, T_t)

    transition_time = t_idx < (spec_lengths - 1).clamp_min(0).view(B, 1, 1)
    real_text = j_idx < text_lengths.view(B, 1, 1)
    non_last_text = j_idx < (text_lengths - 1).clamp_min(0).view(B, 1, 1)
    last_text = j_idx == (text_lengths - 1).view(B, 1, 1)

    normal_transition = transition_time & non_last_text
    absorbing_transition = transition_time & real_text & last_text

    if normal_transition.any():
        log_norm = torch.logaddexp(log_a_stay, log_a_adv)
        norm_error = log_norm[normal_transition].abs().max().item()
        logger.info("transition normal logaddexp error=%.8f", norm_error)
        assert (
            norm_error < 1e-5
        ), "stay/advance probabilities do not sum to 1 on normal transitions."

        expected_adv_logit = (
            kappa_adv.to(
                dtype=advance_delta.dtype,
                device=advance_delta.device,
            )
            + advance_delta
        )

        actual_adv_logit = log_a_adv - log_a_stay
        logit_error = (
            (actual_adv_logit[normal_transition] - expected_adv_logit[normal_transition])
            .abs()
            .max()
            .item()
        )

        logger.info("transition advance_logit error=%.8f", logit_error)
        assert (
            logit_error < 1e-5
        ), "log_a_adv - log_a_stay does not match kappa_adv + advance_delta."

    if absorbing_transition.any():
        stay_error = log_a_stay[absorbing_transition].abs().max().item()
        adv_max = log_a_adv[absorbing_transition].max().item()

        logger.info(
            "transition absorbing stay_error=%.8f | adv_max=%.6f",
            stay_error,
            adv_max,
        )

        assert stay_error < 1e-6, "Last-state stay transition should be log-prob 0."
        assert (
            adv_max <= neg_large_threshold
        ), "Last-state advance transition should be blocked with neg_large."


def _safe_logsumexp_over_states(
    x: torch.Tensor,
    state_valid: torch.Tensor,
    neg_large: float = -1e9,
) -> torch.Tensor:
    x_masked = x.masked_fill(~state_valid, neg_large)
    return torch.logsumexp(x_masked, dim=-1)


def _log_forward_backward_diagnostics(
    prefix: str,
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor,
    log_gamma: torch.Tensor,
    log_z: torch.Tensor,
    state_valid: torch.Tensor,
    spec_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    max_batches: int = 4,
) -> None:
    """
    Meaningful diagnostics for forward-backward variables.

    log_alpha[t, j]:
        log probability mass of all valid prefixes ending at state j at frame t.

    log_beta[t, j]:
        log probability mass of all valid suffixes starting from state j at frame t.

    log_gamma[t, j]:
        posterior occupancy log P(z_t = j | full sequence).
    """
    with torch.no_grad():
        B, _, _ = log_alpha.shape
        device = log_alpha.device

        alpha_lse = _safe_logsumexp_over_states(log_alpha, state_valid)
        beta_lse = _safe_logsumexp_over_states(log_beta, state_valid)
        gamma_lse = _safe_logsumexp_over_states(log_gamma, state_valid)

        frame_valid = state_valid.any(dim=-1)

        gamma_row_logerr = gamma_lse[frame_valid].abs().max().item()
        logger.info(
            "%s | gamma row logsumexp max_abs_error=%.8f",
            prefix,
            gamma_row_logerr,
        )

        batch_idx = torch.arange(B, device=device)
        terminal_t = spec_lengths - 1
        terminal_j = text_lengths - 1

        alpha_terminal = log_alpha[batch_idx, terminal_t, terminal_j]
        beta_terminal = log_beta[batch_idx, terminal_t, terminal_j]

        alpha_terminal_err = (alpha_terminal - log_z).abs().max().item()
        beta_terminal_err = beta_terminal.abs().max().item()

        logger.info(
            "%s | terminal alpha-logZ max_abs_error=%.8f | terminal beta max_abs_error=%.8f",
            prefix,
            alpha_terminal_err,
            beta_terminal_err,
        )

        for b in range(min(B, max_batches)):
            T_b = int(spec_lengths[b].item())
            N_b = int(text_lengths[b].item())

            probe_ts = sorted(
                set(
                    [
                        0,
                        max(0, T_b // 4),
                        max(0, T_b // 2),
                        max(0, (3 * T_b) // 4),
                        T_b - 1,
                    ]
                )
            )

            logger.info(
                "%s | batch=%d | T=%d | N=%d | logZ=%.6f",
                prefix,
                b,
                T_b,
                N_b,
                log_z[b].item(),
            )

            for t in probe_ts:
                valid_j = state_valid[b, t, :N_b]
                valid_indices = torch.nonzero(valid_j, as_tuple=False).flatten()

                if valid_indices.numel() == 0:
                    logger.info(
                        "%s | batch=%d t=%d | no valid states",
                        prefix,
                        b,
                        t,
                    )
                    continue

                j_min = int(valid_indices.min().item())
                j_max = int(valid_indices.max().item())

                alpha_slice = log_alpha[b, t, :N_b].masked_fill(~valid_j, -1e9)
                beta_slice = log_beta[b, t, :N_b].masked_fill(~valid_j, -1e9)
                gamma_slice = log_gamma[b, t, :N_b].masked_fill(~valid_j, -1e9)

                alpha_argmax = int(alpha_slice.argmax().item())
                beta_argmax = int(beta_slice.argmax().item())
                gamma_argmax = int(gamma_slice.argmax().item())

                gamma_prob = gamma_slice.exp().masked_fill(~valid_j, 0.0)
                gamma_sum = gamma_prob.sum().item()
                gamma_entropy = -(gamma_prob[valid_j] * gamma_slice[valid_j]).sum().item()

                logger.info(
                    (
                        "%s | batch=%d t=%03d | valid_j=[%02d,%02d] "
                        "| alpha_lse=%.6f argmax=%02d max=%.6f "
                        "| beta_lse=%.6f argmax=%02d max=%.6f "
                        "| gamma_sum=%.6f gamma_argmax=%02d gamma_max=%.6f entropy=%.6f"
                    ),
                    prefix,
                    b,
                    t,
                    j_min,
                    j_max,
                    alpha_lse[b, t].item(),
                    alpha_argmax,
                    alpha_slice.max().item(),
                    beta_lse[b, t].item(),
                    beta_argmax,
                    beta_slice.max().item(),
                    gamma_sum,
                    gamma_argmax,
                    gamma_prob.max().item(),
                    gamma_entropy,
                )


def _log_viterbi_vs_posterior_diagnostics(
    prefix: str,
    soft_attn: torch.Tensor,
    hard_attn: torch.Tensor,
    viterbi_path: torch.Tensor,
    spec_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    max_batches: int = 4,
) -> None:
    """
    Compare posterior argmax path and Viterbi MAP path.

    They are not required to be identical, but this tells you whether
    soft posterior mass agrees with the hard MAP path.
    """
    with torch.no_grad():
        B = soft_attn.size(0)
        posterior_argmax_path = soft_attn.argmax(dim=-1)

        for b in range(min(B, max_batches)):
            T_b = int(spec_lengths[b].item())
            N_b = int(text_lengths[b].item())

            vit = viterbi_path[b, :T_b]
            post = posterior_argmax_path[b, :T_b]

            agreement = (vit == post).float().mean().item()

            vit_dur = hard_attn[b, :T_b, :N_b].sum(dim=0)
            soft_dur = soft_attn[b, :T_b, :N_b].sum(dim=0)

            dur_l1 = (vit_dur - soft_dur).abs().sum().item()
            dur_max_abs = (vit_dur - soft_dur).abs().max().item()

            logger.info(
                (
                    "%s | batch=%d | posterior_argmax_vs_viterbi agreement=%.4f "
                    "| duration_l1=%.6f | duration_max_abs=%.6f"
                ),
                prefix,
                b,
                agreement,
                dur_l1,
                dur_max_abs,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_hmm_unary_evidence_aligner_soft_and_hard(
    visualize: bool,
    caplog: pytest.LogCaptureFixture,
):
    """
    Test the left-to-right monotone aligner with:

      1. dot-product log-emission
      2. U-Net contextual transition residual
      3. forward-backward posterior gamma
      4. Viterbi hard path

    Important:
      - log_emission is now a raw potential, not a token-normalized log-probability.
      - log_b is the actual emission score used by DP after temperature scaling/masking.
      - transition is represented by advance_delta, log_a_stay, log_a_adv.
    """
    caplog.set_level(logging.INFO)

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, D_spec, D_text, D_cond, D_hidden = 4, 128, 128, 256, 64
    T_spec_max, T_text_max = 1500, 300

    spec_lengths = torch.tensor([1500, 130, 100, 80], device=device)
    text_lengths = torch.tensor([300, 25, 20, 17], device=device)

    spec_idx = torch.arange(T_spec_max, device=device).unsqueeze(0)
    spec_mask = spec_idx < spec_lengths.unsqueeze(1)

    text_idx = torch.arange(T_text_max, device=device).unsqueeze(0)
    text_mask = text_idx < text_lengths.unsqueeze(1)

    # Gaussian emission inputs.
    # Use small means and broad text variance so the initial emission map is not overly sharp.
    mu_spec = torch.randn(B, D_spec, T_spec_max, device=device) / math.sqrt(D_spec)
    logvar_spec = torch.full(
        (B, D_spec, T_spec_max),
        -4.0,
        device=device,
    ) + 0.01 * torch.randn(
        B, D_spec, T_spec_max, device=device
    ) / math.sqrt(D_spec)

    mu_text = torch.randn(B, D_text, T_text_max, device=device) / math.sqrt(D_text)
    logvar_text = torch.full(
        (B, D_text, T_text_max),
        3.0,
        device=device,
    ) + 0.01 * torch.randn(
        B, D_text, T_text_max, device=device
    ) / math.sqrt(D_text)

    cond = torch.randn(B, D_cond, device=device)

    mu_spec = mu_spec * spec_mask.unsqueeze(1).to(dtype=mu_spec.dtype)
    logvar_spec = logvar_spec * spec_mask.unsqueeze(1).to(dtype=logvar_spec.dtype)

    mu_text = mu_text * text_mask.unsqueeze(1).to(dtype=mu_text.dtype)
    logvar_text = logvar_text * text_mask.unsqueeze(1).to(dtype=logvar_text.dtype)

    aligner = MonotonicAlignmentNet(
        dim_speech=D_spec,
        dim_text=D_text,
        dim_hidden=D_hidden,
        dim_cond=D_cond,
        dim_unet_latent=64,
        transition_adv_init=0.05,
        transition_eta=0.0,
    ).to(device)

    temperature_value = 10.0
    aligner.eval()

    state_valid = MonotonicAlignmentNet.build_state_valid_mask(
        speech_lengths=spec_lengths,
        text_lengths=text_lengths,
        max_speech_len=T_spec_max,
        max_text_len=T_text_max,
        device=device,
    )

    pad_mask_2d = spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)
    state_valid = state_valid & pad_mask_2d

    # -------------------------------------------------------------------------
    # 1. Soft-only forward path
    # -------------------------------------------------------------------------
    with torch.no_grad():
        (
            soft_attn_only,
            log_alpha_only,
            log_beta_only,
            log_gamma_only,
            log_z_only,
            norm_log_z_only,
            log_emission_only,
            masked_log_emission_only,
            advance_delta_only,
            masked_advance_delta_only,
            log_b_only,
            log_a_stay_only,
            log_a_adv_only,
            soft_durations_only,
        ) = aligner(
            z_spec=mu_spec,
            spec_mask=spec_mask,
            mu_text=mu_text,
            logvar_text=logvar_text,
            text_mask=text_mask,
            cond=cond,
            temperature=temperature_value,
            return_hard=False,
        )

    logger.info("========== Soft-only forward ==========")
    logger.info("pi_adv_baseline=%.6f", torch.sigmoid(aligner.kappa_adv).item())

    _log_tensor_stats("soft_attn_only", soft_attn_only, state_valid)
    _log_tensor_stats("log_alpha_only", log_alpha_only, state_valid)
    _log_tensor_stats("log_beta_only", log_beta_only, state_valid)
    _log_tensor_stats("log_gamma_only", log_gamma_only, state_valid)
    _log_tensor_stats("log_z_only", log_z_only)
    _log_tensor_stats("norm_log_z_only", norm_log_z_only)
    _log_tensor_stats("log_emission_only", log_emission_only, state_valid)
    _log_tensor_stats("masked_log_emission_only", masked_log_emission_only, state_valid)
    _log_tensor_stats("advance_delta_only", advance_delta_only, state_valid)
    _log_tensor_stats("masked_advance_delta_only", masked_advance_delta_only, state_valid)
    _log_tensor_stats("log_b_only", log_b_only, state_valid)
    _log_tensor_stats("log_a_stay_only", log_a_stay_only)
    _log_tensor_stats("log_a_adv_only", log_a_adv_only)
    _log_tensor_stats("soft_durations_only", soft_durations_only, text_mask)

    _log_forward_backward_diagnostics(
        prefix="soft_only_fb",
        log_alpha=log_alpha_only,
        log_beta=log_beta_only,
        log_gamma=log_gamma_only,
        log_z=log_z_only,
        state_valid=state_valid,
        spec_lengths=spec_lengths,
        text_lengths=text_lengths,
    )

    assert soft_attn_only.shape == (B, T_spec_max, T_text_max)
    assert log_alpha_only.shape == (B, T_spec_max, T_text_max)
    assert log_beta_only.shape == (B, T_spec_max, T_text_max)
    assert log_gamma_only.shape == (B, T_spec_max, T_text_max)
    assert log_z_only.shape == (B,)
    assert norm_log_z_only.shape == (B,)

    assert log_emission_only.shape == (B, T_spec_max, T_text_max)
    assert masked_log_emission_only.shape == (B, T_spec_max, T_text_max)
    assert advance_delta_only.shape == (B, T_spec_max, T_text_max)
    assert masked_advance_delta_only.shape == (B, T_spec_max, T_text_max)
    assert log_b_only.shape == (B, T_spec_max, T_text_max)
    assert log_a_stay_only.shape == (B, T_spec_max, T_text_max)
    assert log_a_adv_only.shape == (B, T_spec_max, T_text_max)
    assert soft_durations_only.shape == (B, T_text_max)

    tensors_no_nan = {
        "soft_attn_only": soft_attn_only,
        "log_alpha_only": log_alpha_only,
        "log_beta_only": log_beta_only,
        "log_gamma_only": log_gamma_only,
        "log_z_only": log_z_only,
        "norm_log_z_only": norm_log_z_only,
        "log_emission_only": log_emission_only,
        "masked_log_emission_only": masked_log_emission_only,
        "advance_delta_only": advance_delta_only,
        "masked_advance_delta_only": masked_advance_delta_only,
        "log_b_only": log_b_only,
        "log_a_stay_only": log_a_stay_only,
        "log_a_adv_only": log_a_adv_only,
    }

    for name, tensor in tensors_no_nan.items():
        assert not torch.isnan(tensor).any(), f"NaN detected in {name}."

    _assert_no_pad_leakage("soft_attn_only", soft_attn_only, state_valid)

    _assert_masked_log_emission_consistency(
        log_emission=log_emission_only,
        masked_log_emission=masked_log_emission_only,
        state_valid=state_valid,
    )
    _assert_masked_advance_delta_consistency(
        advance_delta=advance_delta_only,
        masked_advance_delta=masked_advance_delta_only,
        state_valid=state_valid,
    )
    _assert_transition_consistency(
        log_a_stay=log_a_stay_only,
        log_a_adv=log_a_adv_only,
        advance_delta=advance_delta_only,
        kappa_adv=aligner.kappa_adv,
        spec_lengths=spec_lengths,
        text_lengths=text_lengths,
    )

    # Posterior occupancy should sum to 1 over valid token states for each valid speech frame.
    soft_row_sum_only = soft_attn_only.sum(dim=-1)
    row_sum_error = (soft_row_sum_only[spec_mask] - 1.0).abs().max().item()
    logger.info("soft_attn_only row_sum_error=%.8f", row_sum_error)

    assert row_sum_error < 1e-4, (
        "Soft posterior gamma does not sum to 1 over states. " f"max_error={row_sum_error:.8f}"
    )

    # Soft durations should sum to the real speech length.
    soft_duration_sum_only = soft_durations_only.sum(dim=-1)
    assert torch.allclose(
        soft_duration_sum_only,
        spec_lengths.float(),
        atol=1e-3,
        rtol=1e-4,
    ), "Soft durations do not sum to speech lengths."

    # Invalid text positions should receive zero duration.
    assert torch.allclose(
        soft_durations_only.masked_fill(text_mask, 0.0),
        torch.zeros_like(soft_durations_only),
        atol=1e-5,
        rtol=0.0,
    ), "Soft duration leaked into padded text positions."

    # -------------------------------------------------------------------------
    # 2. Soft + hard forward path
    # -------------------------------------------------------------------------
    with torch.no_grad():
        (
            soft_attn,
            log_alpha,
            log_beta,
            log_gamma,
            log_z,
            norm_log_z,
            log_emission,
            masked_log_emission,
            advance_delta,
            masked_advance_delta,
            log_b,
            log_a_stay,
            log_a_adv,
            soft_durations,
            hard_attn,
            hard_durations,
            viterbi_path,
            viterbi_logp,
        ) = aligner(
            z_spec=mu_spec,
            spec_mask=spec_mask,
            mu_text=mu_text,
            logvar_text=logvar_text,
            text_mask=text_mask,
            cond=cond,
            temperature=temperature_value,
            return_hard=True,
        )

    logger.info("========== Soft + hard forward ==========")
    _log_tensor_stats("soft_attn", soft_attn, state_valid)
    _log_tensor_stats("hard_attn", hard_attn, state_valid)
    _log_tensor_stats("log_emission", log_emission, state_valid)
    _log_tensor_stats("masked_log_emission", masked_log_emission, state_valid)
    _log_tensor_stats("advance_delta", advance_delta, state_valid)
    _log_tensor_stats("masked_advance_delta", masked_advance_delta, state_valid)
    _log_tensor_stats("log_b", log_b, state_valid)
    _log_tensor_stats("log_a_stay", log_a_stay)
    _log_tensor_stats("log_a_adv", log_a_adv)
    _log_tensor_stats("soft_durations", soft_durations, text_mask)
    _log_tensor_stats("hard_durations", hard_durations, text_mask)
    _log_tensor_stats("viterbi_path", viterbi_path.float(), spec_mask)
    _log_tensor_stats("viterbi_logp", viterbi_logp)

    assert soft_attn.shape == (B, T_spec_max, T_text_max)
    assert log_alpha.shape == (B, T_spec_max, T_text_max)
    assert log_beta.shape == (B, T_spec_max, T_text_max)
    assert log_gamma.shape == (B, T_spec_max, T_text_max)
    assert log_z.shape == (B,)
    assert norm_log_z.shape == (B,)

    assert log_emission.shape == (B, T_spec_max, T_text_max)
    assert masked_log_emission.shape == (B, T_spec_max, T_text_max)
    assert advance_delta.shape == (B, T_spec_max, T_text_max)
    assert masked_advance_delta.shape == (B, T_spec_max, T_text_max)
    assert log_b.shape == (B, T_spec_max, T_text_max)
    assert log_a_stay.shape == (B, T_spec_max, T_text_max)
    assert log_a_adv.shape == (B, T_spec_max, T_text_max)
    assert soft_durations.shape == (B, T_text_max)

    assert hard_attn.shape == (B, T_spec_max, T_text_max)
    assert hard_durations.shape == (B, T_text_max)
    assert viterbi_path.shape == (B, T_spec_max)
    assert viterbi_logp.shape == (B,)

    tensors_no_nan = {
        "soft_attn": soft_attn,
        "hard_attn": hard_attn,
        "log_alpha": log_alpha,
        "log_beta": log_beta,
        "log_gamma": log_gamma,
        "log_z": log_z,
        "norm_log_z": norm_log_z,
        "log_emission": log_emission,
        "masked_log_emission": masked_log_emission,
        "advance_delta": advance_delta,
        "masked_advance_delta": masked_advance_delta,
        "log_b": log_b,
        "log_a_stay": log_a_stay,
        "log_a_adv": log_a_adv,
        "viterbi_logp": viterbi_logp,
    }

    for name, tensor in tensors_no_nan.items():
        assert not torch.isnan(tensor).any(), f"NaN detected in {name}."

    # Soft-only and return_hard=True should agree on shared outputs.
    assert torch.allclose(soft_attn, soft_attn_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_alpha, log_alpha_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_beta, log_beta_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_gamma, log_gamma_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_z, log_z_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(norm_log_z, norm_log_z_only, atol=1e-5, rtol=1e-5)

    assert torch.allclose(log_emission, log_emission_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        masked_log_emission,
        masked_log_emission_only,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(advance_delta, advance_delta_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        masked_advance_delta,
        masked_advance_delta_only,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(log_b, log_b_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_a_stay, log_a_stay_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_a_adv, log_a_adv_only, atol=1e-5, rtol=1e-5)
    assert torch.allclose(soft_durations, soft_durations_only, atol=1e-5, rtol=1e-5)

    _assert_no_pad_leakage("soft_attn", soft_attn, state_valid)
    _assert_no_pad_leakage("hard_attn", hard_attn, state_valid)

    _assert_masked_log_emission_consistency(
        log_emission=log_emission,
        masked_log_emission=masked_log_emission,
        state_valid=state_valid,
    )
    _assert_masked_advance_delta_consistency(
        advance_delta=advance_delta,
        masked_advance_delta=masked_advance_delta,
        state_valid=state_valid,
    )
    _assert_transition_consistency(
        log_a_stay=log_a_stay,
        log_a_adv=log_a_adv,
        advance_delta=advance_delta,
        kappa_adv=aligner.kappa_adv,
        spec_lengths=spec_lengths,
        text_lengths=text_lengths,
    )

    _log_forward_backward_diagnostics(
        prefix="soft_hard_fb",
        log_alpha=log_alpha,
        log_beta=log_beta,
        log_gamma=log_gamma,
        log_z=log_z,
        state_valid=state_valid,
        spec_lengths=spec_lengths,
        text_lengths=text_lengths,
    )

    _log_viterbi_vs_posterior_diagnostics(
        prefix="soft_hard_compare",
        soft_attn=soft_attn,
        hard_attn=hard_attn,
        viterbi_path=viterbi_path,
        spec_lengths=spec_lengths,
        text_lengths=text_lengths,
    )

    # log_emission is raw potential now.
    # Do NOT assert exp(log_emission).sum(-1) == 1.
    # Instead, only gamma should normalize over states.
    soft_row_sum = soft_attn.sum(dim=-1)

    assert torch.allclose(
        soft_row_sum[spec_mask],
        torch.ones_like(soft_row_sum[spec_mask]),
        atol=1e-4,
        rtol=1e-4,
    ), "Soft posterior gamma does not sum to 1 over states."

    # Hard alignment should be one-hot over token states at each valid frame.
    hard_row_sum = hard_attn.sum(dim=-1)

    assert torch.allclose(
        hard_row_sum[spec_mask],
        torch.ones_like(hard_row_sum[spec_mask]),
        atol=1e-5,
        rtol=0.0,
    ), "Hard Viterbi alignment is not one-hot over valid speech frames."

    assert torch.allclose(
        hard_row_sum[~spec_mask],
        torch.zeros_like(hard_row_sum[~spec_mask]),
        atol=1e-5,
        rtol=0.0,
    ), "Hard Viterbi alignment leaked into padded speech frames."

    # Duration consistency.
    assert torch.allclose(
        soft_durations.sum(dim=-1),
        spec_lengths.float(),
        atol=1e-3,
        rtol=1e-4,
    ), "Soft durations do not sum to speech lengths."

    assert torch.allclose(
        hard_durations.sum(dim=-1),
        spec_lengths.float(),
        atol=1e-5,
        rtol=0.0,
    ), "Hard durations do not sum to speech lengths."

    assert torch.allclose(
        hard_durations,
        hard_attn.sum(dim=1),
        atol=1e-5,
        rtol=0.0,
    ), "hard_durations is inconsistent with hard_attn.sum(dim=1)."

    # Invalid text positions should have zero hard duration.
    assert torch.allclose(
        hard_durations.masked_fill(text_mask, 0.0),
        torch.zeros_like(hard_durations),
        atol=1e-5,
        rtol=0.0,
    ), "Hard duration leaked into padded text positions."

    # Strict positive-duration property.
    assert torch.all(hard_durations[text_mask] >= 1.0), (
        "Strict-monotone Viterbi path should assign positive duration " "to every valid text token."
    )

    # -------------------------------------------------------------------------
    # 3. Viterbi path validity
    # -------------------------------------------------------------------------
    for b in range(B):
        T_b = int(spec_lengths[b].item())
        N_b = int(text_lengths[b].item())

        path_b = viterbi_path[b, :T_b]

        logger.info(
            "batch=%d | T=%d | N=%d | hard_duration_sum=%.1f | viterbi_logp=%.6f",
            b,
            T_b,
            N_b,
            hard_durations[b].sum().item(),
            viterbi_logp[b].item(),
        )
        logger.info("batch=%d | hard_durations=%s", b, hard_durations[b, :N_b].tolist())

        assert path_b[0].item() == 0, f"Batch {b}: Viterbi path does not start at token 0."
        assert path_b[-1].item() == N_b - 1, f"Batch {b}: Viterbi path does not end at token N-1."

        diff = path_b[1:] - path_b[:-1]

        assert torch.all((diff == 0) | (diff == 1)), (
            f"Batch {b}: Viterbi path has invalid jumps. "
            "Allowed transitions are stay or advance only."
        )

        assert path_b.min().item() >= 0
        assert path_b.max().item() < N_b

        # Padded path positions should remain -1.
        if T_b < T_spec_max:
            assert torch.all(
                viterbi_path[b, T_b:] == -1
            ), f"Batch {b}: padded Viterbi path positions should be -1."

    # -------------------------------------------------------------------------
    # 4. Optional visualization
    # -------------------------------------------------------------------------
    if visualize:
        from ..utils.visualize_2d_map import visualize_2d_map

        batch_idx = 0
        T_b = int(spec_lengths[batch_idx].item())
        N_b = int(text_lengths[batch_idx].item())

        valid_2d = state_valid[batch_idx, :T_b, :N_b]

        soft_attn_2d = soft_attn[batch_idx, :T_b, :N_b].detach().cpu()
        hard_attn_2d = hard_attn[batch_idx, :T_b, :N_b].detach().cpu()

        log_emission_2d = log_emission[batch_idx, :T_b, :N_b].detach().cpu()
        masked_log_emission_2d = masked_log_emission[batch_idx, :T_b, :N_b].detach().cpu()
        advance_delta_2d = advance_delta[batch_idx, :T_b, :N_b].detach().cpu()
        masked_advance_delta_2d = masked_advance_delta[batch_idx, :T_b, :N_b].detach().cpu()
        log_b_2d = log_b[batch_idx, :T_b, :N_b].detach().cpu()

        log_a_stay_2d = log_a_stay[batch_idx, :T_b, :N_b].detach()
        log_a_adv_2d = log_a_adv[batch_idx, :T_b, :N_b].detach()
        advance_logit_2d = (log_a_adv_2d - log_a_stay_2d).cpu()
        advance_prob_2d = log_a_adv_2d.exp().cpu()
        stay_prob_2d = log_a_stay_2d.exp().cpu()

        log_alpha_2d = log_alpha[batch_idx, :T_b, :N_b].detach()
        log_beta_2d = log_beta[batch_idx, :T_b, :N_b].detach()
        log_gamma_2d = log_gamma[batch_idx, :T_b, :N_b].detach()
        valid_2d_cuda = valid_2d.detach()

        alpha_raw = _prepare_visual_log_map(log_alpha_2d, valid_2d_cuda, mode="raw").cpu()
        beta_raw = _prepare_visual_log_map(log_beta_2d, valid_2d_cuda, mode="raw").cpu()

        alpha_prob = _prepare_visual_log_map(log_alpha_2d, valid_2d_cuda, mode="prob").cpu()
        beta_prob = _prepare_visual_log_map(log_beta_2d, valid_2d_cuda, mode="prob").cpu()
        gamma_prob = _prepare_visual_log_map(log_gamma_2d, valid_2d_cuda, mode="prob").cpu()

        visualize_2d_map(
            soft_attn_2d,
            title="Soft Posterior Gamma",
            save_path="test_hmm_aligner_soft_gamma.png",
        )

        visualize_2d_map(
            hard_attn_2d,
            title="Hard Viterbi Alignment",
            save_path="test_hmm_aligner_hard_viterbi.png",
        )

        visualize_2d_map(
            log_emission_2d,
            title="Raw Dot-Product Log-Emission",
            save_path="test_hmm_aligner_log_emission.png",
        )

        visualize_2d_map(
            masked_log_emission_2d,
            title="Masked Log-Emission",
            save_path="test_hmm_aligner_masked_log_emission.png",
        )

        visualize_2d_map(
            log_b_2d,
            title="Log-B Used By DP",
            save_path="test_hmm_aligner_log_b.png",
        )

        visualize_2d_map(
            advance_delta_2d,
            title="UNet Advance Delta",
            save_path="test_hmm_aligner_advance_delta.png",
        )

        visualize_2d_map(
            masked_advance_delta_2d,
            title="Masked UNet Advance Delta",
            save_path="test_hmm_aligner_masked_advance_delta.png",
        )

        visualize_2d_map(
            advance_logit_2d,
            title="Final Advance Logit",
            save_path="test_hmm_aligner_advance_logit.png",
        )

        visualize_2d_map(
            advance_prob_2d,
            title="Advance Probability",
            save_path="test_hmm_aligner_advance_prob.png",
        )

        visualize_2d_map(
            stay_prob_2d,
            title="Stay Probability",
            save_path="test_hmm_aligner_stay_prob.png",
        )

        visualize_2d_map(
            alpha_raw,
            title="Raw Log-Alpha",
            save_path="test_hmm_aligner_log_alpha_raw.png",
        )

        visualize_2d_map(
            beta_raw,
            title="Raw Log-Beta",
            save_path="test_hmm_aligner_log_beta_raw.png",
        )

        visualize_2d_map(
            alpha_prob,
            title="Alpha Prefix-State Distribution",
            save_path="test_hmm_aligner_alpha_prob.png",
        )

        visualize_2d_map(
            beta_prob,
            title="Beta Suffix-State Distribution",
            save_path="test_hmm_aligner_beta_prob.png",
        )

        visualize_2d_map(
            gamma_prob,
            title="Gamma Posterior Distribution",
            save_path="test_hmm_aligner_gamma_prob.png",
        )
