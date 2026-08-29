from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch

from .forward_backward_cuda import MonotoneForwardBackwardCUDA

NEG_LARGE = -1.0e9

# ============================================================
# Generic helpers
# ============================================================


@dataclass(frozen=True)
class Case:
    name: str
    spec_lengths: tuple[int, ...]
    text_lengths: tuple[int, ...]
    speech_size: int
    text_size: int
    optional_positions: tuple[tuple[int, ...], ...] | None
    score_mode: str = "log_softmax"
    seed: int = 0


def synchronize() -> None:
    torch.cuda.synchronize()


def recover_lengths(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        mask:
            Shape: (B, T_speech, T_text)

    Returns:
        spec_lengths:
            Shape: (B,)

        text_lengths:
            Shape: (B,)
    """
    spec_lengths = mask[:, :, 0].sum(dim=1)
    text_lengths = mask[:, 0, :].sum(dim=1)
    return spec_lengths, text_lengths


def make_rectangular_mask(
    spec_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    speech_size: int,
    text_size: int,
) -> torch.Tensor:
    """
    Args:
        spec_lengths:
            Shape: (B,)

        text_lengths:
            Shape: (B,)

    Returns:
        mask:
            Shape: (B, T_speech, T_text)
    """
    spec_axis = torch.arange(
        speech_size,
        device=spec_lengths.device,
    )
    text_axis = torch.arange(
        text_size,
        device=text_lengths.device,
    )

    spec_mask = spec_axis.unsqueeze(0) < spec_lengths.unsqueeze(1)
    text_mask = text_axis.unsqueeze(0) < text_lengths.unsqueeze(1)

    return spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)


def make_case_tensors(
    case: Case,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Returns:
        log_b:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

        opt_sep_mask:
            Shape: (B, T_text), or None
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed)

    spec_lengths = torch.tensor(
        case.spec_lengths,
        device=device,
        dtype=torch.long,
    )
    text_lengths = torch.tensor(
        case.text_lengths,
        device=device,
        dtype=torch.long,
    )

    mask = make_rectangular_mask(
        spec_lengths,
        text_lengths,
        case.speech_size,
        case.text_size,
    ).contiguous()

    raw = torch.randn(
        len(case.spec_lengths),
        case.speech_size,
        case.text_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    if case.score_mode == "log_softmax":
        log_b = torch.log_softmax(
            raw,
            dim=-1,
        )
    elif case.score_mode == "normal":
        log_b = raw
    elif case.score_mode == "wide":
        log_b = raw * 25.0
    elif case.score_mode == "very_negative":
        log_b = -500.0 - raw.abs() * 500.0
    elif case.score_mode == "near_equal":
        log_b = raw * 1.0e-5
    elif case.score_mode == "dominant":
        log_b = raw
        log_b[..., 0] += 50.0
    else:
        raise ValueError(f"Unknown score_mode={case.score_mode!r}.")

    log_b = log_b.contiguous()

    if case.optional_positions is None:
        opt_sep_mask = None
    else:
        opt_sep_mask = torch.zeros(
            len(case.spec_lengths),
            case.text_size,
            device=device,
            dtype=torch.bool,
        )

        for batch_idx, positions in enumerate(case.optional_positions):
            for position in positions:
                if not (0 <= position < case.text_lengths[batch_idx]):
                    raise ValueError(
                        f"Invalid optional position {position} "
                        + f"for sample {batch_idx} with "
                        + f"text_length={case.text_lengths[batch_idx]}."
                    )
                opt_sep_mask[batch_idx, position] = True

        opt_sep_mask = opt_sep_mask.contiguous()

    return log_b, mask, opt_sep_mask


def run_cuda(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float = NEG_LARGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        log_b:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

        opt_sep_mask:
            Shape: (B, T_text), or None

    Returns:
        log_alpha:
            Shape: (B, T_speech, T_text)

        log_beta:
            Shape: (B, T_speech, T_text)
    """
    return MonotoneForwardBackwardCUDA.apply(
        log_b,
        mask,
        opt_sep_mask,
        float(neg_large),
    )


def valid_values(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        tensor:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

    Returns:
        values:
            Shape: (num_valid,)
    """
    return tensor.masked_select(mask)


def error_stats(
    reference: torch.Tensor,
    actual: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """
    Returns:
        max_abs_error, mean_abs_error, rmse
    """
    difference = (reference.double() - actual.double()).abs()

    if mask is not None:
        difference = difference.masked_select(mask)

    if difference.numel() == 0:
        return 0.0, 0.0, 0.0

    max_abs_error = difference.max().item()
    mean_abs_error = difference.mean().item()
    rmse = difference.square().mean().sqrt().item()

    return max_abs_error, mean_abs_error, rmse


def assert_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
) -> None:
    if reference.shape != actual.shape:
        raise AssertionError(
            f"{name}: shape mismatch: "
            + f"reference={tuple(reference.shape)}, "
            + f"actual={tuple(actual.shape)}."
        )

    if mask is None:
        reference_view = reference
        actual_view = actual
    else:
        reference_view = reference.masked_select(mask)
        actual_view = actual.masked_select(mask)

    # Comparison tensors:
    # Shape: (num_compared_elements,)
    # Dtype: torch.float64
    reference_compare = reference_view.double()
    actual_compare = actual_view.double()

    difference = (reference_compare - actual_compare).abs()

    if difference.numel() == 0:
        max_abs_error = 0.0
        mean_abs_error = 0.0
        rmse = 0.0
    else:
        max_abs_error = difference.max().item()
        mean_abs_error = difference.mean().item()
        rmse = difference.square().mean().sqrt().item()

    print(
        f"{name}: max={max_abs_error:.8e}, " + f"mean={mean_abs_error:.8e}, " + f"rmse={rmse:.8e}"
    )

    if torch.allclose(
        reference_compare,
        actual_compare,
        atol=atol,
        rtol=rtol,
    ):
        return

    allowed = atol + rtol * reference_compare.abs()

    normalized = difference / allowed.clamp_min(torch.finfo(torch.float64).tiny)

    flat_index = normalized.argmax().item()

    raise AssertionError(
        f"{name}: mismatch at flattened valid index "
        + f"{flat_index}: "
        + f"reference="
        + f"{reference_compare.flatten()[flat_index].item()}, "  # type: ignore
        + f"actual="
        + f"{actual_compare.flatten()[flat_index].item()}, "  # type: ignore
        + f"abs_error="
        + f"{difference.flatten()[flat_index].item()}, "  # type: ignore
        + f"allowed="
        + f"{allowed.flatten()[flat_index].item()}."  # type: ignore
    )


def assert_exact_zero(
    name: str,
    tensor: torch.Tensor,
) -> None:
    nonzero = torch.count_nonzero(tensor).item()
    if nonzero != 0:
        raise AssertionError(
            f"{name}: expected exact zeros, " + f"but found {nonzero} nonzero elements."
        )


# ============================================================
# Differentiable FP64 reference
# ============================================================


def logaddexp_pair_reference(
    first: torch.Tensor,
    second: torch.Tensor,
    neg_large: float,
) -> torch.Tensor:
    """
    CUDA helper semantics.

    Args:
        first:
            Shape: arbitrary

        second:
            Shape: same as first

    Returns:
        result:
            Shape: same as first
    """
    both_dead = (first <= neg_large) & (second <= neg_large)

    regular = torch.logaddexp(
        first,
        second,
    )
    dead = torch.full_like(
        regular,
        neg_large,
    )

    return torch.where(
        both_dead,
        dead,
        regular,
    )


def initial_alpha_reference(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float,
) -> torch.Tensor:
    """
    Returns:
        alpha_0:
            Shape: (B, T_text)
    """
    batch_size, _, text_size = log_b.shape

    alpha_0 = log_b.new_full(
        (batch_size, text_size),
        neg_large,
    )

    alpha_0[:, 0] = log_b[:, 0, 0]

    if opt_sep_mask is not None and text_size >= 2:
        alpha_0[:, 1] = torch.where(
            opt_sep_mask[:, 0],
            log_b[:, 0, 1],
            alpha_0[:, 1],
        )

    return alpha_0.masked_fill(
        ~mask[:, 0, :],
        neg_large,
    )


def alpha_candidates_reference(
    previous_alpha: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Args:
        previous_alpha:
            Shape: (B, T_text)

    Returns:
        stay:
            Shape: (B, T_text)

        advance:
            Shape: (B, T_text)

        skip:
            Shape: (B, T_text), or None
    """
    batch_size, text_size = previous_alpha.shape

    stay = previous_alpha

    advance = torch.cat(
        (
            previous_alpha.new_full(
                (batch_size, 1),
                neg_large,
            ),
            previous_alpha[:, :-1],
        ),
        dim=1,
    )

    if opt_sep_mask is None:
        return stay, advance, None

    skip = torch.cat(
        (
            previous_alpha.new_full(
                (batch_size, 2),
                neg_large,
            ),
            previous_alpha[:, :-2],
        ),
        dim=1,
    )

    skip_allowed = torch.zeros(
        batch_size,
        text_size,
        device=previous_alpha.device,
        dtype=torch.bool,
    )

    if text_size >= 3:
        skip_allowed[:, 2:] = opt_sep_mask[:, 1:-1]

    skip = skip.masked_fill(
        ~skip_allowed,
        neg_large,
    )

    return stay, advance, skip


def log_alpha_reference(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float = NEG_LARGE,
) -> torch.Tensor:
    """
    Args:
        log_b:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

        opt_sep_mask:
            Shape: (B, T_text), or None

    Returns:
        log_alpha:
            Shape: (B, T_speech, T_text)
    """
    _, speech_size, _ = log_b.shape

    alpha_t = initial_alpha_reference(
        log_b,
        mask,
        opt_sep_mask,
        neg_large,
    )

    alpha_steps = [alpha_t]

    for speech_idx in range(1, speech_size):
        stay, advance, skip = alpha_candidates_reference(
            alpha_t,
            opt_sep_mask,
            neg_large,
        )

        predecessor_sum = logaddexp_pair_reference(
            stay,
            advance,
            neg_large,
        )

        if skip is not None:
            predecessor_sum = logaddexp_pair_reference(
                predecessor_sum,
                skip,
                neg_large,
            )

        alpha_t = log_b[:, speech_idx, :] + predecessor_sum

        alpha_t = alpha_t.masked_fill(
            ~mask[:, speech_idx, :],
            neg_large,
        )

        alpha_steps.append(alpha_t)

    return torch.stack(
        alpha_steps,
        dim=1,
    )


def beta_candidates_reference(
    next_score: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Args:
        next_score:
            Shape: (B, T_text)

    Returns:
        stay:
            Shape: (B, T_text)

        advance:
            Shape: (B, T_text)

        skip:
            Shape: (B, T_text), or None
    """
    batch_size, text_size = next_score.shape

    stay = next_score

    advance = torch.cat(
        (
            next_score[:, 1:],
            next_score.new_full(
                (batch_size, 1),
                neg_large,
            ),
        ),
        dim=1,
    )

    if opt_sep_mask is None:
        return stay, advance, None

    skip = torch.cat(
        (
            next_score[:, 2:],
            next_score.new_full(
                (batch_size, 2),
                neg_large,
            ),
        ),
        dim=1,
    )

    skip_allowed = torch.zeros(
        batch_size,
        text_size,
        device=next_score.device,
        dtype=torch.bool,
    )

    if text_size >= 3:
        skip_allowed[:, :-2] = opt_sep_mask[:, 1:-1]

    skip = skip.masked_fill(
        ~skip_allowed,
        neg_large,
    )

    return stay, advance, skip


def log_beta_reference(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    neg_large: float = NEG_LARGE,
) -> torch.Tensor:
    """
    Args:
        log_b:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

        opt_sep_mask:
            Shape: (B, T_text), or None

    Returns:
        log_beta:
            Shape: (B, T_speech, T_text)
    """
    batch_size, speech_size, text_size = log_b.shape
    device = log_b.device

    spec_lengths, text_lengths = recover_lengths(mask)

    text_indices = torch.arange(
        text_size,
        device=device,
    ).view(1, text_size)

    terminal_state = text_indices == (text_lengths - 1).view(
        batch_size,
        1,
    )

    beta_steps: list[torch.Tensor | None] = [None] * speech_size
    beta_next: torch.Tensor | None = None

    for speech_idx in range(
        speech_size - 1,
        -1,
        -1,
    ):
        base = log_b.new_full(
            (batch_size, text_size),
            neg_large,
        )

        terminal_batches = (spec_lengths - 1) == speech_idx

        base = torch.where(
            terminal_batches.view(
                batch_size,
                1,
            )
            & terminal_state,
            torch.zeros_like(base),
            base,
        )

        if speech_idx == speech_size - 1:
            beta_t = base
        else:
            assert beta_next is not None

            next_score = log_b[:, speech_idx + 1, :] + beta_next

            stay, advance, skip = beta_candidates_reference(
                next_score,
                opt_sep_mask,
                neg_large,
            )

            recursive = logaddexp_pair_reference(
                stay,
                advance,
                neg_large,
            )

            if skip is not None:
                recursive = logaddexp_pair_reference(
                    recursive,
                    skip,
                    neg_large,
                )

            beta_t = torch.where(
                terminal_batches.view(
                    batch_size,
                    1,
                ),
                base,
                recursive,
            )

        beta_t = beta_t.masked_fill(
            ~mask[:, speech_idx, :],
            neg_large,
        )

        beta_steps[speech_idx] = beta_t
        beta_next = beta_t

    return torch.stack(
        beta_steps,  # type: ignore[arg-type]
        dim=1,
    )


# ============================================================
# Partition, gamma, and transition posterior
# ============================================================


def terminal_log_z(
    log_alpha: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        log_alpha:
            Shape: (B, T_speech, T_text)

        mask:
            Shape: (B, T_speech, T_text)

    Returns:
        log_z:
            Shape: (B,)
    """
    spec_lengths, text_lengths = recover_lengths(mask)
    batch_indices = torch.arange(
        log_alpha.size(0),
        device=log_alpha.device,
    )

    return log_alpha[
        batch_indices,
        spec_lengths - 1,
        text_lengths - 1,
    ]


def start_log_z(
    log_b: torch.Tensor,
    log_beta: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Args:
        log_b:
            Shape: (B, T_speech, T_text)

        log_beta:
            Shape: (B, T_speech, T_text)

        opt_sep_mask:
            Shape: (B, T_text), or None

    Returns:
        log_z:
            Shape: (B,)
    """
    candidate_0 = log_b[:, 0, 0] + log_beta[:, 0, 0]

    if opt_sep_mask is None or log_b.size(2) < 2:
        return candidate_0

    candidate_1 = log_b[:, 0, 1] + log_beta[:, 0, 1]

    return torch.where(
        opt_sep_mask[:, 0],
        torch.logaddexp(
            candidate_0,
            candidate_1,
        ),
        candidate_0,
    )


def posterior_gamma(
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor,
    log_z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        gamma:
            Shape: (B, T_speech, T_text)
    """
    log_gamma = log_alpha + log_beta - log_z[:, None, None]

    gamma = torch.exp(log_gamma)
    return gamma.masked_fill(
        ~mask,
        0.0,
    )


def transition_posterior(
    log_alpha: torch.Tensor,
    log_b: torch.Tensor,
    log_beta: torch.Tensor,
    log_z: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Returns:
        xi_stay:
            Shape: (B, T_speech - 1, T_text)

        xi_advance:
            Shape: (B, T_speech - 1, T_text)

        xi_skip:
            Shape: (B, T_speech - 1, T_text)

    Destination index k is used on the final axis.
    """
    batch_size, speech_size, text_size = log_b.shape

    xi_stay = log_b.new_zeros(
        batch_size,
        speech_size - 1,
        text_size,
    )
    xi_advance = torch.zeros_like(xi_stay)
    xi_skip = torch.zeros_like(xi_stay)

    if speech_size <= 1:
        return xi_stay, xi_advance, xi_skip

    destination_score = log_b[:, 1:, :] + log_beta[:, 1:, :]

    # stay: j -> j, destination k=j
    stay_log = log_alpha[:, :-1, :] + destination_score - log_z[:, None, None]
    xi_stay = torch.exp(stay_log)

    # advance: j=k-1 -> k
    if text_size >= 2:
        advance_log = log_alpha[:, :-1, :-1] + destination_score[:, :, 1:] - log_z[:, None, None]
        xi_advance[:, :, 1:] = torch.exp(advance_log)

    # optional skip: j=k-2 -> k
    if opt_sep_mask is not None and text_size >= 3:
        skip_allowed = opt_sep_mask[:, 1:-1]
        skip_log = log_alpha[:, :-1, :-2] + destination_score[:, :, 2:] - log_z[:, None, None]
        skip_values = torch.exp(skip_log)
        xi_skip[:, :, 2:] = torch.where(
            skip_allowed[:, None, :],
            skip_values,
            torch.zeros_like(skip_values),
        )

    transition_mask = mask[:, :-1, :] & mask[:, 1:, :]
    xi_stay = xi_stay.masked_fill(
        ~transition_mask,
        0.0,
    )

    advance_mask = torch.zeros_like(transition_mask)
    if text_size >= 2:
        advance_mask[:, :, 1:] = mask[:, :-1, :-1] & mask[:, 1:, 1:]
    xi_advance = xi_advance.masked_fill(
        ~advance_mask,
        0.0,
    )

    skip_mask = torch.zeros_like(transition_mask)
    if text_size >= 3:
        skip_mask[:, :, 2:] = mask[:, :-1, :-2] & mask[:, 1:, 2:]
    xi_skip = xi_skip.masked_fill(
        ~skip_mask,
        0.0,
    )

    return xi_stay, xi_advance, xi_skip


# ============================================================
# Exhaustive path enumeration for tiny cases
# ============================================================


def successors(
    state: int,
    text_length: int,
    opt_sep: torch.Tensor | None,
) -> tuple[int, ...]:
    result: list[int] = [state]

    if state + 1 < text_length:
        result.append(state + 1)

    if state + 2 < text_length and opt_sep is not None and bool(opt_sep[state + 1].item()):
        result.append(state + 2)

    return tuple(result)


def enumerate_paths(
    spec_length: int,
    text_length: int,
    opt_sep: torch.Tensor | None,
) -> list[tuple[int, ...]]:
    """
    Returns:
        paths:
            Every valid complete path. Each path has length
            T_speech and ends at T_text - 1.
    """
    starts = [0]

    if text_length >= 2 and opt_sep is not None and bool(opt_sep[0].item()):
        starts.append(1)

    paths: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...]) -> None:
        if len(prefix) == spec_length:
            if prefix[-1] == text_length - 1:
                paths.append(prefix)
            return

        for next_state in successors(
            prefix[-1],
            text_length,
            opt_sep,
        ):
            visit(prefix + (next_state,))

    for start in starts:
        visit((start,))

    return paths


def exhaustive_log_z_gamma(
    log_b: torch.Tensor,
    opt_sep: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """
    Args:
        log_b:
            Shape: (T_speech, T_text)

        opt_sep:
            Shape: (T_text,), or None

    Returns:
        log_z:
            Shape: scalar

        gamma:
            Shape: (T_speech, T_text)

        num_paths:
            Python int
    """
    speech_size, text_size = log_b.shape

    paths = enumerate_paths(
        speech_size,
        text_size,
        opt_sep,
    )

    if not paths:
        return (
            log_b.new_tensor(NEG_LARGE),
            log_b.new_zeros(
                speech_size,
                text_size,
            ),
            0,
        )

    path_scores = torch.stack(
        [torch.stack([log_b[t, state] for t, state in enumerate(path)]).sum() for path in paths]
    )

    log_z = torch.logsumexp(
        path_scores,
        dim=0,
    )
    probabilities = torch.exp(path_scores - log_z)

    gamma = log_b.new_zeros(
        speech_size,
        text_size,
    )

    for path_idx, path in enumerate(paths):
        for speech_idx, state in enumerate(path):
            gamma[speech_idx, state] += probabilities[path_idx]

    return log_z, gamma, len(paths)


# ============================================================
# Mathematical consistency tests
# ============================================================


@torch.no_grad()
def test_forward_reference(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[forward/reference] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    alpha_cuda, beta_cuda = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )

    alpha_reference = log_alpha_reference(
        log_b,
        mask,
        opt_sep_mask,
    )
    beta_reference = log_beta_reference(
        log_b,
        mask,
        opt_sep_mask,
    )

    assert_close(
        "log_alpha reference vs CUDA",
        alpha_reference,
        alpha_cuda,
        mask=mask,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert_close(
        "log_beta reference vs CUDA",
        beta_reference,
        beta_cuda,
        mask=mask,
        atol=3.0e-5,
        rtol=3.0e-5,
    )

    assert torch.all(alpha_cuda.masked_select(~mask) == NEG_LARGE)
    assert torch.all(beta_cuda.masked_select(~mask) == NEG_LARGE)


@torch.no_grad()
def test_partition_and_gamma(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[math/partition-gamma] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )
    alpha, beta = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )

    log_z_terminal = terminal_log_z(
        alpha,
        mask,
    )
    log_z_start = start_log_z(
        log_b,
        beta,
        opt_sep_mask,
    )

    assert_close(
        "terminal logZ vs start logZ",
        log_z_terminal,
        log_z_start,
        atol=5.0e-5,
        rtol=5.0e-5,
    )

    gamma = posterior_gamma(
        alpha,
        beta,
        log_z_terminal,
        mask,
    )

    spec_lengths, _ = recover_lengths(mask)

    for batch_idx in range(log_b.size(0)):
        spec_length = int(spec_lengths[batch_idx].item())

        row_sums = gamma[
            batch_idx,
            :spec_length,
            :,
        ].sum(dim=-1)

        assert_close(
            f"gamma text-axis sum sample={batch_idx}",
            torch.ones_like(row_sums),
            row_sums,
            atol=1.0e-4,
            rtol=1.0e-4,
        )

        log_z_by_row = torch.logsumexp(
            alpha[
                batch_idx,
                :spec_length,
                :,
            ]
            + beta[
                batch_idx,
                :spec_length,
                :,
            ],
            dim=-1,
        )

        assert_close(
            f"logZ constant over time sample={batch_idx}",
            log_z_terminal[batch_idx].expand_as(log_z_by_row),
            log_z_by_row,
            atol=1.0e-4,
            rtol=1.0e-4,
        )


@torch.no_grad()
def test_transition_posterior(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[math/transition-posterior] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )
    alpha, beta = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )
    log_z = terminal_log_z(alpha, mask)
    gamma = posterior_gamma(
        alpha,
        beta,
        log_z,
        mask,
    )

    xi_stay, xi_advance, xi_skip = transition_posterior(
        alpha,
        log_b,
        beta,
        log_z,
        mask,
        opt_sep_mask,
    )
    xi_total = xi_stay + xi_advance + xi_skip

    spec_lengths, text_lengths = recover_lengths(mask)

    for batch_idx in range(log_b.size(0)):
        spec_length = int(spec_lengths[batch_idx].item())
        text_length = int(text_lengths[batch_idx].item())

        if spec_length <= 1:
            continue

        # Total transition posterior at each boundary.
        total_mass = xi_total[
            batch_idx,
            : spec_length - 1,
            :text_length,
        ].sum(dim=-1)

        assert_close(
            f"xi total mass sample={batch_idx}",
            torch.ones_like(total_mass),
            total_mass,
            atol=1.0e-4,
            rtol=1.0e-4,
        )

        # Outgoing transition mass equals gamma[t, j].
        outgoing = torch.zeros(
            spec_length - 1,
            text_length,
            device=device,
            dtype=log_b.dtype,
        )
        outgoing += xi_stay[
            batch_idx,
            : spec_length - 1,
            :text_length,
        ]

        if text_length >= 2:
            outgoing[:, :-1] += xi_advance[
                batch_idx,
                : spec_length - 1,
                1:text_length,
            ]

        if text_length >= 3:
            outgoing[:, :-2] += xi_skip[
                batch_idx,
                : spec_length - 1,
                2:text_length,
            ]

        assert_close(
            f"xi outgoing vs gamma sample={batch_idx}",
            gamma[
                batch_idx,
                : spec_length - 1,
                :text_length,
            ],
            outgoing,
            atol=1.0e-4,
            rtol=1.0e-4,
        )

        # Incoming transition mass equals gamma[t+1, k].
        incoming = xi_total[
            batch_idx,
            : spec_length - 1,
            :text_length,
        ]

        assert_close(
            f"xi incoming vs gamma sample={batch_idx}",
            gamma[
                batch_idx,
                1:spec_length,
                :text_length,
            ],
            incoming,
            atol=1.0e-4,
            rtol=1.0e-4,
        )


@torch.no_grad()
def test_exhaustive_enumeration(
    device: torch.device,
) -> None:
    print("\n[math/exhaustive-path-enumeration]")

    tiny_cases = (
        Case(
            name="tiny_no_optional",
            spec_lengths=(5,),
            text_lengths=(4,),
            speech_size=5,
            text_size=4,
            optional_positions=None,
            seed=101,
        ),
        Case(
            name="tiny_optional_start_middle",
            spec_lengths=(5,),
            text_lengths=(5,),
            speech_size=5,
            text_size=5,
            optional_positions=((0, 2),),
            seed=102,
        ),
        Case(
            name="tiny_skip_required",
            spec_lengths=(3,),
            text_lengths=(4,),
            speech_size=3,
            text_size=4,
            optional_positions=((1,),),
            seed=103,
        ),
    )

    for case in tiny_cases:
        log_b, mask, opt_sep_mask = make_case_tensors(
            case,
            device,
            torch.float64,
        )

        alpha_ref = log_alpha_reference(
            log_b,
            mask,
            opt_sep_mask,
        )
        beta_ref = log_beta_reference(
            log_b,
            mask,
            opt_sep_mask,
        )
        log_z_ref = terminal_log_z(
            alpha_ref,
            mask,
        )[0]
        gamma_ref = posterior_gamma(
            alpha_ref,
            beta_ref,
            log_z_ref.view(1),
            mask,
        )[0]

        opt_sep_single = None if opt_sep_mask is None else opt_sep_mask[0]
        log_z_enum, gamma_enum, num_paths = exhaustive_log_z_gamma(
            log_b[0],
            opt_sep_single,
        )

        print(f"{case.name}: enumerated paths={num_paths}")

        assert_close(
            f"{case.name} exhaustive logZ",
            log_z_enum,
            log_z_ref,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert_close(
            f"{case.name} exhaustive gamma",
            gamma_enum,
            gamma_ref,
            atol=1.0e-10,
            rtol=1.0e-10,
        )


# ============================================================
# Backward tests
# ============================================================


def fp64_reference_gradient(
    branch: str,
    log_b_fp32: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    upstream_fp32: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        log_b_fp32:
            Shape: (B, T_speech, T_text)
        mask:
            Shape: (B, T_speech, T_text)
        opt_sep_mask:
            Shape: (B, T_text), or None
        upstream_fp32:
            Shape: (B, T_speech, T_text)

    Returns:
        grad_log_b_fp64:
            Shape: (B, T_speech, T_text)
    """
    log_b_fp64 = log_b_fp32.detach().double().clone().requires_grad_(True)

    if branch == "alpha":
        # output:
        # Shape: (B, T_speech, T_text)
        output = log_alpha_reference(
            log_b_fp64,
            mask,
            opt_sep_mask,
        )
    elif branch == "beta":
        # output:
        # Shape: (B, T_speech, T_text)
        output = log_beta_reference(
            log_b_fp64,
            mask,
            opt_sep_mask,
        )
    else:
        raise ValueError(f"Unknown branch={branch!r}.")

    # 예: T_speech == 1인 beta는 terminal constant만 포함하므로
    # log_b와 계산 그래프가 전혀 연결되지 않는다.
    if not output.requires_grad:
        # grad_log_b_fp64:
        # Shape: (B, T_speech, T_text)
        return torch.zeros_like(log_b_fp64)

    # grad_log_b_fp64:
    # Shape: (B, T_speech, T_text)
    gradient = torch.autograd.grad(
        outputs=output,
        inputs=log_b_fp64,
        grad_outputs=upstream_fp32.double(),
        retain_graph=False,
        create_graph=False,
    )[0]

    return gradient


def cuda_branch_gradient(
    branch: str,
    log_b_fp32: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
    upstream_fp32: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        grad_log_b:
            Shape: (B, T_speech, T_text)
    """
    log_b = log_b_fp32.detach().clone().requires_grad_(True)

    alpha, beta = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )

    output = alpha if branch == "alpha" else beta

    gradient = torch.autograd.grad(
        outputs=output,
        inputs=log_b,
        grad_outputs=upstream_fp32,
        retain_graph=False,
        create_graph=False,
    )[0]

    return gradient


def test_backward_reference(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[backward/reference] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed + 10000)

    upstream_alpha = (
        torch.randn(
            log_b.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .masked_fill(~mask, 0.0)
        .contiguous()
    )

    upstream_beta = (
        torch.randn(
            log_b.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .masked_fill(~mask, 0.0)
        .contiguous()
    )

    for branch, upstream in (
        ("alpha", upstream_alpha),
        ("beta", upstream_beta),
    ):
        reference = fp64_reference_gradient(
            branch,
            log_b,
            mask,
            opt_sep_mask,
            upstream,
        )
        actual = cuda_branch_gradient(
            branch,
            log_b,
            mask,
            opt_sep_mask,
            upstream,
        )

        assert_close(
            f"{branch} FP64 reference vs CUDA",
            reference,
            actual,
            mask=mask,
            atol=8.0e-4,
            rtol=8.0e-4,
        )

        assert_exact_zero(
            f"{branch} padding gradient",
            actual.masked_select(~mask),
        )

    # Branch additivity.
    log_b_joint = log_b.detach().clone().requires_grad_(True)
    alpha_joint, beta_joint = run_cuda(
        log_b_joint,
        mask,
        opt_sep_mask,
    )
    grad_joint = torch.autograd.grad(
        outputs=(alpha_joint, beta_joint),
        inputs=log_b_joint,
        grad_outputs=(
            upstream_alpha,
            upstream_beta,
        ),
    )[0]

    grad_alpha = cuda_branch_gradient(
        "alpha",
        log_b,
        mask,
        opt_sep_mask,
        upstream_alpha,
    )
    grad_beta = cuda_branch_gradient(
        "beta",
        log_b,
        mask,
        opt_sep_mask,
        upstream_beta,
    )

    assert_close(
        "alpha+beta backward additivity",
        grad_alpha + grad_beta,
        grad_joint,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_backward_structural_zeros(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[backward/structural-zeros] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    zero_upstream = torch.zeros_like(log_b)

    grad_alpha_zero = cuda_branch_gradient(
        "alpha",
        log_b,
        mask,
        opt_sep_mask,
        zero_upstream,
    )
    grad_beta_zero = cuda_branch_gradient(
        "beta",
        log_b,
        mask,
        opt_sep_mask,
        zero_upstream,
    )

    assert_exact_zero(
        "alpha zero-upstream gradient",
        grad_alpha_zero,
    )
    assert_exact_zero(
        "beta zero-upstream gradient",
        grad_beta_zero,
    )

    random_upstream = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    grad_beta = cuda_branch_gradient(
        "beta",
        log_b,
        mask,
        opt_sep_mask,
        random_upstream,
    )

    # beta recurrence never consumes log_b[:, 0, :].
    assert_exact_zero(
        "beta grad_log_b first speech row",
        grad_beta[:, 0, :],
    )

    grad_alpha = cuda_branch_gradient(
        "alpha",
        log_b,
        mask,
        opt_sep_mask,
        random_upstream,
    )

    _spec_lengths, text_lengths = recover_lengths(mask)
    for batch_idx in range(log_b.size(0)):
        text_length = int(text_lengths[batch_idx].item())

        allowed = torch.zeros(
            text_length,
            device=device,
            dtype=torch.bool,
        )
        allowed[0] = True

        if (
            text_length >= 2
            and opt_sep_mask is not None
            and bool(opt_sep_mask[batch_idx, 0].item())
        ):
            allowed[1] = True

        assert_exact_zero(
            f"alpha t=0 unused states sample={batch_idx}",
            grad_alpha[
                batch_idx,
                0,
                :text_length,
            ].masked_select(~allowed),
        )


def test_logz_gradient_equals_gamma(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[backward/logZ-gradient=gamma] {case.name}")

    log_b_base, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    # Alpha terminal construction.
    log_b_alpha = log_b_base.detach().clone().requires_grad_(True)
    alpha, beta = run_cuda(
        log_b_alpha,
        mask,
        opt_sep_mask,
    )
    log_z_alpha = terminal_log_z(
        alpha,
        mask,
    )
    gamma = posterior_gamma(
        alpha.detach(),
        beta.detach(),
        log_z_alpha.detach(),
        mask,
    )

    grad_alpha = torch.autograd.grad(
        log_z_alpha.sum(),
        log_b_alpha,
    )[0]

    assert_close(
        "grad terminal-alpha logZ vs gamma",
        gamma,
        grad_alpha,
        mask=mask,
        atol=2.0e-4,
        rtol=2.0e-4,
    )

    # Beta start construction. This includes the direct first-row
    # log_b term outside the beta kernel.
    log_b_beta = log_b_base.detach().clone().requires_grad_(True)
    alpha_2, beta_2 = run_cuda(
        log_b_beta,
        mask,
        opt_sep_mask,
    )
    log_z_beta = start_log_z(
        log_b_beta,
        beta_2,
        opt_sep_mask,
    )
    gamma_2 = posterior_gamma(
        alpha_2.detach(),
        beta_2.detach(),
        log_z_beta.detach(),
        mask,
    )

    grad_beta = torch.autograd.grad(
        log_z_beta.sum(),
        log_b_beta,
    )[0]

    assert_close(
        "grad start-beta logZ vs gamma",
        gamma_2,
        grad_beta,
        mask=mask,
        atol=2.0e-4,
        rtol=2.0e-4,
    )


# ============================================================
# Batch, padding, reachability, and determinism
# ============================================================


@torch.no_grad()
def test_batched_vs_single(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[batch/batched-vs-single] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )
    alpha_batch, beta_batch = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )

    spec_lengths, text_lengths = recover_lengths(mask)

    for batch_idx in range(log_b.size(0)):
        spec_length = int(spec_lengths[batch_idx].item())
        text_length = int(text_lengths[batch_idx].item())

        single_log_b = log_b[
            batch_idx : batch_idx + 1,
            :spec_length,
            :text_length,
        ].contiguous()
        single_mask = torch.ones_like(
            single_log_b,
            dtype=torch.bool,
        )
        single_opt_sep = (
            None
            if opt_sep_mask is None
            else opt_sep_mask[
                batch_idx : batch_idx + 1,
                :text_length,
            ].contiguous()
        )

        alpha_single, beta_single = run_cuda(
            single_log_b,
            single_mask,
            single_opt_sep,
        )

        assert_close(
            f"alpha batched vs single sample={batch_idx}",
            alpha_batch[
                batch_idx : batch_idx + 1,
                :spec_length,
                :text_length,
            ],
            alpha_single,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        assert_close(
            f"beta batched vs single sample={batch_idx}",
            beta_batch[
                batch_idx : batch_idx + 1,
                :spec_length,
                :text_length,
            ],
            beta_single,
            atol=2.0e-5,
            rtol=2.0e-5,
        )


def test_padding_invariance(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[padding/invariance] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    changed = log_b.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000.0

    alpha_a, beta_a = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )
    alpha_b, beta_b = run_cuda(
        changed,
        mask,
        opt_sep_mask,
    )

    assert_close(
        "alpha padding invariance",
        alpha_a,
        alpha_b,
        mask=mask,
        atol=0.0,
        rtol=0.0,
    )
    assert_close(
        "beta padding invariance",
        beta_a,
        beta_b,
        mask=mask,
        atol=0.0,
        rtol=0.0,
    )

    upstream_alpha = torch.randn_like(log_b).masked_fill(~mask, 0.0)
    upstream_beta = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    def total_gradient(
        values: torch.Tensor,
    ) -> torch.Tensor:
        x = values.detach().clone().requires_grad_(True)
        alpha, beta = run_cuda(
            x,
            mask,
            opt_sep_mask,
        )
        return torch.autograd.grad(
            outputs=(alpha, beta),
            inputs=x,
            grad_outputs=(
                upstream_alpha,
                upstream_beta,
            ),
        )[0]

    grad_a = total_gradient(log_b)
    grad_b = total_gradient(changed)

    assert_close(
        "gradient padding invariance",
        grad_a,
        grad_b,
        mask=mask,
        atol=0.0,
        rtol=0.0,
    )
    assert_exact_zero(
        "padding gradient",
        grad_a.masked_select(~mask),
    )


@torch.no_grad()
def test_reachability(
    device: torch.device,
) -> None:
    print("\n[topology/reachability]")

    case = Case(
        name="reachability_no_optional",
        spec_lengths=(6,),
        text_lengths=(6,),
        speech_size=6,
        text_size=6,
        optional_positions=None,
        seed=400,
    )
    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )
    alpha, beta = run_cuda(
        log_b,
        mask,
        opt_sep_mask,
    )

    # Without optional start/skip, alpha[t, j] is unreachable for j > t.
    for speech_idx in range(case.speech_size):
        if speech_idx + 1 < case.text_size:
            unreachable = alpha[
                0,
                speech_idx,
                speech_idx + 1 :,
            ]
            if not torch.all(unreachable == NEG_LARGE):
                raise AssertionError("Alpha reachability invariant failed.")

    # From state j at time t, reaching N-1 requires at least
    # N-1-j advances. If not enough speech steps remain, beta is dead.
    for speech_idx in range(case.speech_size):
        remaining_steps = case.speech_size - 1 - speech_idx
        for text_idx in range(case.text_size):
            needed_advances = case.text_size - 1 - text_idx
            if needed_advances > remaining_steps:
                if (
                    beta[
                        0,
                        speech_idx,
                        text_idx,
                    ].item()
                    != NEG_LARGE
                ):
                    raise AssertionError(
                        "Beta reachability invariant failed " + f"at t={speech_idx}, j={text_idx}."
                    )

    log_z = terminal_log_z(alpha, mask)
    gamma = posterior_gamma(
        alpha,
        beta,
        log_z,
        mask,
    )

    dead = (alpha <= NEG_LARGE) | (beta <= NEG_LARGE)
    assert_exact_zero(
        "unreachable gamma",
        gamma.masked_select(dead),
    )


def test_determinism(
    case: Case,
    device: torch.device,
) -> None:
    print(f"\n[determinism] {case.name}")

    # log_b:
    # Shape: (B, T_speech, T_text)
    #
    # mask:
    # Shape: (B, T_speech, T_text)
    #
    # opt_sep_mask:
    # Shape: (B, T_text), or None
    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    with torch.no_grad():
        # alpha_0:
        # Shape: (B, T_speech, T_text)
        #
        # beta_0:
        # Shape: (B, T_speech, T_text)
        alpha_0, beta_0 = run_cuda(
            log_b,
            mask,
            opt_sep_mask,
        )

        for repetition in range(5):
            # alpha_i:
            # Shape: (B, T_speech, T_text)
            #
            # beta_i:
            # Shape: (B, T_speech, T_text)
            alpha_i, beta_i = run_cuda(
                log_b,
                mask,
                opt_sep_mask,
            )

            if not torch.equal(alpha_0, alpha_i):
                raise AssertionError(
                    "Alpha is not bitwise deterministic " + f"at repetition {repetition}."
                )

            if not torch.equal(beta_0, beta_i):
                raise AssertionError(
                    "Beta is not bitwise deterministic " + f"at repetition {repetition}."
                )

    # upstream_alpha:
    # Shape: (B, T_speech, T_text)
    upstream_alpha = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    # upstream_beta:
    # Shape: (B, T_speech, T_text)
    upstream_beta = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    def gradient_once() -> torch.Tensor:
        # x:
        # Shape: (B, T_speech, T_text)
        x = log_b.detach().clone().requires_grad_(True)

        # alpha:
        # Shape: (B, T_speech, T_text)
        #
        # beta:
        # Shape: (B, T_speech, T_text)
        alpha, beta = run_cuda(
            x,
            mask,
            opt_sep_mask,
        )

        # gradient:
        # Shape: (B, T_speech, T_text)
        return torch.autograd.grad(
            outputs=(alpha, beta),
            inputs=x,
            grad_outputs=(
                upstream_alpha,
                upstream_beta,
            ),
            retain_graph=False,
            create_graph=False,
        )[0]

    # gradient_0:
    # Shape: (B, T_speech, T_text)
    gradient_0 = gradient_once()

    for repetition in range(5):
        # gradient_i:
        # Shape: (B, T_speech, T_text)
        gradient_i = gradient_once()

        if not torch.equal(
            gradient_0,
            gradient_i,
        ):
            raise AssertionError(
                "Backward is not bitwise deterministic " + f"at repetition {repetition}."
            )


# ============================================================
# Numerical stability
# ============================================================


def test_numerical_stability(
    device: torch.device,
) -> None:
    print("\n[numerical-stability]")

    modes = (
        "normal",
        "wide",
        "very_negative",
        "near_equal",
        "dominant",
    )

    for mode_idx, mode in enumerate(modes):
        case = Case(
            name=f"numerical_{mode}",
            spec_lengths=(113, 79),
            text_lengths=(37, 29),
            speech_size=128,
            text_size=48,
            optional_positions=(
                (0, 5, 11, 20),
                (3, 9, 17),
            ),
            score_mode=mode,
            seed=500 + mode_idx,
        )

        log_b, mask, opt_sep_mask = make_case_tensors(
            case,
            device,
            torch.float32,
        )

        log_b = log_b.requires_grad_(True)
        alpha, beta = run_cuda(
            log_b,
            mask,
            opt_sep_mask,
        )

        valid_alpha = alpha.masked_select(mask)
        valid_beta = beta.masked_select(mask)

        if torch.isnan(valid_alpha).any():
            raise AssertionError(f"{mode}: alpha contains NaN.")
        if torch.isnan(valid_beta).any():
            raise AssertionError(f"{mode}: beta contains NaN.")
        if torch.isposinf(valid_alpha).any():
            raise AssertionError(f"{mode}: alpha contains +inf.")
        if torch.isposinf(valid_beta).any():
            raise AssertionError(f"{mode}: beta contains +inf.")

        upstream_alpha = torch.randn_like(alpha).masked_fill(~mask, 0.0)
        upstream_beta = torch.randn_like(beta).masked_fill(~mask, 0.0)

        gradient = torch.autograd.grad(
            outputs=(alpha, beta),
            inputs=log_b,
            grad_outputs=(
                upstream_alpha,
                upstream_beta,
            ),
        )[0]

        if not torch.isfinite(gradient).all():
            raise AssertionError(f"{mode}: backward contains nonfinite values.")

        print(f"{mode}: passed")


# ============================================================
# Performance
# ============================================================


def benchmark_cuda(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    """
    Returns:
        milliseconds per call
    """
    for _ in range(warmup):
        function()

    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        function()
    end.record()

    synchronize()

    return start.elapsed_time(end) / iterations


def benchmark_wall_clock(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    """
    Returns:
        milliseconds per call
    """
    for _ in range(warmup):
        function()

    synchronize()
    start = perf_counter()

    for _ in range(iterations):
        function()

    synchronize()
    elapsed = perf_counter() - start

    return elapsed * 1000.0 / iterations


def benchmark_case(
    case: Case,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> None:
    print(f"\n[benchmark] {case.name}")

    log_b, mask, opt_sep_mask = make_case_tensors(
        case,
        device,
        torch.float32,
    )

    # ============================================================
    # DP computation only
    # ============================================================

    def cuda_dp() -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        return run_cuda(
            log_b,
            mask,
            opt_sep_mask,
        )

    def reference_dp() -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            log_alpha_reference(
                log_b,
                mask,
                opt_sep_mask,
            ),
            log_beta_reference(
                log_b,
                mask,
                opt_sep_mask,
            ),
        )

    cuda_dp_ms = benchmark_wall_clock(
        cuda_dp,
        warmup=warmup,
        iterations=iterations,
    )

    reference_iterations = max(
        3,
        iterations // 10,
    )

    reference_dp_ms = benchmark_wall_clock(
        reference_dp,
        warmup=1,
        iterations=reference_iterations,
    )

    # ============================================================
    # DP computation + autograd
    # ============================================================

    upstream_alpha = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    upstream_beta = torch.randn_like(log_b).masked_fill(~mask, 0.0)

    def cuda_dp_autograd() -> torch.Tensor:
        x = log_b.detach().clone().requires_grad_(True)

        alpha, beta = run_cuda(
            x,
            mask,
            opt_sep_mask,
        )

        return torch.autograd.grad(
            outputs=(alpha, beta),
            inputs=x,
            grad_outputs=(
                upstream_alpha,
                upstream_beta,
            ),
            retain_graph=False,
            create_graph=False,
        )[0]

    def reference_dp_autograd() -> torch.Tensor:
        x = log_b.detach().clone().requires_grad_(True)

        alpha = log_alpha_reference(
            x,
            mask,
            opt_sep_mask,
        )
        beta = log_beta_reference(
            x,
            mask,
            opt_sep_mask,
        )

        return torch.autograd.grad(
            outputs=(alpha, beta),
            inputs=x,
            grad_outputs=(
                upstream_alpha,
                upstream_beta,
            ),
            retain_graph=False,
            create_graph=False,
        )[0]

    # 대표 configuration에서만 reference autograd benchmark.
    run_autograd_benchmark = log_b.size(0) == 1 and log_b.size(1) == 500 and log_b.size(2) == 100

    cuda_autograd_ms = benchmark_wall_clock(
        cuda_dp_autograd,
        warmup=max(2, warmup // 2),
        iterations=max(5, iterations // 4),
    )

    # ============================================================
    # Print
    # ============================================================

    print(f"shape: B={log_b.size(0)}, " + f"T={log_b.size(1)}, " + f"N={log_b.size(2)}")

    print(f"reference DP (alpha+beta): " + f"{reference_dp_ms:.3f} ms")
    print(f"CUDA DP      (alpha+beta): " + f"{cuda_dp_ms:.3f} ms")
    print(f"DP speedup:               " + f"{reference_dp_ms / cuda_dp_ms:.2f}x")

    if run_autograd_benchmark:
        reference_autograd_ms = benchmark_wall_clock(
            reference_dp_autograd,
            warmup=1,
            iterations=max(
                1,
                iterations // 20,
            ),
        )

        print(f"reference DP + autograd:   " + f"{reference_autograd_ms:.3f} ms")
        print(f"CUDA DP + autograd:        " + f"{cuda_autograd_ms:.3f} ms")
        print(f"autograd speedup:          " + f"{reference_autograd_ms / cuda_autograd_ms:.2f}x")
    else:
        print(f"CUDA DP + autograd:        " + f"{cuda_autograd_ms:.3f} ms")


# ============================================================
# Test suite
# ============================================================


def correctness_cases() -> tuple[Case, ...]:
    return (
        Case(
            name="minimal_1x1",
            spec_lengths=(1,),
            text_lengths=(1,),
            speech_size=1,
            text_size=1,
            optional_positions=None,
            seed=1,
        ),
        Case(
            name="optional_start_t1",
            spec_lengths=(1,),
            text_lengths=(2,),
            speech_size=1,
            text_size=2,
            optional_positions=((0,),),
            seed=2,
        ),
        Case(
            name="no_optional",
            spec_lengths=(9, 7),
            text_lengths=(5, 4),
            speech_size=9,
            text_size=5,
            optional_positions=None,
            seed=3,
        ),
        Case(
            name="optional_start_and_middle",
            spec_lengths=(9, 8),
            text_lengths=(6, 5),
            speech_size=9,
            text_size=6,
            optional_positions=(
                (0, 2, 4),
                (0, 3),
            ),
            seed=4,
        ),
        Case(
            name="variable_length_batch",
            spec_lengths=(17, 11, 8, 1),
            text_lengths=(9, 7, 5, 1),
            speech_size=17,
            text_size=9,
            optional_positions=(
                (0, 3, 6),
                (2, 5),
                (1, 3),
                (),
            ),
            seed=5,
        ),
        Case(
            name="stay_dominant_long_speech",
            spec_lengths=(64,),
            text_lengths=(7,),
            speech_size=64,
            text_size=7,
            optional_positions=((2, 4),),
            seed=6,
        ),
        Case(
            name="advance_dominant_t_near_n",
            spec_lengths=(10,),
            text_lengths=(10,),
            speech_size=10,
            text_size=10,
            optional_positions=None,
            seed=7,
        ),
        Case(
            name="skip_required",
            spec_lengths=(3,),
            text_lengths=(4,),
            speech_size=3,
            text_size=4,
            optional_positions=((1,),),
            seed=8,
        ),
    )


def benchmark_cases() -> tuple[Case, ...]:
    return (
        Case(
            name="benchmark_B1_T500_N100",
            spec_lengths=(500,),
            text_lengths=(100,),
            speech_size=500,
            text_size=100,
            optional_positions=(tuple(range(10, 100, 15)),),
            seed=1001,
        ),
        Case(
            name="benchmark_B8_T500_N100",
            spec_lengths=(
                500,
                487,
                463,
                441,
                419,
                397,
                373,
                349,
            ),
            text_lengths=(
                100,
                97,
                93,
                89,
                83,
                79,
                73,
                67,
            ),
            speech_size=500,
            text_size=100,
            optional_positions=tuple(
                tuple(range(10, length, 15))
                for length in (
                    100,
                    97,
                    93,
                    89,
                    83,
                    79,
                    73,
                    67,
                )
            ),
            seed=1002,
        ),
        Case(
            name="benchmark_B32_T1000_N200",
            spec_lengths=tuple(1000 - 7 * index for index in range(32)),
            text_lengths=tuple(200 - 3 * (index % 20) for index in range(32)),
            speech_size=1000,
            text_size=200,
            optional_positions=tuple(
                tuple(range(15, length, 23))
                for length in tuple(200 - 3 * (index % 20) for index in range(32))
            ),
            seed=1003,
        ),
    )


def run_correctness_suite(
    device: torch.device,
) -> None:
    for case in correctness_cases():
        test_forward_reference(
            case,
            device,
        )
        test_partition_and_gamma(
            case,
            device,
        )
        test_transition_posterior(
            case,
            device,
        )
        test_backward_reference(
            case,
            device,
        )
        test_backward_structural_zeros(
            case,
            device,
        )
        test_logz_gradient_equals_gamma(
            case,
            device,
        )
        test_batched_vs_single(
            case,
            device,
        )
        test_padding_invariance(
            case,
            device,
        )
        test_determinism(
            case,
            device,
        )

    test_exhaustive_enumeration(device)
    test_reachability(device)
    test_numerical_stability(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MonotoneForwardBackwardCUDA "
            "forward, backward, mathematical consistency, "
            "padding behavior, and performance."
        )
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the correctness and invariant suite.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip performance benchmarks.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="CUDA benchmark warm-up iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="CUDA benchmark iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device, e.g. cuda or cuda:1.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError(f"--device must be a CUDA device, but received {device}.")

    if device.index is None:  # type: ignore
        device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    else:
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Trigger extension compilation/loading outside benchmarks.
    smoke_case = correctness_cases()[0]
    smoke_log_b, smoke_mask, smoke_opt = make_case_tensors(
        smoke_case,
        device,
        torch.float32,
    )
    run_cuda(
        smoke_log_b,
        smoke_mask,
        smoke_opt,
    )
    synchronize()

    if not args.skip_correctness:
        run_correctness_suite(device)
        print("\nAll correctness tests passed.")

    if not args.skip_benchmark:
        for case in benchmark_cases():
            benchmark_case(
                case,
                device,
                warmup=args.warmup,
                iterations=args.iterations,
            )

    print("\nTest suite completed successfully.")


if __name__ == "__main__":
    main()
