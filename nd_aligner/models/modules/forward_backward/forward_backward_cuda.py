from __future__ import annotations

from pathlib import Path
from typing import Any, cast, override

import torch
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

# ============================================================
# CUDA extension loader
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent

_ext = None


def _load_ext():
    global _ext

    if _ext is not None:
        return _ext

    _ext = load(
        name="monotone_crf_forward_backward_ext",
        sources=[
            str(_THIS_DIR / "forward_backward.cpp"),
            str(_THIS_DIR / "log_alpha_beta.cu"),
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )

    return _ext


# ============================================================
# Input validation
# ============================================================


def _validate_inputs(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None,
) -> None:
    """
    Args:
        log_b:
            Log node potentials.
            Shape: (B, T_speech, T_text)
            Dtype: torch.float32
            Device: CUDA

        mask:
            Valid rectangular region for each batch item.
            Shape: (B, T_speech, T_text)
            Dtype: torch.bool
            Device: same as log_b

        opt_sep_mask:
            Optional-token indicators over the text axis.
            Shape: (B, T_text)
            Dtype: torch.bool
            Device: same as log_b
    """
    if not log_b.is_cuda:
        raise ValueError("log_b must be a CUDA tensor, " + f"but received device={log_b.device}.")

    if log_b.dtype != torch.float32:
        raise TypeError(
            "log_b must have dtype=torch.float32, " + f"but received dtype={log_b.dtype}."
        )

    if log_b.ndim != 3:
        raise ValueError(
            "log_b must have shape (B, T_speech, T_text), "
            + f"but received shape={tuple(log_b.shape)}."
        )

    batch_size, speech_size, text_size = log_b.shape

    if batch_size <= 0:
        raise ValueError("Batch size B must be positive.")

    if speech_size <= 0:
        raise ValueError("T_speech must be positive.")

    if text_size <= 0:
        raise ValueError("T_text must be positive.")

    if mask.device != log_b.device:
        raise ValueError(
            "mask and log_b must be on the same device, "
            + f"but received mask.device={mask.device} and "
            + f"log_b.device={log_b.device}."
        )

    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype=torch.bool, " + f"but received dtype={mask.dtype}.")

    if mask.shape != log_b.shape:
        raise ValueError(
            "mask must have shape (B, T_speech, T_text) "
            + "equal to log_b.shape, "
            + f"but received mask.shape={tuple(mask.shape)} and "
            + f"log_b.shape={tuple(log_b.shape)}."
        )

    if opt_sep_mask is None:
        return

    if opt_sep_mask.device != log_b.device:
        raise ValueError(
            "opt_sep_mask and log_b must be on the same device, "
            + f"but received opt_sep_mask.device={opt_sep_mask.device} "
            + f"and log_b.device={log_b.device}."
        )

    if opt_sep_mask.dtype != torch.bool:
        raise TypeError(
            "opt_sep_mask must have dtype=torch.bool, "
            + f"but received dtype={opt_sep_mask.dtype}."
        )

    expected_shape = (batch_size, text_size)

    if opt_sep_mask.shape != expected_shape:
        raise ValueError(
            "opt_sep_mask must have shape (B, T_text), "
            + f"but received shape={tuple(opt_sep_mask.shape)}; "
            + f"expected shape={expected_shape}."
        )


# ============================================================
# Custom autograd function
# ============================================================


class MonotoneForwardBackwardCUDA(torch.autograd.Function):
    """
    CUDA forward-backward dynamic programming with explicit backward
    kernels for log-alpha and log-beta.

    Inputs:
        log_b:
            Shape: (B, T_speech, T_text)
            Dtype: torch.float32
        mask:
            Shape: (B, T_speech, T_text)
            Dtype: torch.bool
        opt_sep_mask:
            Shape: (B, T_text), or None
            Dtype: torch.bool
        neg_large:
            Python float used as the finite approximation to -inf.

    Outputs:
        log_alpha:
            Shape: (B, T_speech, T_text)
            Dtype: torch.float32
        log_beta:
            Shape: (B, T_speech, T_text)
            Dtype: torch.float32
    """

    @staticmethod
    @override
    def forward(
        ctx: Any,
        log_b: torch.Tensor,
        mask: torch.Tensor,
        opt_sep_mask: torch.Tensor | None = None,
        neg_large: float = -1.0e9,
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
        _validate_inputs(
            log_b=log_b,
            mask=mask,
            opt_sep_mask=opt_sep_mask,
        )

        ext = _load_ext()

        # log_b_contiguous:
        # Shape: (B, T_speech, T_text)
        log_b_contiguous = log_b.contiguous()

        # mask_contiguous:
        # Shape: (B, T_speech, T_text)
        mask_contiguous = mask.contiguous()

        if opt_sep_mask is not None:
            # opt_sep_contiguous:
            # Shape: (B, T_text)
            opt_sep_contiguous = opt_sep_mask.contiguous()

            has_opt_sep = True
        else:
            # Placeholder saved because save_for_backward cannot save None.
            # Shape: (0,)
            opt_sep_contiguous = torch.empty(
                (0,),
                dtype=torch.bool,
                device=log_b.device,
            )

            has_opt_sep = False

        extension_opt_sep_mask = opt_sep_contiguous if has_opt_sep else None

        # log_alpha:
        # Shape: (B, T_speech, T_text)
        log_alpha = ext.log_alpha_forward(  # type: ignore
            log_b_contiguous,
            mask_contiguous,
            extension_opt_sep_mask,
            float(neg_large),
        )

        # log_beta:
        # Shape: (B, T_speech, T_text)
        log_beta = ext.log_beta_forward(  # type: ignore
            log_b_contiguous,
            mask_contiguous,
            extension_opt_sep_mask,
            float(neg_large),
        )

        ctx.save_for_backward(
            log_b_contiguous,  # Shape: (B, T_speech, T_text)
            log_alpha,  # Shape: (B, T_speech, T_text)
            log_beta,  # Shape: (B, T_speech, T_text)
            mask_contiguous,  # Shape: (B, T_speech, T_text)
            opt_sep_contiguous,  # Shape: (B, T_text) or (0,)
        )

        ctx.has_opt_sep = has_opt_sep
        ctx.neg_large = float(neg_large)

        return log_alpha, log_beta

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_log_alpha: torch.Tensor | None,
        grad_log_beta: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        """
        Args:
            grad_log_alpha:
                Upstream gradient for log_alpha.
                Shape: (B, T_speech, T_text), or None

            grad_log_beta:
                Upstream gradient for log_beta.
                Shape: (B, T_speech, T_text), or None

        Returns:
            grad_log_b:
                Total gradient from the alpha and beta branches.
                Shape: (B, T_speech, T_text)

            None:
                mask is non-differentiable.

            None:
                opt_sep_mask is non-differentiable.

            None:
                neg_large is non-differentiable.
        """
        (
            log_b,  # Shape: (B, T_speech, T_text)
            log_alpha,  # Shape: (B, T_speech, T_text)
            log_beta,  # Shape: (B, T_speech, T_text)
            mask,  # Shape: (B, T_speech, T_text)
            saved_opt_sep_mask,  # Shape: (B, T_text) or (0,)
        ) = ctx.saved_tensors

        if not ctx.needs_input_grad[0]:
            return None, None, None, None

        ext = _load_ext()

        opt_sep_mask = saved_opt_sep_mask if ctx.has_opt_sep else None

        # grad_log_b:
        # Shape: (B, T_speech, T_text)
        #
        # This accumulates:
        #   dL / d log_b
        #       = dL_alpha / d log_b
        #       + dL_beta  / d log_b
        grad_log_b = torch.zeros_like(log_b)

        if grad_log_alpha is not None:
            # grad_log_alpha_contiguous:
            # Shape: (B, T_speech, T_text)
            grad_log_alpha_contiguous = grad_log_alpha.contiguous()

            # grad_log_b_from_alpha:
            # Shape: (B, T_speech, T_text)
            grad_log_b_from_alpha = ext.log_alpha_backward(  # type: ignore
                grad_log_alpha_contiguous,
                log_b,
                log_alpha,
                mask,
                opt_sep_mask,
                ctx.neg_large,
            )

            grad_log_b.add_(grad_log_b_from_alpha)

        if grad_log_beta is not None:
            # grad_log_beta_contiguous:
            # Shape: (B, T_speech, T_text)
            grad_log_beta_contiguous = grad_log_beta.contiguous()

            # grad_log_b_from_beta:
            # Shape: (B, T_speech, T_text)
            grad_log_b_from_beta = ext.log_beta_backward(  # type: ignore
                grad_log_beta_contiguous,
                log_b,
                log_beta,
                mask,
                opt_sep_mask,
                ctx.neg_large,
            )

            grad_log_b.add_(grad_log_b_from_beta)

        return (
            grad_log_b,  # Shape: (B, T_speech, T_text)
            None,  # mask
            None,  # opt_sep_mask
            None,  # neg_large
        )


# ============================================================
# Public functional API
# ============================================================


def monotone_crf_forward_backward(
    log_b: torch.Tensor,
    mask: torch.Tensor,
    opt_sep_mask: torch.Tensor | None = None,
    neg_large: float = -1.0e9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        log_b:
            Shape: (B, T_speech, T_text)
            Dtype: torch.float32
            Device: CUDA
        mask:
            Shape: (B, T_speech, T_text)
            Dtype: torch.bool
            Device: CUDA
        opt_sep_mask:
            Shape: (B, T_text), or None
            Dtype: torch.bool
            Device: CUDA
        neg_large:
            Finite approximation to negative infinity.

    Returns:
        log_alpha:
            Shape: (B, T_speech, T_text)
        log_beta:
            Shape: (B, T_speech, T_text)
    """
    out = cast(
        tuple[torch.Tensor, torch.Tensor],
        MonotoneForwardBackwardCUDA.apply(
            log_b,
            mask,
            opt_sep_mask,
            float(neg_large),
        ),
    )
    return out
