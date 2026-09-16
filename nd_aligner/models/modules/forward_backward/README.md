# CUDA Monotone Forward–Backward

CUDA implementation of log-space forward–backward dynamic programming for a monotone alignment path with optional separator skips.

The implementation operates on a rectangular speech–text grid and supports three transitions:

- **stay:** `j -> j`
- **advance:** `j -> j + 1`
- **optional skip:** `j -> j + 2`, allowed only when the skipped token `j + 1` is marked optional

Both `log_alpha` and `log_beta` have explicit CUDA backward kernels, and the Python autograd wrapper accumulates their two gradient contributions with respect to the shared `log_b` input.

The recursions, their gradients, and the identities the tests rely on are derived in **`forward_backward_derivations.pdf`**. This file records what the kernels compute and how to call them; anything of the form "why is this recursion correct" is answered there.

## Files

- `forward_backward_cuda.py`: Python interface and custom autograd integration
- `forward_backward.cpp`: PyTorch C++ bindings
- `log_alpha_beta.cu`: CUDA forward and backward kernels
- `test_forward_backward.py`: reference, gradient, invariant, and timing tests

## Inputs

```text
log_b:        float32, (B, T_speech, T_text)
mask:         bool,    (B, T_speech, T_text)
opt_sep_mask: bool,    (B, T_text), optional
neg_large:    finite log-zero value, normally -1e9
```

`log_b` holds the log node potentials of the grid, one per speech–text pair. Invalid and unreachable states are represented by `neg_large` rather than by `-inf`, so that the arithmetic stays finite; the kernels test against this sentinel rather than doing arithmetic on it. The validity mask must be rectangular, that is, the outer product of a speech mask and a text mask. Each CUDA block handles one batch item, and the actual speech and text lengths are recovered from the first column and first row of this mask.

## What the kernels compute

**Forward (`log_alpha`).** For each cell of the grid, the log-sum of all admissible path prefixes ending there, including the potential of the cell itself. A path starts on token 0, or on token 1 when token 0 is optional, and the recursion accumulates one frame at a time over the three transitions above. The last cell of the last valid row is `log Z`, the log-sum over all monotone paths.

**Backward (`log_beta`).** Symmetrically, the log-sum of all admissible path suffixes starting at a cell and ending at the terminal cell, *excluding* the potential of the starting cell. That exclusion is a convention, and it is what makes `log_alpha + log_beta - log_z` the posterior marginal of a cell with no correction factor.

**Backward(=Gradient Computing) kernels.** Each recursion is differentiated with respect to `log_b`, producing two gradient contributions that are summed. The two branches have different shapes. In the alpha recursion the potential is added outside the log-sum-exp, so the gradient with respect to a potential coincides with the adjoint of the forward variable it is added to, and the recursion runs in reverse speech-time order. In the beta recursion the potential sits inside the log-sum-exp, paired with a backward variable of the next frame, so two quantities must be advanced together, and the recursion runs in forward speech-time order.

Both backward kernels evaluate the log-sum-exp derivative as a max-shifted local softmax over the candidate list. They do not reconstruct a weight from stored recursion outputs as `exp(predecessor + log_b - successor)`: that form loses precision to cancellation once the accumulated log scores are large, and it returns a spurious nonzero weight when both operands are the `neg_large` sentinel.

## Kernel structure

One CUDA block per batch item. Threads within a block parallelize over the text dimension, while the speech-time recurrence remains sequential.

## Tests

`test_forward_backward.py` provides a combined correctness and performance suite
covering:

- CUDA forward and backward results against PyTorch references;
- brute-force enumeration of all monotone paths on small grids, with and without optional skips;
- optional skips, variable sequence lengths, padding, and structural-zero gradients;
- consistency of `logZ` between the two passes, posterior occupancy, and the identity `d logZ / d log_b = gamma`;
- deterministic and numerically stable behavior over representative score ranges;
- forward and backward timing on small and larger batch/grid configurations.

Run the test as a package module from the repository root:

```bash
python -m nd_aligner.models.modules.forward_backward.test_forward_backward
```

Useful options include:

```bash
--skip-correctness
--skip-benchmark
--warmup <int>
--iterations <int>
--device cuda[:index]
```

## Implementation status

- [x] `log_alpha` forward CUDA kernel
- [x] `log_beta` forward CUDA kernel
- [x] `log_alpha` backward CUDA kernel
- [x] `log_beta` backward CUDA kernel
- [x] combined Python custom-autograd integration
- [x] reference, invariant, numerical-stability, and benchmark test suite
