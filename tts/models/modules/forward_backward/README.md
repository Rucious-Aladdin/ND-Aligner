# CUDA Monotone Forward–Backward

CUDA implementation of log-space forward–backward dynamic programming for a monotone alignment path with optional separator skips.

The implementation operates on a rectangular speech–text grid and supports three transitions:

- **stay:** `j -> j`
- **advance:** `j -> j + 1`
- **optional skip:** `j -> j + 2`, allowed only when the skipped token `j + 1` is marked optional

Both `log_alpha` and `log_beta` have explicit CUDA backward kernels, and the Python autograd wrapper accumulates their two gradient contributions with respect to the shared `log_b` input.

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

The validity mask must be rectangular:

$$
\operatorname{mask}_{b,t,j}
=
\operatorname{spec\_mask}_{b,t}
\land
\operatorname{text\_mask}_{b,j}.
$$

Each CUDA block handles one batch item. The actual speech and text lengths are recovered from the first column and first row of this mask.

## Notation

For one sample, let

$$
\ell_{t,j} \equiv \log b_{t,j},
\qquad
A_{t,j} \equiv \log \alpha_{t,j},
\qquad
B_{t,j} \equiv \log \beta_{t,j}.
$$

Invalid and unreachable states are represented by `neg_large`.

For valid candidates $x_1,\ldots,x_m$,

$$
\operatorname{LSE}(x_1,\ldots,x_m)
=
\log\sum_{r=1}^{m}e^{x_r}.
$$

## Forward: log-alpha

The initial row is

$$
A_{0,0}=\ell_{0,0}.
$$

When token $0$ is optional, the implementation also permits

$$
A_{0,1}=\ell_{0,1}.
$$

All other initial states are `neg_large`.

For $t\ge1$,

$$
A_{t,k}
=
\ell_{t,k}
+
\operatorname{LSE}
\left(
A_{t-1,k},
A_{t-1,k-1},
A_{t-1,k-2}
\right),
$$

where the second candidate exists only for $k\ge1$, and the third candidate exists only when $k\ge2$ and token $k-1$ is optional.

## Forward: log-beta

The terminal row is initialized by

$$
B_{T-1,N-1}=0,
$$

with all other terminal states set to `neg_large`.

For $t<T-1$,

$$
B_{t,j}
=
\operatorname{LSE}
\left(
q_{t,j\rightarrow j},
q_{t,j\rightarrow j+1},
q_{t,j\rightarrow j+2}
\right),
$$

where

$$
q_{t,j\rightarrow k}
=
\ell_{t+1,k}+B_{t+1,k}.
$$

The skip candidate $j\rightarrow j+2$ exists only when token $j+1$ is optional.

## Stable LSE derivative

Both backward kernels use the closed-form derivative of log-sum-exp. For candidate $x_i$,

$$
\frac{\partial\operatorname{LSE}(x_1,\ldots,x_m)}{\partial x_i}
=
\frac{e^{x_i-m}}{\sum_r e^{x_r-m}},
\qquad
m=\max_r x_r.
$$

The CUDA implementation evaluates this derivative using a max-shifted local softmax. It does not reconstruct a weight from a stored successor value such as

```text
exp(predecessor + log_b - successor)
```

because that form suffers from cancellation error when the accumulated log scores have large magnitude.

## Backward: log-alpha

Let $F_\alpha$ denote all computation after the `log_alpha` output, and define

$$
D^{(\alpha)}_{t,k}
\equiv
\frac{dL}{dA_{t,k}}.
$$

The alpha gradient recurrence is

$$
\boxed{
\begin{aligned}
D^{(\alpha)}_{t,k}
={}&
\underbrace{
\frac{\partial F_\alpha}{\partial A_{t,k}}
}_{\text{upstream}}
\\
&+
D^{(\alpha)}_{t+1,k}
\frac{\partial A_{t+1,k}}{\partial A_{t,k}}
\\
&+
\mathbf 1[k+1<N]\,
D^{(\alpha)}_{t+1,k+1}
\frac{\partial A_{t+1,k+1}}{\partial A_{t,k}}
\\
&+
\mathbf 1[
 k+2<N\land\operatorname{optional}(k+1)
]\,
D^{(\alpha)}_{t+1,k+2}
\frac{\partial A_{t+1,k+2}}{\partial A_{t,k}}.
\end{aligned}
}
$$

Each local partial derivative is the corresponding successor-state LSE weight.

Because $\ell_{t,k}$ is added directly to $A_{t,k}$ for every normally recurrent row,

$$
\boxed{
g^{(\alpha)}_{b,t,k}
\equiv
\frac{dL}{d\ell_{t,k}}
=
D^{(\alpha)}_{t,k},
\qquad t\ge1.
}
$$

At $t=0$, this direct dependency exists only for the explicitly initialized state $k=0$ and, when enabled, the optional-start state $k=1$. All other first-row gradients are zero.

The alpha backward kernel runs in reverse speech-time order.

## Backward: log-beta

Let $F_\beta$ denote all computation after the `log_beta` output, and define

$$
D^{(\beta)}_{t,k}
\equiv
\frac{dL}{dB_{t,k}},
$$

and the beta-branch gradient with respect to `log_b` as

$$
g^{(\beta)}_{b,t,k}
\equiv
\frac{dL}{d\ell_{t,k}}.
$$

Since $\ell_{t,k}$ and $B_{t,k}$ enter every preceding beta candidate through the same sum

$$
q_{t-1,j\rightarrow k}
=
\ell_{t,k}+B_{t,k},
$$

the total beta-state gradient is

$$
\boxed{
D^{(\beta)}_{t,k}
=
\underbrace{
\frac{\partial F_\beta}{\partial B_{t,k}}
}_{\text{upstream}}
+
\underbrace{
g^{(\beta)}_{b,t,k}
}_{\text{accumulated recurrence gradient}}.
}
$$

The next-row `log_b` gradient is gathered from the three possible predecessor states:

$$
\boxed{
\begin{aligned}
g^{(\beta)}_{b,t+1,k}
={}&
D^{(\beta)}_{t,k}
\frac{\partial B_{t,k}}
     {\partial q_{t,k\rightarrow k}}
\\
&+
\mathbf 1[k\ge1]\,
D^{(\beta)}_{t,k-1}
\frac{\partial B_{t,k-1}}
     {\partial q_{t,k-1\rightarrow k}}
\\
&+
\mathbf 1[
 k\ge2\land\operatorname{optional}(k-1)
]\,
D^{(\beta)}_{t,k-2}
\frac{\partial B_{t,k-2}}
     {\partial q_{t,k-2\rightarrow k}}.
\end{aligned}
}
$$

Each local partial derivative is the corresponding source-state LSE weight.

The initial condition is

$$
\boxed{
g^{(\beta)}_{b,0,k}=0,
\qquad
D^{(\beta)}_{0,k}
=
\frac{\partial F_\beta}{\partial B_{0,k}}.
}
$$

The beta recurrence never consumes `log_b[:, 0, :]`, so the beta-branch gradient on the first speech row is exactly zero.

The beta backward kernel runs in forward speech-time order.

## Combined gradient

Both dynamic programs consume the same `log_b`. Their branch gradients therefore add:

$$
\boxed{
\frac{dL}{d\ell_{t,k}}
=
g^{(\alpha)}_{b,t,k}
+
g^{(\beta)}_{b,t,k}.
}
$$

`MonotoneForwardBackwardCUDA.backward()` invokes the alpha and beta CUDA backward kernels as needed and explicitly accumulates both returned tensors into one `grad_log_b`.

## Kernel structure

The implementation uses one CUDA block per batch item. Threads within a block parallelize over the text dimension, while the speech-time recurrence remains sequential.

Shared memory stores:

- two rolling DP or gradient rows;
- local max-shift softmax normalization data.

The backward kernels avoid atomic updates by assigning each thread one state and gathering all valid incoming gradient contributions.

## Tests

`test_forward_backward.py` provides a combined correctness and performance suite. It covers the main implementation properties without requiring separate alpha-only and beta-only test files, including:

- CUDA forward and backward results against PyTorch references;
- optional skips, variable sequence lengths, padding, and structural-zero gradients;
- consistency of `logZ`, posterior occupancy, and the identity `d logZ / d log_b = gamma`;
- deterministic and numerically stable behavior over representative score ranges;
- forward and backward timing on small and larger batch/grid configurations.

Run the test as a package module from the repository root:

```bash
python -m tts.models.modules.forward_backward.test_forward_backward
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
