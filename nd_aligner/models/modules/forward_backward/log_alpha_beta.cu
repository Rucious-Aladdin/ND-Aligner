#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS 256

namespace {

struct SequenceLengths
{
  int spec_length;
  int txt_length;
};

__device__ __forceinline__ SequenceLengths
recover_lengths(const bool* __restrict__ mask,
                const int batch_offset,
                const int speech_size,
                const int txt_size)
{
  SequenceLengths lengths{
    0,
    0,
  };

  for (int speech_idx = 0; speech_idx < speech_size; ++speech_idx) {
    const int index = batch_offset + speech_idx * txt_size;
    if (!mask[index]) {
      break;
    }
    ++lengths.spec_length;
  }

  for (int txt_idx = 0; txt_idx < txt_size; ++txt_idx) {
    const int index = batch_offset + txt_idx;
    if (!mask[index]) {
      break;
    }
    ++lengths.txt_length;
  }

  return lengths;
}

__device__ __forceinline__ float
logaddexp_pair(const float a, const float b, const float neg_large)
{
  if (a <= neg_large && b <= neg_large) {
    return neg_large;
  }

  const float max_value = fmaxf(a, b);

  return max_value + log1pf(expf(-fabsf(a - b)));
}

struct LocalSoftmaxNorm
{
  float max_value;
  float inv_exp_sum;
};

__device__ __forceinline__ LocalSoftmaxNorm
compute_local_softmax_norm(const float first,
                           const float second,
                           const float third,
                           const bool use_third,
                           const float neg_large)
{
  float max_value = fmaxf(first, second);

  if (use_third) {
    max_value = fmaxf(max_value, third);
  }

  if (max_value <= neg_large) {
    return {
      neg_large,
      0.0f,
    };
  }

  float exp_sum = 0.0f;

  if (first > neg_large) {
    exp_sum += expf(first - max_value);
  }

  if (second > neg_large) {
    exp_sum += expf(second - max_value);
  }

  if (use_third && third > neg_large) {
    exp_sum += expf(third - max_value);
  }

  return {
    max_value,
    1.0f / exp_sum,
  };
}

__device__ __forceinline__ float
local_softmax_weight(const float value, const LocalSoftmaxNorm norm, const float neg_large)
{
  if (value <= neg_large || norm.inv_exp_sum == 0.0f) {
    return 0.0f;
  }

  return expf(value - norm.max_value) * norm.inv_exp_sum;
}

// ============================================================
// Forward DP: log_alpha
// ============================================================

__global__ void
log_alpha_forward_kernel(const float* __restrict__ log_b,
                         const bool* __restrict__ mask,
                         const bool* __restrict__ opt_sep_mask,
                         float* __restrict__ log_alpha,
                         const int batch_size,
                         const int speech_size,
                         const int txt_size,
                         const float neg_large)
{
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  /*
   * Shared-memory layout:
   *
   * prev_alpha: [txt_size]
   * curr_alpha: [txt_size]
   */
  extern __shared__ float shared_memory[];

  float* prev_alpha = shared_memory;
  float* curr_alpha = shared_memory + txt_size;

  const int log_b_offset = batch_idx * speech_size * txt_size;
  const int sep_offset = batch_idx * txt_size;
  const bool has_opt_sep = opt_sep_mask != nullptr;

  // --------------------------------------------------------
  // Initialize alpha at t = 0.
  //
  // Normal start:
  //     alpha[0, 0] = log_b[0, 0]
  //
  // Optional first-token skip:
  //     alpha[0, 1] = log_b[0, 1]
  //     iff opt_sep_mask[0] is true.
  //
  // All other states remain neg_large.
  // --------------------------------------------------------

  for (int txt_idx = threadIdx.x; txt_idx < txt_size; txt_idx += blockDim.x) {
    const int index = log_b_offset + txt_idx;
    float value = neg_large;

    if (mask[index]) {
      if (txt_idx == 0) {
        value = log_b[index];
      } else if (txt_idx == 1 && has_opt_sep && opt_sep_mask[sep_offset]) {
        value = log_b[index];
      }
    }

    prev_alpha[txt_idx] = value;
    log_alpha[index] = value;
  }
  __syncthreads();

  // --------------------------------------------------------
  // Forward recursion over speech time.
  // --------------------------------------------------------

  for (int speech_idx = 1; speech_idx < speech_size; ++speech_idx) {
    const int row_offset = log_b_offset + speech_idx * txt_size;
    for (int txt_idx = threadIdx.x; txt_idx < txt_size; txt_idx += blockDim.x) {
      const int index = row_offset + txt_idx;
      float alpha_value = neg_large;

      if (mask[index]) {
        // stay: j -> j
        const float stay = prev_alpha[txt_idx];
        // advance: j - 1 -> j
        const float advance = txt_idx >= 1 ? prev_alpha[txt_idx - 1] : neg_large;

        float predecessor_sum = logaddexp_pair(stay, advance, neg_large);

        // skip: j - 2 -> j
        // Allowed iff the skipped token j - 1 is optional.
        if (has_opt_sep && txt_idx >= 2 && opt_sep_mask[sep_offset + txt_idx - 1]) {
          const float skip = prev_alpha[txt_idx - 2];
          predecessor_sum = logaddexp_pair(predecessor_sum, skip, neg_large);
        }

        alpha_value = log_b[index] + predecessor_sum;
      }

      curr_alpha[txt_idx] = alpha_value;
      log_alpha[index] = alpha_value;
    }
    __syncthreads();

    float* temporary = prev_alpha;
    prev_alpha = curr_alpha;
    curr_alpha = temporary;

    __syncthreads();
  }
}

// ============================================================
// Backward DP: log_beta
// ============================================================

__global__ void
log_beta_forward_kernel(const float* __restrict__ log_b,
                        const bool* __restrict__ mask,
                        const bool* __restrict__ opt_sep_mask,
                        float* __restrict__ log_beta,
                        const int batch_size,
                        const int speech_size,
                        const int txt_size,
                        const float neg_large)
{
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  /*
   * Shared-memory layout:
   *
   * next_beta: [txt_size]
   * curr_beta: [txt_size]
   */
  extern __shared__ float shared_memory[];

  float* next_beta = shared_memory;
  float* curr_beta = shared_memory + txt_size;

  __shared__ int spec_length;
  __shared__ int txt_length;

  const int log_b_offset = batch_idx * speech_size * txt_size;
  const int sep_offset = batch_idx * txt_size;
  const bool has_opt_sep = opt_sep_mask != nullptr;

  if (threadIdx.x == 0) {
    const SequenceLengths lengths = recover_lengths(mask, log_b_offset, speech_size, txt_size);
    spec_length = lengths.spec_length;
    txt_length = lengths.txt_length;
  }
  __syncthreads();

  const int terminal_speech_idx = spec_length - 1;
  const int terminal_txt_idx = txt_length - 1;
  const int terminal_row_offset = log_b_offset + terminal_speech_idx * txt_size;

  // --------------------------------------------------------
  // Initialize beta at the terminal speech frame.
  //
  // beta[t_end, j_end] = 0
  // all other states    = neg_large
  // --------------------------------------------------------
  for (int txt_idx = threadIdx.x; txt_idx < txt_size; txt_idx += blockDim.x) {
    float value = neg_large;

    if (txt_idx == terminal_txt_idx) {
      const int terminal_index = terminal_row_offset + txt_idx;

      if (mask[terminal_index]) {
        value = 0.0f;
      }
    }

    next_beta[txt_idx] = value;
    log_beta[terminal_row_offset + txt_idx] = value;
  }
  __syncthreads();

  // --------------------------------------------------------
  // Backward recursion.
  // --------------------------------------------------------
  for (int speech_idx = terminal_speech_idx - 1; speech_idx >= 0; --speech_idx) {
    const int curr_row_offset = log_b_offset + speech_idx * txt_size;
    const int next_row_offset = log_b_offset + (speech_idx + 1) * txt_size;
    for (int txt_idx = threadIdx.x; txt_idx < txt_size; txt_idx += blockDim.x) {
      const int curr_idx = curr_row_offset + txt_idx;
      float beta_value = neg_large;
      if (mask[curr_idx]) {
        // stay: j -> j
        const float stay = log_b[next_row_offset + txt_idx] + next_beta[txt_idx];

        // advance: j -> j + 1
        const float advance = txt_idx + 1 < txt_size
                                ? (log_b[next_row_offset + txt_idx + 1] + next_beta[txt_idx + 1])
                                : neg_large;

        beta_value = logaddexp_pair(stay, advance, neg_large);

        // skip: j -> j + 2
        //
        // Allowed iff the skipped token j + 1 is optional.
        if (has_opt_sep && txt_idx + 2 < txt_size && opt_sep_mask[sep_offset + txt_idx + 1]) {
          const float skip = log_b[next_row_offset + txt_idx + 2] + next_beta[txt_idx + 2];
          beta_value = logaddexp_pair(beta_value, skip, neg_large);
        }
      }
      curr_beta[txt_idx] = beta_value;
      log_beta[curr_idx] = beta_value;
    }
    __syncthreads();

    float* temporary = next_beta;
    next_beta = curr_beta;
    curr_beta = temporary;

    __syncthreads();
  }
}

// ============================================================
// Autograd backward: log_alpha
// ============================================================

__global__ void
log_alpha_backward_kernel(const float* __restrict__ grad_log_alpha,
                          const float* __restrict__ log_b,
                          const float* __restrict__ log_alpha,
                          const bool* __restrict__ mask,
                          const bool* __restrict__ opt_sep_mask,
                          float* __restrict__ grad_log_b,
                          const int batch_size,
                          const int speech_size,
                          const int txt_size,
                          const float neg_large)
{
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  /*
   * Shared-memory layout:
   *
   * next_delta: [txt_size]
   * curr_delta: [txt_size]
   *
   * next_delta[j]
   *     = dL / d log_alpha[t + 1, j]
   *
   * curr_delta[j]
   *     = dL / d log_alpha[t, j]
   */
  extern __shared__ float shared_memory[];

  float* next_delta = shared_memory;
  float* curr_delta = shared_memory + txt_size;

  LocalSoftmaxNorm* successor_norm =
    reinterpret_cast<LocalSoftmaxNorm*>(shared_memory + 2 * txt_size);

  __shared__ int spec_length;
  __shared__ int txt_length;

  const int batch_offset = batch_idx * speech_size * txt_size;
  const int sep_offset = batch_idx * txt_size;
  const bool has_opt_sep = opt_sep_mask != nullptr;

  // --------------------------------------------------------
  // Recover the actual sequence lengths for this batch item.
  //
  // Only thread 0 performs the scan. All blocks perform their
  // own scans concurrently for their corresponding samples.
  // --------------------------------------------------------

  if (threadIdx.x == 0) {
    const SequenceLengths lengths = recover_lengths(mask, batch_offset, speech_size, txt_size);

    spec_length = lengths.spec_length;
    txt_length = lengths.txt_length;
  }
  __syncthreads();

  // The operator assumes non-empty speech/txt sequences,
  // consistently with the existing beta-forward kernel.

  const int last_speech_idx = spec_length - 1;
  const int last_row_offset = batch_offset + last_speech_idx * txt_size;

  // --------------------------------------------------------
  // Initialize the adjoint at the final valid speech row.
  //
  // There is no alpha row after this row, so:
  //
  //     delta[T_b - 1, j]
  //         = grad_log_alpha[T_b - 1, j]
  //
  // where T_b is the actual speech length of this sample.
  // --------------------------------------------------------
  for (int txt_idx = threadIdx.x; txt_idx < txt_length; txt_idx += blockDim.x) {
    const int index = last_row_offset + txt_idx;
    float delta = 0.0f;

    if (mask[index]) {
      delta = grad_log_alpha[index];
    }
    next_delta[txt_idx] = delta;

    // If spec_length > 1, the last row was produced by:
    //     log_alpha[t, j] = log_b[t, j] + predecessor_sum
    // and therefore directly depends on log_b[t, j].
    //
    // If spec_length == 1, this is t == 0 and the special
    // alpha initialization rule must be used.
    if (spec_length > 1) {
      grad_log_b[index] = delta;
    } else {
      bool depends_on_log_b = false;

      if (mask[index]) {
        // Normal initial state:
        //     alpha[0, 0] = log_b[0, 0]
        if (txt_idx == 0) {
          depends_on_log_b = true;
        }

        // Optional first-token skip:
        //     alpha[0, 1] = log_b[0, 1]
        // iff token 0 is optional.
        else if (txt_idx == 1 && has_opt_sep && opt_sep_mask[sep_offset]) {
          depends_on_log_b = true;
        }
      }
      grad_log_b[index] = depends_on_log_b ? delta : 0.0f;
    }
  }
  __syncthreads();

  // --------------------------------------------------------
  // Reverse recursion over the actual speech length.
  //
  // For state alpha[t, j], gather contributions from:
  //
  //     alpha[t + 1, j]      stay
  //     alpha[t + 1, j + 1]  advance
  //     alpha[t + 1, j + 2]  optional skip
  //
  // Each thread writes only curr_delta[j], so atomicAdd is
  // unnecessary.
  // --------------------------------------------------------

  for (int speech_idx = spec_length - 2; speech_idx >= 0; --speech_idx) {
    const int curr_row_offset = batch_offset + speech_idx * txt_size;

    // improve numerical precision
    for (int succ_txt_idx = threadIdx.x; succ_txt_idx < txt_length; succ_txt_idx += blockDim.x) {
      // Predecessors of alpha[t + 1, k].
      const float stay = log_alpha[curr_row_offset + succ_txt_idx];

      const float advance =
        succ_txt_idx >= 1 ? log_alpha[curr_row_offset + succ_txt_idx - 1] : neg_large;

      const bool skip_allowed =
        has_opt_sep && succ_txt_idx >= 2 && opt_sep_mask[sep_offset + succ_txt_idx - 1];

      const float skip = skip_allowed ? log_alpha[curr_row_offset + succ_txt_idx - 2] : neg_large;

      successor_norm[succ_txt_idx] =
        compute_local_softmax_norm(stay, advance, skip, skip_allowed, neg_large);
    }
    __syncthreads();

    for (int txt_idx = threadIdx.x; txt_idx < txt_length; txt_idx += blockDim.x) {
      const int curr_idx = curr_row_offset + txt_idx;
      float delta = 0.0f;

      if (mask[curr_idx]) {
        // Gradient entering this log_alpha output from
        // operations outside the custom CUDA operator.
        delta = grad_log_alpha[curr_idx];

        const float curr_alpha = log_alpha[curr_idx];
        const bool curr_reachable = curr_alpha > neg_large;

        // ------------------------------------------------
        // Stay: alpha[t, j] -> alpha[t + 1, j]
        // successor -> (t+1, j)
        // current -> (t, j)
        // next_delta -> (t+1, j)
        // ------------------------------------------------
        if (curr_reachable) {
          const int succ_txt_idx = txt_idx;
          const float weight =
            local_softmax_weight(curr_alpha, successor_norm[succ_txt_idx], neg_large);
          delta += next_delta[succ_txt_idx] * weight;
        }

        // ------------------------------------------------
        // Advance: alpha[t, j] -> alpha[t + 1, j + 1]
        // successor -> (t+1, j+1)
        // current -> (t, j)
        // next_delta -> (t+1, j+1)
        // ------------------------------------------------
        if (curr_reachable && txt_idx + 1 < txt_length) {
          const int succ_txt_idx = txt_idx + 1;
          const float weight =
            local_softmax_weight(curr_alpha, successor_norm[succ_txt_idx], neg_large);
          delta += next_delta[succ_txt_idx] * weight;
        }

        // ------------------------------------------------
        // Optional skip: alpha[t, j] -> alpha[t + 1, j + 2]
        // successor -> (t+1, j+2)
        // current -> (t, j)
        // next_delta -> (t+1, j+2)
        // ------------------------------------------------
        if (curr_reachable && has_opt_sep && txt_idx + 2 < txt_length &&
            opt_sep_mask[sep_offset + txt_idx + 1]) {
          const int succ_txt_idx = txt_idx + 2;
          const float weight =
            local_softmax_weight(curr_alpha, successor_norm[succ_txt_idx], neg_large);
          delta += next_delta[succ_txt_idx] * weight;
        }
      }
      curr_delta[txt_idx] = delta;

      // ----------------------------------------------------
      // Direct gradient to log_b.
      //
      // For t >= 1:
      //
      //     log_alpha[t, j]
      //         = log_b[t, j] + predecessor_sum
      //
      // so:
      //
      //     dL / d log_b[t, j] = delta[t, j].
      //
      // At t == 0, only the explicitly initialized states
      // depend on log_b.
      // ----------------------------------------------------

      if (speech_idx >= 1) {
        grad_log_b[curr_idx] = mask[curr_idx] ? delta : 0.0f;
      } else {
        bool depends_on_log_b = false;

        if (mask[curr_idx]) {
          if (txt_idx == 0) {
            depends_on_log_b = true;
          } else if (txt_idx == 1 && has_opt_sep && opt_sep_mask[sep_offset]) {
            depends_on_log_b = true;
          }
        }

        grad_log_b[curr_idx] = depends_on_log_b ? delta : 0.0f;
      }
    }
    __syncthreads();

    float* temporary = next_delta;
    next_delta = curr_delta;
    curr_delta = temporary;

    __syncthreads();
  }
}

// ============================================================
// Autograd backward: log_beta
// ============================================================

__global__ void
log_beta_backward_kernel(const float* __restrict__ grad_log_beta,
                         const float* __restrict__ log_b,
                         const float* __restrict__ log_beta,
                         const bool* __restrict__ mask,
                         const bool* __restrict__ opt_sep_mask,
                         float* __restrict__ grad_log_b,
                         const int batch_size,
                         const int speech_size,
                         const int txt_size,
                         const float neg_large)
{
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  /*
   * Shared-memory layout:
   *
   * curr_delta: [txt_size]
   * next_delta: [txt_size]
   * source_norm: [txt_size]
   *
   * curr_delta[j]
   *     = dL / d log_beta[t, j]
   *
   * next_delta[k]
   *     = dL / d log_beta[t + 1, k]
   */
  extern __shared__ float shared_memory[];

  float* curr_delta = shared_memory;
  float* next_delta = shared_memory + txt_size;

  LocalSoftmaxNorm* source_norm = reinterpret_cast<LocalSoftmaxNorm*>(shared_memory + 2 * txt_size);

  __shared__ int spec_length;
  __shared__ int txt_length;

  const int batch_offset = batch_idx * speech_size * txt_size;
  const int sep_offset = batch_idx * txt_size;
  const bool has_opt_sep = opt_sep_mask != nullptr;

  // --------------------------------------------------------
  // Recover per-sample sequence lengths.
  // --------------------------------------------------------

  if (threadIdx.x == 0) {
    const SequenceLengths lengths = recover_lengths(mask, batch_offset, speech_size, txt_size);

    spec_length = lengths.spec_length;
    txt_length = lengths.txt_length;
  }
  __syncthreads();

  if (spec_length <= 0 || txt_length <= 0) {
    return;
  }

  // --------------------------------------------------------
  // Initialize at beta row t = 0.
  //
  // No earlier beta row depends on beta[0], so its total
  // gradient initially consists only of the upstream gradient.
  //
  //     curr_delta[0, j]
  //         = grad_log_beta[0, j]
  //
  // grad_log_b[0, j] remains zero because beta recurrence
  // never uses log_b at t = 0.
  // --------------------------------------------------------

  const int first_row_offset = batch_offset;
  for (int txt_idx = threadIdx.x; txt_idx < txt_length; txt_idx += blockDim.x) {
    const int index = first_row_offset + txt_idx;
    curr_delta[txt_idx] = mask[index] ? grad_log_beta[index] : 0.0f;
  }
  __syncthreads();

  // --------------------------------------------------------
  // Forward recursion through beta rows.
  //
  // beta[t, j] depends on:
  //
  //     q_stay = log_b[t+1, j]   + beta[t+1, j]
  //     q_adv  = log_b[t+1, j+1] + beta[t+1, j+1]
  //     q_skip = log_b[t+1, j+2] + beta[t+1, j+2]
  //
  // For destination k, gather gradient from sources:
  //
  //     j = k      stay
  //     j = k - 1  advance
  //     j = k - 2  optional skip
  // --------------------------------------------------------

  for (int speech_idx = 0; speech_idx < spec_length - 1; ++speech_idx) {
    const int curr_row_offset = batch_offset + speech_idx * txt_size;
    const int next_row_offset = batch_offset + (speech_idx + 1) * txt_size;

    // ----------------------------------------------------
    // Compute each source state's local LSE normalization.
    // ----------------------------------------------------
    for (int src_txt_idx = threadIdx.x; src_txt_idx < txt_length; src_txt_idx += blockDim.x) {
      const int curr_idx = curr_row_offset + src_txt_idx;

      LocalSoftmaxNorm norm{
        neg_large,
        0.0f,
      };

      if (mask[curr_idx]) {
        // stay: j -> j
        const float stay =
          log_b[next_row_offset + src_txt_idx] + log_beta[next_row_offset + src_txt_idx];

        // advance: j -> j + 1
        const float advance = src_txt_idx + 1 < txt_length
                                ? (log_b[next_row_offset + src_txt_idx + 1] +
                                   log_beta[next_row_offset + src_txt_idx + 1])
                                : neg_large;

        // skip: j -> j + 2
        const bool skip_allowed =
          has_opt_sep && src_txt_idx + 2 < txt_length && opt_sep_mask[sep_offset + src_txt_idx + 1];

        const float skip = skip_allowed ? (log_b[next_row_offset + src_txt_idx + 2] +
                                           log_beta[next_row_offset + src_txt_idx + 2])
                                        : neg_large;

        norm = compute_local_softmax_norm(stay, advance, skip, skip_allowed, neg_large);
      }
      source_norm[src_txt_idx] = norm;
    }
    __syncthreads();

    // ----------------------------------------------------
    // Gather source contributions for each destination k.
    //
    // contribution
    //     = dL / d log_b[t + 1, k]
    //
    // The same contribution reaches beta[t + 1, k]
    // because each candidate is:
    //
    //     q = log_b[t + 1, k] + beta[t + 1, k].
    //
    // Therefore:
    //
    // next_delta[k]
    //     = grad_log_beta[t + 1, k]
    //       + contribution
    // ----------------------------------------------------

    for (int dst_txt_idx = threadIdx.x; dst_txt_idx < txt_length; dst_txt_idx += blockDim.x) {
      const int next_idx = next_row_offset + dst_txt_idx;
      float contribution = 0.0f;

      if (mask[next_idx]) {
        const float destination_value = log_b[next_idx] + log_beta[next_idx];

        // --------------------------------------------
        // Stay:
        //
        // source      = k
        // transition  = k -> k
        // --------------------------------------------
        {
          const float weight =
            local_softmax_weight(destination_value, source_norm[dst_txt_idx], neg_large);
          contribution += curr_delta[dst_txt_idx] * weight;
        }

        // --------------------------------------------
        // Advance:
        //
        // source      = k - 1
        // transition  = k - 1 -> k
        // --------------------------------------------

        if (dst_txt_idx >= 1) {
          const float weight =
            local_softmax_weight(destination_value, source_norm[dst_txt_idx - 1], neg_large);
          contribution += curr_delta[dst_txt_idx - 1] * weight;
        }

        // --------------------------------------------
        // Optional skip:
        //
        // source      = k - 2
        // transition  = k - 2 -> k
        //
        // The skipped token is k - 1.
        // --------------------------------------------

        const bool skip_allowed =
          has_opt_sep && dst_txt_idx >= 2 && opt_sep_mask[sep_offset + dst_txt_idx - 1];

        if (skip_allowed) {
          const float weight =
            local_softmax_weight(destination_value, source_norm[dst_txt_idx - 2], neg_large);
          contribution += curr_delta[dst_txt_idx - 2] * weight;
        }
      }

      grad_log_b[next_idx] = contribution;
      next_delta[dst_txt_idx] = mask[next_idx] ? grad_log_beta[next_idx] + contribution : 0.0f;
    }
    __syncthreads();

    float* temporary = curr_delta;
    curr_delta = next_delta;
    next_delta = temporary;

    __syncthreads();
  }
}

} // namespace

// ============================================================
// CUDA launcher: log_alpha
// ============================================================

torch::Tensor
log_alpha_forward_cuda(torch::Tensor log_b,
                       torch::Tensor mask,
                       c10::optional<torch::Tensor> opt_sep_mask,
                       double neg_large)
{
  // boiler plating..
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard device_guard(log_b.device());

  // data sizes
  const int batch_size = static_cast<int>(log_b.size(0));
  const int speech_size = static_cast<int>(log_b.size(1));
  const int txt_size = static_cast<int>(log_b.size(2));

  // buffers
  const size_t shared_memory_bytes = 2 * static_cast<size_t>(txt_size) * sizeof(float);
  const bool* opt_sep_ptr = nullptr;

  // init result Tensor
  torch::Tensor log_alpha = torch::full_like(log_b, static_cast<float>(neg_large));

  if (opt_sep_mask.has_value()) {
    opt_sep_ptr = opt_sep_mask.value().data_ptr<bool>();
  }

  log_alpha_forward_kernel<<< // not forward in forward-backward algorithm!
    batch_size,
    MAX_THREADS,
    shared_memory_bytes,
    stream>>>(log_b.data_ptr<float>(),
              mask.data_ptr<bool>(),
              opt_sep_ptr,
              log_alpha.data_ptr<float>(),
              batch_size,
              speech_size,
              txt_size,
              static_cast<float>(neg_large));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return log_alpha;
}

// ============================================================
// CUDA launcher: log_beta
// ============================================================

torch::Tensor
log_beta_forward_cuda(torch::Tensor log_b,
                      torch::Tensor mask,
                      c10::optional<torch::Tensor> opt_sep_mask,
                      double neg_large)
{
  // boiler plating..
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard device_guard(log_b.device());

  // data sizes
  const int batch_size = static_cast<int>(log_b.size(0));
  const int speech_size = static_cast<int>(log_b.size(1));
  const int txt_size = static_cast<int>(log_b.size(2));

  // buffers
  const size_t shared_memory_bytes = 2 * static_cast<size_t>(txt_size) * sizeof(float);
  const bool* opt_sep_ptr = nullptr;

  // init result Tensor
  torch::Tensor log_beta = torch::full_like(log_b, static_cast<float>(neg_large));

  if (opt_sep_mask.has_value()) {
    opt_sep_ptr = opt_sep_mask.value().data_ptr<bool>();
  }

  log_beta_forward_kernel<<< // not forward in forward-backward algorithm!
    batch_size,
    MAX_THREADS,
    shared_memory_bytes,
    stream>>>(log_b.data_ptr<float>(),
              mask.data_ptr<bool>(),
              opt_sep_ptr,
              log_beta.data_ptr<float>(),
              batch_size,
              speech_size,
              txt_size,
              static_cast<float>(neg_large));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return log_beta;
}

// ============================================================
// CUDA launcher: log_alpha backward
// ============================================================

torch::Tensor
log_alpha_backward_cuda(torch::Tensor grad_log_alpha,
                        torch::Tensor log_b,
                        torch::Tensor log_alpha,
                        torch::Tensor mask,
                        c10::optional<torch::Tensor> opt_sep_mask,
                        double neg_large)
{
  // Select the device before retrieving its current stream.
  const c10::cuda::CUDAGuard device_guard(log_b.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  grad_log_alpha = grad_log_alpha.contiguous();

  const int batch_size = static_cast<int>(log_b.size(0));
  const int speech_size = static_cast<int>(log_b.size(1));
  const int txt_size = static_cast<int>(log_b.size(2));

  const bool* opt_sep_ptr = nullptr;
  const size_t shared_memory_bytes = 2 * static_cast<size_t>(txt_size) * sizeof(float) +
                                     static_cast<size_t>(txt_size) * sizeof(LocalSoftmaxNorm);

  torch::Tensor grad_log_b = torch::zeros_like(log_b);

  if (opt_sep_mask.has_value()) {
    opt_sep_ptr = opt_sep_mask.value().data_ptr<bool>();
  }

  log_alpha_backward_kernel<<<batch_size, MAX_THREADS, shared_memory_bytes, stream>>>(
    grad_log_alpha.data_ptr<float>(),
    log_b.data_ptr<float>(),
    log_alpha.data_ptr<float>(),
    mask.data_ptr<bool>(),
    opt_sep_ptr,
    grad_log_b.data_ptr<float>(),
    batch_size,
    speech_size,
    txt_size,
    static_cast<float>(neg_large));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_log_b;
}

// ============================================================
// CUDA launcher: log_beta backward
// ============================================================

torch::Tensor
log_beta_backward_cuda(torch::Tensor grad_log_beta,
                       torch::Tensor log_b,
                       torch::Tensor log_beta,
                       torch::Tensor mask,
                       c10::optional<torch::Tensor> opt_sep_mask,
                       double neg_large)
{
  const c10::cuda::CUDAGuard device_guard(log_b.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  grad_log_beta = grad_log_beta.contiguous();

  const int batch_size = static_cast<int>(log_b.size(0));
  const int speech_size = static_cast<int>(log_b.size(1));
  const int txt_size = static_cast<int>(log_b.size(2));

  const bool* opt_sep_ptr = nullptr;
  const size_t shared_memory_bytes = 2 * static_cast<size_t>(txt_size) * sizeof(float) +
                                     static_cast<size_t>(txt_size) * sizeof(LocalSoftmaxNorm);

  torch::Tensor grad_log_b = torch::zeros_like(log_b);
  if (opt_sep_mask.has_value()) {
    opt_sep_ptr = opt_sep_mask.value().data_ptr<bool>();
  }

  log_beta_backward_kernel<<<batch_size, MAX_THREADS, shared_memory_bytes, stream>>>(
    grad_log_beta.data_ptr<float>(),
    log_b.data_ptr<float>(),
    log_beta.data_ptr<float>(),
    mask.data_ptr<bool>(),
    opt_sep_ptr,
    grad_log_b.data_ptr<float>(),
    batch_size,
    speech_size,
    txt_size,
    static_cast<float>(neg_large));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_log_b;
}
