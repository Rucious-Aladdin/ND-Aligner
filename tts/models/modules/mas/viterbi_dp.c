
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define VITERBI_EXPORT __declspec(dllexport)
#else
#define VITERBI_EXPORT __attribute__((visibility("default")))
#endif

/*
 * Batched Viterbi forward recursion and backtracking.
 *
 * This function corresponds to the Python section beginning with:
 *
 *     for t in range(1, T_speech):
 *
 * and ending after path backtracking.
 *
 * The initial DP row, delta_t at t=0, must already be computed in Python
 * and passed through initial_delta.
 *
 * Array layouts (all C-contiguous, row-major):
 *
 *     log_b                     float32  [B, T_speech, T_text]
 *     dp_valid                  uint8    [B, T_speech, T_text]
 *     initial_delta             float32  [B, T_text]
 *     spec_lengths              int64    [B]
 *     text_lengths              int64    [B]
 *     optional_separator_mask   uint8    [B, T_text], or NULL
 *     path                      int64    [B, T_speech]
 *     viterbi_logp              float32  [B]
 *
 * Transition topology for current state j:
 *
 *     stay:     previous state j
 *     advance:  previous state j - 1
 *     skip:     previous state j - 2
 *
 * Skip is allowed only when optional_separator_mask[b, j - 1] is true.
 *
 * Tie-breaking exactly follows torch.argmax over candidates ordered as:
 *
 *     [stay, advance, skip]
 *
 * because candidates replace the current best only on strict `>`.
 *
 * Return values:
 *
 *      0: success
 *     -1: null required pointer
 *     -2: invalid dimensions
 *     -3: invalid sequence length
 *     -4: memory allocation failure
 */
VITERBI_EXPORT int viterbi_forward_backtrack_f32(
    const float *log_b,
    const uint8_t *dp_valid,
    const float *initial_delta,
    const int64_t *spec_lengths,
    const int64_t *text_lengths,
    const uint8_t *optional_separator_mask,
    int64_t B,
    int64_t T_speech,
    int64_t T_text,
    float neg_large,
    int64_t *path,
    float *viterbi_logp
) {
    const size_t text_size = (size_t)T_text;
    const size_t speech_text_size = (size_t)T_speech * text_size;
    const size_t path_size = (size_t)B * (size_t)T_speech;

    float *prev = (float *)malloc(text_size * sizeof(float));
    float *curr = (float *)malloc(text_size * sizeof(float));

    /*
     * Only the selected move is needed for backtracking:
     *   0 = stay, 1 = advance, 2 = skip.
     *
     * A uint8_t move table uses one byte per DP cell instead of storing
     * an int64 previous-state index.
     */
    uint8_t *moves = (uint8_t *)malloc(speech_text_size * sizeof(uint8_t));

    if (prev == NULL || curr == NULL || moves == NULL) {
        free(prev);
        free(curr);
        free(moves);
        return -4;
    }

    for (size_t i = 0; i < path_size; ++i) {
        path[i] = -1;
    }

    for (int64_t b = 0; b < B; ++b) {
        const int64_t speech_len = spec_lengths[b];
        const int64_t text_len = text_lengths[b];
        const size_t batch_dp_offset = (size_t)b * speech_text_size;
        const size_t batch_text_offset = (size_t)b * text_size;
        const size_t batch_path_offset = (size_t)b * (size_t)T_speech;

        memcpy(
            prev,
            initial_delta + batch_text_offset,
            text_size * sizeof(float)
        );

        /*
         * Forward recursion.
         *
         * t is sequential because delta[t] depends on delta[t - 1].
         * b and j are explicit C loops replacing PyTorch batch/state
         * parallel tensor operations.
         */
        for (int64_t t = 1; t < speech_len; ++t) {
            const size_t time_offset =
                batch_dp_offset + (size_t)t * text_size;

            for (int64_t j = 0; j < T_text; ++j) {
                float best_score = prev[j];
                uint8_t best_move = 0;  /* stay */

                if (j >= 1) {
                    const float advance_score = prev[j - 1];

                    // torch tie breaking
                    if (advance_score > best_score) {
                        best_score = advance_score;
                        best_move = 1;
                    }
                }

                if (
                    optional_separator_mask != NULL &&
                    j >= 2 &&
                    optional_separator_mask[
                        batch_text_offset + (size_t)(j - 1)
                    ] != 0
                ) {
                    const float skip_score = prev[j - 2];

                    // torch tie breaking
                    if (skip_score > best_score) {
                        best_score = skip_score;
                        best_move = 2;
                    }
                }

                const size_t index = time_offset + (size_t)j;

                if (dp_valid[index] != 0) {
                    curr[j] = log_b[index] + best_score;
                } else {
                    curr[j] = neg_large;
                }

                moves[(size_t)t * text_size + (size_t)j] = best_move;
            }

            float *tmp = prev;
            prev = curr;
            curr = tmp;
        }

        viterbi_logp[b] = prev[text_len - 1];

        /*
         * Backtracking.
         *
         * Padded frames remain -1 because the full path output was
         * initialized before processing the batch.
         */
        int64_t j_cur = text_len - 1;

        for (int64_t t = speech_len - 1; t >= 0; --t) {
            path[batch_path_offset + (size_t)t] = j_cur;

            if (t > 0) {
                const uint8_t move =
                    moves[(size_t)t * text_size + (size_t)j_cur];

                j_cur -= (int64_t)move;
            }
        }
    }

    free(prev);
    free(curr);
    free(moves);

    return 0;
}
