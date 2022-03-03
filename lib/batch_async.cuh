/*
 * Copyright (c) 2021 Quim Aguado
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef BATCH_ASYNC_CUH
#define BATCH_ASYNC_CUH

#include "utils/sequences.h"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "wfa_types.h"

#if __cplusplus
extern "C" {
#endif

void launch_alignments_batched (const char* sequences_buffer,
                        const size_t sequences_buffer_size,
                        sequence_pair_t* const sequences_metadata,
                        const size_t num_alignments,
                        const affine_penalties_t penalties,
                        alignment_result_t* results,
                        wfa_backtrace_t* backtraces,
                        const int max_distance,
                        const int threads_per_block,
                        const int num_blocks,
                        size_t batch_size,
                        const int band,
                        bool check_correctness);

#if __cplusplus // end of extern "C"
}
#endif

#endif
