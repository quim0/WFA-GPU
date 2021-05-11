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

#include "kernels/sequence_alignment_kernel.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logger.h"
#include "sequence_alignment.cuh"

void launch_alignments_async (const char* packed_sequences_buffer,
                              const sequence_pair_t* sequences_metadata,
                              const size_t num_alignments,
                              const affine_penalties_t penalties,
                              alignment_result_t* results) {
    // TODO: Free results_d
    alignment_result_t *results_d;
    cudaMalloc(&results_d, num_alignments * sizeof(alignment_result_t));
    CUDA_CHECK_ERR

    // TODO: Reduction of penalties
    const int max_steps = 16;
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    size_t sh_mem_size = \
                    // Offsets space
                    (active_working_set * max_wf_size * sizeof(wfa_offset_t) * 3)
                    // Wavefronts structs space
                    + (active_working_set * sizeof(wfa_wavefront_t) * 3)
                    // Wavefronts pointers arrays space
                    + (active_working_set * sizeof(wfa_wavefront_t*) * 3);

    // TODO
    dim3 gridSize(num_alignments);
    dim3 blockSize(64);

    // TODO !!!!!!!!
    sh_mem_size *= 10;

    LOG_DEBUG("Launching %d blocks of %d threads with %.2fKiB of shared memory",
              gridSize.x, blockSize.x, (float(sh_mem_size) / (2 << 10)));

    LOG_DEBUG("Working with penalties: X=%d, O=%d, E=%d", penalties.x,
              penalties.o, penalties.e);

    alignment_kernel<<<gridSize, blockSize, sh_mem_size>>>(
                                              packed_sequences_buffer,
                                              sequences_metadata,
                                              num_alignments,
                                              max_steps,
                                              penalties,
                                              results_d);

    // TODO: Make this async, another function to copy the results?
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR

    cudaMemcpy(results, results_d, num_alignments * sizeof(alignment_result_t),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR
}
