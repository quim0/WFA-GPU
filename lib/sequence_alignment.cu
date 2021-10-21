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
#include "wfa_types.h"
#include "utils/cuda_utils.cuh"
#include "utils/logger.h"
#include "sequence_alignment.cuh"

void allocate_offloaded_bt_d (wfa_backtrace_t** bt_offloaded_d,
                              const int max_steps,
                              const size_t num_alignments) {
    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps);

    if (bt_offloaded_size >= (1ULL<<(wfa_backtrace_bits-1))) {
        LOG_ERROR("Trying to allocate more backtrace elements than the ones"
                  " that we can address")
        exit(-1);
    }

    bt_offloaded_size *= num_alignments;

    // Add the results array
    size_t bt_offloaded_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps)
                                       * num_alignments;

    LOG_DEBUG("Allocating %f MiB to store backtraces of %zu alignments.",
              (float)((bt_offloaded_size + bt_offloaded_results_size) * sizeof(wfa_backtrace_t)) / (1 << 20),
              num_alignments)

    cudaMalloc(bt_offloaded_d,
               (bt_offloaded_size + bt_offloaded_results_size) * sizeof(wfa_backtrace_t));
    CUDA_CHECK_ERR
}

void reset_offloaded_bt_d (wfa_backtrace_t* bt_offloaded_d,
                              const int max_steps,
                              const size_t num_alignments,
                              cudaStream_t stream) {
    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps);

    bt_offloaded_size *= num_alignments;

    // Add the results array
    size_t bt_offloaded_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps)
                                       * num_alignments;

    cudaMemsetAsync(
        bt_offloaded_d,
        0,
        (bt_offloaded_size + bt_offloaded_results_size) * sizeof(wfa_backtrace_t),
        stream
    );
    CUDA_CHECK_ERR
}

void allocate_wf_data_buffer_d (uint8_t** wf_data_buffer,
                                const size_t max_steps,
                                const affine_penalties_t penalties,
                                const size_t num_alignments) {

    // Create the active working set buffer on global memory
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    size_t cells_elements = active_working_set * max_wf_size;
    cells_elements = cells_elements + (4 - (cells_elements % 4));

    size_t wf_data_buffer_size = cells_elements * 3 * sizeof(wfa_cell_t);
    wf_data_buffer_size *= num_alignments;

    LOG_DEBUG("Allocating %f MiB to store working set data of %zu alignments.",
              (float)(wf_data_buffer_size) / (1 << 20), num_alignments)

    cudaMalloc(wf_data_buffer, wf_data_buffer_size);
    CUDA_CHECK_ERR;
}

void reset_wf_data_buffer_d (uint8_t* wf_data_buffer,
                             const size_t max_steps,
                             const affine_penalties_t penalties,
                             const size_t num_alignments,
                             cudaStream_t stream) {

    // Create the active working set buffer on global memory
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;

    size_t cells_elements = active_working_set * max_wf_size;
    cells_elements = cells_elements+ (4 - (cells_elements % 4));

    size_t wf_data_buffer_size = cells_elements * 3 * sizeof(wfa_cell_t);
    wf_data_buffer_size *= num_alignments;

    cudaMemsetAsync(wf_data_buffer, 0, wf_data_buffer_size, stream);
    CUDA_CHECK_ERR;
}

void launch_alignments_async (const char* packed_sequences_buffer,
                              const sequence_pair_t* sequences_metadata,
                              const size_t num_alignments,
                              const affine_penalties_t penalties,
                              alignment_result_t* const results,
                              wfa_backtrace_t* const backtraces,
                              alignment_result_t *results_d,
                              wfa_backtrace_t* bt_offloaded_d,
                              uint8_t* const wf_data_buffer,
                              const int max_steps,
                              const int threads_per_block,
                              cudaStream_t stream) {

    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps) * num_alignments;
    size_t bt_offloaded_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps)
                                       * num_alignments;

    wfa_backtrace_t* bt_offloaded_results_d = bt_offloaded_d
                                              + bt_offloaded_size;

    // TODO: Reduction of penalties
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;

    size_t sh_mem_size = \
                    // Wavefronts structs space
                    + (active_working_set * sizeof(wfa_wavefront_t) * 3)
                    // Position of the last used element in the offloaded
                    // backtraces. It will be atomically increased.
                    + sizeof(int);

    // TODO
    dim3 gridSize(num_alignments);
    dim3 blockSize(threads_per_block);

    LOG_DEBUG("Launching %d blocks of %d threads with %.2fKiB of shared memory",
              gridSize.x, blockSize.x, (float(sh_mem_size) / (2 << 10)));

    LOG_DEBUG("Working with penalties: X=%d, O=%d, E=%d", penalties.x,
              penalties.o, penalties.e);

    alignment_kernel<<<gridSize, blockSize, sh_mem_size, stream>>>(
                                              packed_sequences_buffer,
                                              sequences_metadata,
                                              num_alignments,
                                              max_steps,
                                              wf_data_buffer,
                                              penalties,
                                              bt_offloaded_d,
                                              bt_offloaded_results_d,
                                              results_d);
    CUDA_CHECK_ERR

    // TODO: Unify results and backtraces memory buffers to do a signle memcpy
    cudaMemcpyAsync(results, results_d, num_alignments * sizeof(alignment_result_t),
               cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERR
    cudaMemcpyAsync(backtraces, bt_offloaded_results_d,
               bt_offloaded_results_size * sizeof(wfa_backtrace_t),
               cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERR

    // TODO: CUDAFREE
}
