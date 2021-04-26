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

#include "sequence_packing.cuh"
#include "utils/logger.h"
#include "utils/cuda_utils.cuh"
#include "kernels/sequence_packing_kernel.cuh"

void prepare_pack_sequences_gpu_async (const char* sequences_buffer,
                         const size_t sequences_buffer_size,
                         sequence_pair_t* sequences_metadata,
                         const size_t num_alignments,
                         char** device_sequences_buffer_unpacked,
                         char** device_sequences_buffer_packed,
                         size_t* device_sequences_buffer_packed_size,
                         sequence_pair_t** device_sequences_metadata,
                         cudaStream_t memcpy_stream) {
    // +1 for the final nullbyte
    size_t mem_needed_unpacked =
                            sequences_metadata[num_alignments-1].text_offset
                            + sequences_metadata[num_alignments-1].text_len + 1;

    if (mem_needed_unpacked == 0) {
        LOG_ERROR("No alignments to do or invalid sequence metadata.")
    }

    LOG_DEBUG("Allocating %.2f MiB to store the unpacked sequences on the device.",
              float(mem_needed_unpacked) / (1<<20));

    cudaMalloc(device_sequences_buffer_unpacked, mem_needed_unpacked);
    CUDA_CHECK_ERR

    cudaMemcpyAsync(*device_sequences_buffer_unpacked,
                    sequences_buffer,
                    mem_needed_unpacked,
                    cudaMemcpyHostToDevice,
                    memcpy_stream);
    CUDA_CHECK_ERR

    size_t mem_needed_packed = 0;

    // Calculate the amount of memory that will be needed to store the packed
    // sequences on the GPU. Each packed sequence has to be 32 bits aligned, so
    // the corresponding padding is added.
    // Update the metadata so the offsets matches the packed sequences.
    for (int i=0; i<num_alignments; i++) {
        // Pattern
        sequence_pair_t* curr_alignment = &sequences_metadata[i];
        size_t pattern_length_packed = curr_alignment->pattern_len/4;
        curr_alignment->pattern_offset_packed = mem_needed_packed;
        mem_needed_packed += (pattern_length_packed
                        + (4 - (pattern_length_packed % 4)));
        // Text
        size_t text_length_packed = curr_alignment->text_len/4;
        curr_alignment->text_offset_packed = mem_needed_packed;
        mem_needed_packed += (text_length_packed
                        + (4 - (text_length_packed % 4)));
    }

    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences on the device.",
              (float(mem_needed_packed) / (1<<20)));

    cudaMalloc(device_sequences_buffer_packed, mem_needed_packed);
    CUDA_CHECK_ERR
    *device_sequences_buffer_packed_size = mem_needed_packed;

    size_t mem_needed_metadata = num_alignments * sizeof(sequence_pair_t);
    LOG_DEBUG("Allocating %.2f KiB to store the packed sequences metadata on "
              "the device.", float(mem_needed_metadata) / (1<<10));
    cudaMalloc(device_sequences_metadata, mem_needed_metadata);
    CUDA_CHECK_ERR

    cudaMemcpyAsync(*device_sequences_metadata,
                    sequences_metadata,
                    mem_needed_metadata,
                    cudaMemcpyHostToDevice,
                    memcpy_stream);
}

void pack_sequences_gpu_async (const char* const d_sequences_buffer_unpacked,
                               char* const d_sequences_buffer_packed,
                               size_t sequences_buffer_size_unpacked,
                               size_t sequences_buffer_size_packed,
                               const sequence_pair_t* d_sequences_metadata,
                               size_t num_alignments,
                               cudaStream_t stream) {
    // Allocate 4KiB of shared memory to store unpacked sequence
    // Shared memory size must be a multiple of 32 bit (4 bytes)
    // TODO: Fine tune this number
    size_t sh_mem = 4 << 10;

    dim3 gridSize(num_alignments * 2);
    // TODO: Make this editable as a runtime parameter or in compile time
    // Number of threads working per sequence
    dim3 blockSize(512);

    LOG_DEBUG("Launching packing kernel with %d threads per block. %d blocks."
              " Using %.2fKiB of shared memory.",
               blockSize.x, gridSize.x, float(sh_mem) / (1 << 10))

    compact_sequences<<<gridSize, blockSize, sh_mem, stream>>>(
                                            d_sequences_buffer_unpacked,
                                            d_sequences_buffer_packed,
                                            d_sequences_metadata,
                                            sh_mem);
    CUDA_CHECK_ERR
}
