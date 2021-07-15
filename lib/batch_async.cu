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

#include "batch_async.cuh"
#include "sequence_packing.cuh"
#include "sequence_alignment.cuh"
// TODO: launch_batched_alignments (num_aligns, num_batches, num_steams... etc)

extern "C" void launch_batch_async (const char* sequences_buffer,
                         const size_t sequences_buffer_size,
                         sequence_pair_t* sequences_metadata,
                         const size_t num_alignments,
                         const affine_penalties_t penalties,
                         alignment_result_t* results,
                         wfa_backtrace_t* backtraces,
                         const int max_distance,
                         const int threads_per_block) {
    // TODO: Make this stream reusable instead of creating a new one per batch
    //cudaStream_t stream;
    //cudaStreamCreate(&stream);
    // Sequence packing
    char* d_seq_buffer_unpacked;
    char* d_seq_buffer_packed;
    sequence_pair_t* d_sequences_metadata;
    size_t seq_buffer_packed_size;

    prepare_pack_sequences_gpu_async(
        sequences_buffer,
        sequences_buffer_size,
        sequences_metadata,
        num_alignments,
        &d_seq_buffer_unpacked,
        &d_seq_buffer_packed,
        &seq_buffer_packed_size,
        &d_sequences_metadata,
        0
    );

    pack_sequences_gpu_async(
        d_seq_buffer_unpacked,
        d_seq_buffer_packed,
        sequences_buffer_size,
        seq_buffer_packed_size,
        d_sequences_metadata,
        num_alignments,
        0
    );

    // Alignment
    // TODO: This is not async for now

    launch_alignments_async(
        d_seq_buffer_packed,
        d_sequences_metadata,
        num_alignments,
        penalties,
        results,
        backtraces,
        max_distance,
        threads_per_block
    );

    // TODO
    cudaDeviceSynchronize();

    // Backtrace & CIGAR recovery
}
