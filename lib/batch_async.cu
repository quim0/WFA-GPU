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
#include "utils/cuda_utils.cuh"
#include "utils/logger.h"

size_t bytes_to_copy_unpacked (int from,
                               int to,
                               sequence_pair_t* sequences_metadata) {
    // +1 for the final byte
    size_t final_byte = sequences_metadata[to].text_offset +
                                 sequences_metadata[to].text_len + 1;
    size_t initial_byte = sequences_metadata[from].pattern_offset;
    return final_byte - initial_byte;
}

void launch_alignments_batched (const char* sequences_buffer,
                        const size_t sequences_buffer_size,
                        sequence_pair_t* const sequences_metadata,
                        const size_t num_alignments,
                        const affine_penalties_t penalties,
                        alignment_result_t* results,
                        wfa_backtrace_t* backtraces,
                        const int max_distance,
                        const int threads_per_block,
                        size_t batch_size) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    if (batch_size == 0) batch_size = num_alignments;

    int num_batchs = ceil(num_alignments / batch_size);

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    CUDA_CHECK_ERR
    cudaStreamCreate(&stream2);
    CUDA_CHECK_ERR

    // TODO: Now its assumed that all the sequences will have more or less the
    // same sequence length, this may not be the case, but with this assumption
    // is possible to have only one buffer for the sequences
    char *d_seq_buffer_unpacked;
    size_t mem_needed_unpacked = sequences_metadata[batch_size-1].text_offset +
                                 sequences_metadata[batch_size-1].text_offset + 1;
    // Make the buffer 20% bigger to have some extra room in case next batch
    // have slightly bigger sequences
    mem_needed_unpacked *= 1.2;

    LOG_DEBUG("Allocating %.2f MiB to store the unpacked sequences on the device.",
              (float(mem_needed_unpacked) / (1<<20)));

    cudaMalloc(&d_seq_buffer_unpacked, mem_needed_unpacked);
    CUDA_CHECK_ERR

    char *d_seq_buffer_packed;
    size_t mem_needed_packed = 0;

    // Calculate the amount of memory that will be needed to store the packed
    // sequences on the GPU. Each packed sequence has to be 32 bits aligned, so
    // the corresponding padding is added.
    // Update the metadata so the offsets matches the packed sequences.
    for (int i=0; i<batch_size; i++) {
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

    // Make buffer 20% bigger
    mem_needed_packed *= 1.2;
    mem_needed_packed = mem_needed_packed - (4 - (mem_needed_packed % 4));

    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences on the device.",
              (float(mem_needed_packed) / (1<<20)));

    cudaMalloc(&d_seq_buffer_packed, mem_needed_packed);
    CUDA_CHECK_ERR

    sequence_pair_t *d_seq_metadata;
    size_t mem_needed_metadata = batch_size * sizeof(sequence_pair_t);
    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences metadata on "
              "the device.", float(mem_needed_metadata) / (1<<20));
    cudaMalloc(&d_seq_metadata, mem_needed_metadata);
    CUDA_CHECK_ERR

    LOG_DEBUG("Aligning %d alignments using %d batchs of %d elements.",
              num_alignments, num_batchs, batch_size);

    // Copy unpacked sequences of current batch
    size_t bytes_to_copy_seqs = bytes_to_copy_unpacked(0, batch_size-1,
                                                       sequences_metadata);
    const char *initial_seq = &sequences_buffer[sequences_metadata[0].pattern_offset];
    cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                    cudaMemcpyHostToDevice, stream1);

    // Results of the alignments
    alignment_result_t *results_d;
    cudaMalloc(&results_d, batch_size * sizeof(alignment_result_t));
    CUDA_CHECK_ERR

    // Space for the kernel to store the offloaded backtraces on the GPU, this
    // is allocated just once here and reused.
    wfa_backtrace_t* bt_offloaded_d;
    // TODO: max_distance = max_steps (?)
    allocate_offloaded_bt_d(&bt_offloaded_d, max_distance, batch_size);

    for (int batch=0; batch < num_batchs; batch++) {
        const int from = batch * batch_size;
        const int to = (batch == (num_batchs-1))
                       ? num_alignments : (((batch+1) * batch_size) - 1);
        const int curr_batch_size = to - from;


        // Copy metadata of current batch
        cudaMemcpyAsync(d_seq_metadata, &sequences_metadata[from],
                        curr_batch_size * sizeof(sequence_pair_t),
                        cudaMemcpyHostToDevice, stream1);

        // Launch packing kernel
        pack_sequences_gpu_async(d_seq_buffer_unpacked,
                                 d_seq_buffer_packed,
                                 d_seq_metadata,
                                 curr_batch_size,
                                 stream1);

        // Make sure packing kernel have finished before sending the next
        // unpacked sequences to the device
        cudaStreamSynchronize(stream1);

        // Copy unpacked sequences of next batch while the alignment kernel is
        // running.
        const int next_from = (batch+1) * batch_size;
        const int next_to = ((batch+1) == (num_batchs-1))
                       ? num_alignments : (((batch+2) * batch_size) - 1);
        bytes_to_copy_seqs = bytes_to_copy_unpacked(next_from, next_to,
                                                           sequences_metadata);
        initial_seq = &sequences_buffer[sequences_metadata[next_from].pattern_offset];
        cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                        cudaMemcpyHostToDevice, stream1);

        // TODO: max_distance = max_steps (?)
        // Align current batch
        launch_alignments_async(
            d_seq_buffer_packed,
            d_seq_metadata,
            curr_batch_size,
            penalties,
            results,
            backtraces,
            results_d,
            bt_offloaded_d,
            max_distance,
            threads_per_block,
            stream2
        );

        // Make sure that alignment kernel has finished before packing the next
        // batch sequences
        cudaStreamSynchronize(stream2);
    }

    cudaStreamDestroy(stream1);
    CUDA_CHECK_ERR
    cudaStreamDestroy(stream2);
    CUDA_CHECK_ERR

    cudaFree(d_seq_buffer_unpacked);
    cudaFree(d_seq_buffer_packed);
    cudaFree(d_seq_metadata);
    cudaFree(results_d);
    cudaFree(bt_offloaded_d);

}

extern "C" void launch_alignments (const char* sequences_buffer,
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

    prepare_pack_sequences_gpu(
        sequences_buffer,
        sequences_buffer_size,
        sequences_metadata,
        num_alignments,
        &d_seq_buffer_unpacked,
        &d_seq_buffer_packed,
        &seq_buffer_packed_size,
        &d_sequences_metadata
    );

    pack_sequences_gpu_async(
        d_seq_buffer_unpacked,
        d_seq_buffer_packed,
        d_sequences_metadata,
        num_alignments,
        0
    );

    // Results of the alignments
    // TODO: CudaMallocAsync (?)
    alignment_result_t *results_d;
    cudaMalloc(&results_d, num_alignments * sizeof(alignment_result_t));
    CUDA_CHECK_ERR

    wfa_backtrace_t* bt_offloaded_d;
    // TODO: max_distance == max_steps (?)
    allocate_offloaded_bt_d(&bt_offloaded_d, max_distance, num_alignments);

    // TODO: max_distance == max_steps (?)
    launch_alignments_async(
        d_seq_buffer_packed,
        d_sequences_metadata,
        num_alignments,
        penalties,
        results,
        backtraces,
        results_d,
        bt_offloaded_d,
        max_distance,
        threads_per_block,
        0
    );

    cudaFree(results_d);
    cudaFree(bt_offloaded_d);
}
