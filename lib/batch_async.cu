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
#include "utils/wf_clock.h"
#include "utils/verification.cuh"
#include "utils/wfa_cpu.h"

size_t bytes_to_copy_unpacked (const int from,
                               const int to,
                               const sequence_pair_t* sequences_metadata) {
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
                        const int num_blocks,
                        size_t batch_size,
                        bool check_correctness) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    if (batch_size == 0) batch_size = num_alignments;

    int num_batchs = ceil((double)num_alignments / batch_size);

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    CUDA_CHECK_ERR
    cudaStreamCreate(&stream2);
    CUDA_CHECK_ERR

    // TODO: Now its assumed that all the sequences will have more or less the
    // same sequence length, this may not be the case, but with this assumption
    // is possible to have only one buffer for the sequences
    char *d_seq_buffer_unpacked;
    char *h_seq_buffer_unpacked;
    size_t mem_needed_unpacked = sequences_metadata[batch_size-1].text_offset +
                                 sequences_metadata[batch_size-1].text_offset + 1;
    // Make the buffer 20% bigger to have some extra room in case next batch
    // have slightly bigger sequences
    mem_needed_unpacked *= 1.2;

    LOG_DEBUG("Allocating %.2f MiB to store the unpacked sequences on the device.",
              (float(mem_needed_unpacked) / (1<<20)));

    cudaMalloc(&d_seq_buffer_unpacked, mem_needed_unpacked);
    CUDA_CHECK_ERR
    // Allocated pinned memory on host
    cudaMallocHost(&h_seq_buffer_unpacked, mem_needed_unpacked);
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
    sequence_pair_t *h_seq_metadata;
    size_t mem_needed_metadata = batch_size * sizeof(sequence_pair_t);
    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences metadata on "
              "the device.", float(mem_needed_metadata) / (1<<20));
    cudaMalloc(&d_seq_metadata, mem_needed_metadata);
    CUDA_CHECK_ERR
    cudaMallocHost(&h_seq_metadata, mem_needed_metadata);
    CUDA_CHECK_ERR

    LOG_DEBUG("Aligning %zu alignments using %d batchs of %zu elements.",
              num_alignments, num_batchs, batch_size);

    // Copy unpacked sequences of current batch
    size_t bytes_to_copy_seqs = bytes_to_copy_unpacked(0, batch_size - 1,
                                                       sequences_metadata);
    const char *initial_seq = &sequences_buffer[sequences_metadata[0].pattern_offset];
    memcpy(h_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs);
    cudaMemcpyAsync(d_seq_buffer_unpacked, h_seq_buffer_unpacked, bytes_to_copy_seqs,
                    cudaMemcpyHostToDevice, stream1);
    CUDA_CHECK_ERR

    // Results of the alignments
    alignment_result_t *results_d;
    alignment_result_t *results_h;
    cudaMalloc(&results_d, batch_size * sizeof(alignment_result_t));
    CUDA_CHECK_ERR
    cudaMallocHost(&results_h, batch_size * sizeof(alignment_result_t));
    CUDA_CHECK_ERR

    // Space for the kernel to store the offloaded backtraces on the GPU, this
    // is allocated just once here and reused.
    wfa_backtrace_t* bt_offloaded_d;
    // TODO: max_distance = max_steps (?)
    allocate_offloaded_bt_d(&bt_offloaded_d, max_distance, num_blocks, num_alignments);

    uint8_t* wf_data_buffer;
    allocate_wf_data_buffer_d(&wf_data_buffer, max_distance,
                              penalties, num_blocks);

    uint32_t* next_alignment_idx_d = (uint32_t*)(wf_data_buffer +
                                     wf_data_buffer_size(
                                         penalties,
                                         max_distance)
                                     );

    // Pinned memory region on host to store the backtrace chain result for each
    // alignment in the batch
    uint32_t bt_result_offloaded_size = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
    wfa_backtrace_t* bt_results_offloaded_h;
    cudaMallocHost(&bt_results_offloaded_h,
                   batch_size * bt_result_offloaded_size * sizeof(wfa_backtrace_t));

    for (int batch=0; batch < num_batchs; batch++) {
        const int from = batch * batch_size;
        const int to = (batch == (num_batchs-1))
                       ? num_alignments-1 : (((batch+1) * batch_size) - 1);
        const int curr_batch_size = to - from + 1;

        // Copy metadata of current batch
        memcpy(h_seq_metadata,
               &sequences_metadata[from],
               curr_batch_size * sizeof(sequence_pair_t));
        cudaMemcpyAsync(d_seq_metadata, h_seq_metadata,
                        curr_batch_size * sizeof(sequence_pair_t),
                        cudaMemcpyHostToDevice, stream1);
        CUDA_CHECK_ERR

        // Reset the memory regions for the alignment kernel
        reset_offloaded_bt_d(bt_offloaded_d, max_distance, num_blocks, batch_size, stream2);
        reset_wf_data_buffer_d(wf_data_buffer, max_distance,
                                  penalties, num_blocks, stream2);

        // Launch packing kernel
        pack_sequences_gpu_async(d_seq_buffer_unpacked,
                                 d_seq_buffer_packed,
                                 d_seq_metadata,
                                 curr_batch_size,
                                 stream1);

        // Make sure packing kernel have finished before sending the next
        // unpacked sequences to the device
        cudaStreamSynchronize(stream1);

        // TODO: max_distance = max_steps (?)
        // Align current batch
        launch_alignments_async(
            d_seq_buffer_packed,
            d_seq_metadata,
            curr_batch_size,
            penalties,
            results_h,
            bt_results_offloaded_h,
            results_d,
            bt_offloaded_d,
            wf_data_buffer,
            max_distance,
            threads_per_block,
            num_blocks,
            stream2
        );


        // Copy unpacked sequences of next batch while the alignment kernel is
        // running. Don't do it in the final batch as there is no next batch in
        // that case.
        if (batch < (num_batchs-1)) {
            const int next_from = (batch+1) * batch_size;
            const int next_to = ((batch+1) == (num_batchs-1))
                           ? num_alignments-1 : (((batch+2) * batch_size) - 1);
            bytes_to_copy_seqs = bytes_to_copy_unpacked(next_from, next_to,
                                                        sequences_metadata);
            initial_seq = &sequences_buffer[sequences_metadata[next_from].pattern_offset];
            memcpy(h_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs);
            cudaMemcpyAsync(d_seq_buffer_unpacked, h_seq_buffer_unpacked,
                            bytes_to_copy_seqs, cudaMemcpyHostToDevice,
                            stream1);
            CUDA_CHECK_ERR

            size_t next_batch_size = next_to - next_from + 1;

            mem_needed_packed = 0;
            // Prepare metadata for next batch (calculate)
            for (int i=0; i<next_batch_size; i++) {
                // Pattern
                sequence_pair_t* curr_alignment = &sequences_metadata[next_from + i];
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
        }

        // Make sure that alignment kernel has finished before packing the next
        // batch sequences
        cudaStreamSynchronize(stream2);

        LOG_DEBUG("Batch %d/%d computed", batch+1, num_batchs);

        int alignments_computed_cpu = 0;
        for (int i=0; i<curr_batch_size; i++) {
            int real_i = i + from;
            if (!results_h[i].finished) {
                alignments_computed_cpu++;

                size_t toffset = sequences_metadata[real_i].text_offset;
                size_t poffset = sequences_metadata[real_i].pattern_offset;

                const char* text = &sequences_buffer[toffset];
                const char* pattern = &sequences_buffer[poffset];

                size_t tlen = sequences_metadata[real_i].text_len;
                size_t plen = sequences_metadata[real_i].pattern_len;

                results_h[i].distance = compute_alignment_cpu(
                    pattern, text,
                    plen, tlen,
                    penalties.x, penalties.o, penalties.e
                );

                // TODO: CIGAR ?
            }
        }

        if (alignments_computed_cpu > 0) {
            LOG_INFO("(Batch %d) %d/%d alignemnts could not be computed on the"
                     " GPU and where offloaded to the CPU.",
                     batch, alignments_computed_cpu, curr_batch_size)
        }

        memcpy(&results[from], results_h, curr_batch_size * sizeof(alignment_result_t));
        memcpy(&backtraces[from], bt_results_offloaded_h, curr_batch_size * bt_result_offloaded_size * sizeof(wfa_backtrace_t));

        // TODO: check correctness/ recover cigar from previous batch while the
        // kernel from current batch is running (?)

        // Check correctness if asked
        if (check_correctness) {
            LOG_DEBUG("Checking batch %d correctnes.", batch+1);
            const uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
            float avg_distance = 0;
            int correct = 0;
            int incorrect = 0;
            CLOCK_INIT()
            CLOCK_START()
            #pragma omp parallel for reduction(+:avg_distance,correct,incorrect)
            for (int i=from; i<=to; i++) {
                // TODO: Check also CPU distances ?
                if (!results_h[i-from].finished) {
                    correct++;
                    avg_distance += results_h[i-from].distance;
                    continue;
                }

                size_t toffset = sequences_metadata[i].text_offset;
                size_t poffset = sequences_metadata[i].pattern_offset;

                const char* text = &sequences_buffer[toffset];
                const char* pattern = &sequences_buffer[poffset];

                size_t tlen = sequences_metadata[i].text_len;
                size_t plen = sequences_metadata[i].pattern_len;

                int distance = results_h[i-from].distance;
                char* cigar = recover_cigar(text, pattern, tlen,
                                            plen, results_h[i-from].backtrace,
                                            bt_results_offloaded_h + backtraces_offloaded_elements*(i-from),
                                            results_h[i-from]);

                bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
                bool correct_affine_d = check_affine_distance(text, pattern, tlen,
                                                              plen, distance,
                                                              penalties, cigar);

                if (!correct_cigar) {
                    LOG_ERROR("Incorrect cigar %d (%d). Distance: %d. CIGAR: %s\n", i-from, i, distance, cigar);
                }

                avg_distance += distance;

                if (correct_cigar && correct_affine_d) {
                    correct++;
                } else {
                    incorrect++;
                }
                free(cigar);
            }

            avg_distance /= curr_batch_size;
            CLOCK_STOP()
            LOG_INFO("(Batch %d) correct=%d Incorrect=%d Average score=%f (%.3f"
                     " alignments per second checked)\n", batch, correct,
                     incorrect, avg_distance, curr_batch_size/ CLOCK_SECONDS);
        }

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
    cudaFree(wf_data_buffer);
    cudaFreeHost(h_seq_buffer_unpacked);
    cudaFreeHost(h_seq_metadata);
    cudaFreeHost(results_h);
    cudaFreeHost(bt_results_offloaded_h);
    CUDA_CHECK_ERR
}

#if 0
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

    uint8_t* wf_data_buffer;
    allocate_wf_data_buffer_d(&wf_data_buffer, max_distance,
                              penalties, num_alignments);

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
        wf_data_buffer,
        max_distance,
        threads_per_block,
        0
    );

    cudaFree(results_d);
    cudaFree(bt_offloaded_d);
    cudaFree(wf_data_buffer);
}
#endif
