/*
 * Copyright (c) 2022 Quim Aguado
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

#ifndef ALIGNMENT_PARAMETERS_H
#define ALIGNMENT_PARAMETERS_H

#include "utils/sequences.h"
#include "utils/device_query.cuh"
#include "affine_penalties.h"

#define BAND_NONE (-1)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    // Maximum error that for the kernel to support. Having a bigger error
    // implies using more GPU memory, and having more work to clean the internal
    // structures between alignments.
    int max_error;
    // Number of CUDA threads per GPU worker. This needs to be related with
    // the expected maximum WF size to make the best use of the GPU resources.
    int threads_per_block;
    // Number of GPU workers (usually this needs to be chosen depending on the
    // number of SMs in the GPU and the number of threads per block).
    int num_workers;
    // For banded executions, this sets the maximum and minimum initial diagonal
    // (max_k=band, min_k=-band). If the WF area created from band to -band (2 *
    // band + 1) is < than the number of CUDA threads per block, some threads
    // will never do any useful work. Set to <=0 to not use any band.
    int band;
    // Number of alignments that are computed on each kernel launch. Set to 0
    // to have only one batch.
    size_t batch_size;
    // Total number of alignments to compute.
    size_t num_alignments;
    affine_penalties_t penalties;
    // If true, compute the optimal alignment path (CIGAR). If false just
    // compute the alignment distance
    bool compute_cigar;
} wfa_alignment_options_t;

static int wfa_get_threads_per_alignment (const size_t max_error) {
    // Simple function to decide an adecuate number of threads per block
    // (aligner).
    int threads_per_block;
    const size_t max_wf_size = 2 * max_error + 1;
    if (max_wf_size <= 128)       threads_per_block = 64;
    else if (max_wf_size <= 256) threads_per_block = 128;
    else if (max_wf_size <= 512) threads_per_block = 256;
    else if (max_wf_size <= 1024) threads_per_block = 512;
    else                         threads_per_block = 1024;
    return threads_per_block;
}

static int get_num_workers (const int num_threads) {
    // Try to get maximum occupancy, not leaving unused resources on the GPU
    // Get the number of SMs of the GPU (device 0). Try to get an occupancy of
    // 32 warps per block (for older GPUs this may be lower).
    const int num_sm = get_cuda_SM_count(0);
    const int warps_per_block = num_threads / 32;
    const int blocks_per_sm = 32 / warps_per_block;
    return num_sm * blocks_per_sm;
}

static void wfagpu_set_default_options (wfa_alignment_options_t* wfa_options,
                                     sequence_pair_t* sequences_metadata,
                                     affine_penalties_t penalties,
                                     size_t num_alignments) {
    int slen = MAX(sequences_metadata[0].pattern_len, sequences_metadata[0].text_len);
    slen = MIN(slen*0.1, slen);
    wfa_options->max_error = slen * MAX(penalties.x, MAX(penalties.o, penalties.e));

    int num_threads = wfa_get_threads_per_alignment(wfa_options->max_error);
    wfa_options->threads_per_block = num_threads;
    wfa_options->num_workers = get_num_workers(num_threads);
    wfa_options->band = BAND_NONE;
    wfa_options->num_alignments = num_alignments;
    if (num_alignments > 10)
        wfa_options->batch_size = num_alignments / 10;
    else
        wfa_options->batch_size = num_alignments;
    wfa_options->penalties = penalties;
    wfa_options->compute_cigar = false;
}

#endif
