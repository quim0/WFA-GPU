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

#include "affine_penalties.h"

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
    // For banded executions, this sets the maximum and minimum diagonal of any
    // WF (max_k=band, min_k=-band). If the WF area created from band to -band
    // (2 * band + 1) is < than the number of CUDA threads per block, some
    // threads will never do any useful work. Set to 0 to not use any band.
    int band;
    // Number of alignments that are computed on each kernel launch. Set to 0
    // to have only one batch.
    size_t batch_size;
    // Total number of alignments to compute.
    size_t num_alignments;
    affine_penalties_t penalties;
} wfa_alignment_options_t;

#endif
