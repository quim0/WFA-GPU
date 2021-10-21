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

#ifndef SEQUENCE_ALIGNMENT_CUH
#define SEQUENCE_ALIGNMENT_CUH

#include <stdint.h>
#include "wfa_types.h"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "utils/sequences.h"

#define STORE_CELL(cell, offset, bt, bt_prev) { \
    *(uint4*)(&cell) = make_uint4(              \
        (offset),                               \
        (bt_prev),                              \
        (uint64_t)(bt) >> 32,                   \
        (bt) & 0xffffffff);                     \
    }

#define LOAD_CELL(cell) *(uint4*)(&(cell))
#define UINT4_TO_OFFSET(cell) ((wfa_offset_t)(cell.x))
#define UINT4_TO_BT_PREV(cell) ((wfa_offset_t)(cell.y))
#define UINT4_TO_BT_VECTOR(cell) ((wfa_bt_vector_t)                     \
                                  (((wfa_bt_vector_t)(cell.z)) << 32) | \
                                  (cell.w)                              \
                                 )

typedef struct {
    int16_t hi;
    int16_t lo;
    wfa_cell_t* cells;
    bool exist;
} wfa_wavefront_t;

__global__ void alignment_kernel (
                            const char* packed_sequences_buffer,
                            const sequence_pair_t* sequences_metadata,
                            const size_t num_alignments,
                            const int max_steps,
                            uint8_t* const wf_data_buffer,
                            const affine_penalties_t penalties,
                            wfa_backtrace_t* offloaded_backtraces_global,
                            wfa_backtrace_t* offloaded_backtraces_results,
                            alignment_result_t* results);

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)
#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

#endif
