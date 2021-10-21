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

#ifndef WFA_TYPES_H
#define WFA_TYPES_H

#include "stdint.h"

typedef int32_t wfa_offset_t;
typedef uint32_t wfa_bt_prev_t;
typedef uint64_t wfa_bt_vector_t;

#define wfa_backtrace_bits 64
typedef struct {
    wfa_bt_vector_t backtrace;
    wfa_bt_prev_t prev;
} wfa_backtrace_t;

// ! DO NOT CHANGE THE ORDER OF STRUCT MEMBERS !
// The current order is assumed for doing 128 bit loads from global memory on
// the GPU.
#ifdef __CUDACC__
typedef __align__(16) struct
#else
typedef struct
#endif
{
    wfa_offset_t offset;       // 32 bits
    wfa_bt_prev_t bt_prev;     // 32 bits
    wfa_bt_vector_t bt_vector; // 64 bits
} wfa_cell_t;

typedef enum {
    OP_INS = 1,
    OP_SUB = 2,
    OP_DEL = 3
} affine_op_t;

typedef enum {
    GAP_OPEN = 1,
    GAP_EXTEND
} gap_op_t;

typedef struct {
    int len;
    char* buffer;
} wfa_cigar_t;

// Height * width
#define BT_OFFLOADED_ELEMENTS(max_steps) \
                        (((max_steps) * 2 + 1) \
                        * ((max_steps) * 2 / 16))

#define BT_OFFLOADED_RESULT_ELEMENTS(max_steps) \
                        ((max_steps) * 2 / 16)

#endif
