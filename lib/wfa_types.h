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

typedef int16_t wfa_offset_t;

// 31 --------------- 15 ---------------- 0
//       backtrace        last_bt_offset
typedef uint32_t wfa_backtrace_packed_t;

typedef struct __align__(8) {
    uint32_t backtrace;
    uint32_t prev;
} wfa_backtrace_t;

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

#define MAX_STEPS 128

// Height * width
#define BT_OFFLOADED_ELEMENTS(max_steps) \
                        ((max_steps) * 2 + 1) \
                        * ((max_steps) * 2 / 16)

#endif
