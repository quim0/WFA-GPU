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

#ifndef VERIFICATION_H
#define VERIFICATION_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "wfa_types.h"
#include "affine_penalties.h"
#include "alignment_results.h"

#define OPS_PER_BT_WORD 32

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)

bool check_cigar_edit (const char* text,
                           const char* pattern,
                           const int tlen,
                           const int plen,
                           const char* curr_cigar);


bool check_affine_distance (const char* text,
                                     const char* pattern,
                                     const int tlen,
                                     const int plen,
                                     const int distance,
                                     const affine_penalties_t penalties,
                                     const char* cigar);

char* recover_cigar (const char* text,
                     const char* pattern,
                     const size_t tlen,
                     const size_t plen,
                     wfa_backtrace_t final_backtrace,
                     wfa_backtrace_t* offloaded_backtraces_array,
                     alignment_result_t result);

#endif // Header guard
