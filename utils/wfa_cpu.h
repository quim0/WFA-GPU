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

#ifndef WFA_CPU_H
#define WFA_CPU_H

#include "utils/sequences.h"
#include "lib/wfa_types.h"
#include "lib/alignment_results.h"

#ifdef __cplusplus
extern "C" {
#endif

int compute_alignments_cpu_threaded (const int batch_size,
                                      const int from,
                                      alignment_result_t* results,
                                      wfa_alignment_result_t* alignment_results,
                                      const sequence_pair_t* sequences_metadata,
                                      char* sequences_buffer,
                                      wfa_backtrace_t* backtraces,
                                      uint32_t backtraces_offloaded_elements,
                                      const int x, const int o, const int e,
                                      const bool adaptative);

int compute_alignment_cpu (const char* const pattern, const char* const text,
                           const size_t plen, const size_t tlen,
                           const int x, const int o, const int e);


void pprint_cigar_cpu (const char* const pattern, const char* const text,
                           const size_t plen, const size_t tlen,
                           const int x, const int o, const int e);

#ifdef __cplusplus
} // extern C
#endif

#endif
