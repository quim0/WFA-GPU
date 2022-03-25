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

#include "alignment_results.h"

bool initialize_wfa_results (wfa_alignment_result_t** results,
                             const size_t num_alignments,
                             const size_t cigar_length) {
    if (results == NULL) return false;

    *results = calloc(num_alignments, sizeof(wfa_alignment_result_t));    
    if (*results == NULL) {
        return false;
    }

    for (int i=0; i<num_alignments; i++) {
        wfa_alignment_result_t* curr_result = (*results) + i;
        curr_result->cigar.buffer = calloc(cigar_length, 1);
        if (curr_result->cigar.buffer == NULL) return false;
        curr_result->cigar.buffer_size = cigar_length;
    }
    return true;
}

bool destroy_wfa_results (wfa_alignment_result_t* results,
                             const size_t num_alignments) {
    if (results == NULL) return false;

    for (int i=0; i<num_alignments; i++) {
        wfa_alignment_result_t* curr_result = results + i;
        if (curr_result->cigar.buffer != NULL) free(curr_result->cigar.buffer);
    }

    free(results);
    return true;
}

