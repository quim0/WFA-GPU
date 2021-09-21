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

#include "lib/wfa_types.h"
#include "external/WFA/wavefront/wavefront_align.h"

int compute_alignment_cpu (const char* const pattern, const char* const text,
                           const size_t plen, const size_t tlen,
                           const int x, const int o, const int e) {
    // !!! This is the affine_penalties_t struct from the CPU wfa library, do
    // not include the GPU library "affine_penalties.h" header as this struct
    // would be redefined with different fields (giving a compile error).
    affine_penalties_t penalties = {
        .match = 0,
        .mismatch = x,
        .gap_opening = o,
        .gap_extension = e
    };

    wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;

    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.match = 0;
    attributes.affine_penalties.mismatch = x;
    attributes.affine_penalties.gap_opening = o;
    attributes.affine_penalties.gap_extension = e;

    wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(
        &attributes
    );

    wavefront_align(wf_aligner, pattern, plen, text, tlen);
    const int score = wf_aligner->cigar.score;

    wavefront_aligner_delete(wf_aligner);
    // WFA cpu library returns "cost" that is negative, as GPU library use
    // distance as a metric, its needed to convert it to positive
    return -score;
}