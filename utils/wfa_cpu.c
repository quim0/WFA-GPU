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

#include "utils/wfa_cpu.h"
#include "utils/logger.h"
#include "utils/cigar.h"
#include "external/WFA/wavefront/wavefront_align.h"
#include "external/WFA/alignment/cigar.h"

// Compute multiple alignments reusing the aligner object
int compute_alignments_cpu_threaded (const int batch_size,
                                      const int from,
                                      alignment_result_t* results,
                                      wfa_alignment_result_t* alignment_results,
                                      const sequence_pair_t* sequences_metadata,
                                      char* sequences_buffer,
                                      wfa_backtrace_t* backtraces,
                                      uint32_t backtraces_offloaded_elements,
                                      const int x, const int o, const int e,
                                      const bool adaptative) {
    wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;

    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.match = 0;
    attributes.affine_penalties.mismatch = x;
    attributes.affine_penalties.gap_opening = o;
    attributes.affine_penalties.gap_extension = e;
    attributes.memory_mode = wavefront_memory_low;
    if (!adaptative) attributes.heuristic.strategy = wf_heuristic_none;

    int alignments_computed_cpu = 0;

    #pragma omp parallel reduction(+:alignments_computed_cpu)
    {
    // Each thread reuse the aligner
    wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(&attributes);
    #pragma omp for schedule(static)
    for (int i=0; i<batch_size; i++) {
        int real_i = i + from;
        if (!results[i].finished) {
            size_t toffset = sequences_metadata[real_i].text_offset;
            size_t poffset = sequences_metadata[real_i].pattern_offset;

            const char* text = &sequences_buffer[toffset];
            const char* pattern = &sequences_buffer[poffset];

            size_t tlen = sequences_metadata[real_i].text_len;
            size_t plen = sequences_metadata[real_i].pattern_len;

            wavefront_align(wf_aligner, pattern, plen, text, tlen);
            const int score = wf_aligner->cigar.score;

            results[i].distance = -score;
            alignment_results[real_i].error = -score;
            uint32_t cigar_len = wf_aligner->cigar.end_offset - wf_aligner->cigar.begin_offset;
            if (cigar_len >= alignment_results[real_i].cigar.buffer_size) {
                alignment_results[real_i].cigar.buffer = realloc(alignment_results[real_i].cigar.buffer, cigar_len + 1);
                if (alignment_results[real_i].cigar.buffer == NULL) {
                    LOG_ERROR("Can not realloc CIGAR buffer")
                    exit(-1);
                }
            }
            cigar_sprint(alignment_results[real_i].cigar.buffer, &wf_aligner->cigar, true);
            alignments_computed_cpu++;
        }
    }
    wavefront_aligner_delete(wf_aligner);

    #pragma omp for schedule(static)
    for (int i=from; i<=from+batch_size; i++) {
        if (!results[i-from].finished) continue;
        size_t toffset = sequences_metadata[i].text_offset;
        size_t poffset = sequences_metadata[i].pattern_offset;

        char* text = &sequences_buffer[toffset];
        char* pattern = &sequences_buffer[poffset];

        size_t tlen = sequences_metadata[i].text_len;
        size_t plen = sequences_metadata[i].pattern_len;

        int distance = results[i-from].distance;
        alignment_results[i].error = distance;
        recover_cigar_affine(text, pattern, tlen,
                 plen, results[i-from].backtrace,
                 backtraces + backtraces_offloaded_elements*(i-from),
                 results[i - from],
                 &alignment_results[i].cigar);
    }

    } // end of parallel region
    return alignments_computed_cpu;
}

int compute_alignment_cpu (const char* const pattern, const char* const text,
                           const size_t plen, const size_t tlen,
                           const int x, const int o, const int e) {
    wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;

    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.match = 0;
    attributes.affine_penalties.mismatch = x;
    attributes.affine_penalties.gap_opening = o;
    attributes.affine_penalties.gap_extension = e;
    attributes.heuristic.strategy = wf_heuristic_none;

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

void pprint_cigar_cpu (const char* const pattern, const char* const text,
                           const size_t plen, const size_t tlen,
                           const int x, const int o, const int e) {
    wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;

    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.match = 0;
    attributes.affine_penalties.mismatch = x;
    attributes.affine_penalties.gap_opening = o;
    attributes.affine_penalties.gap_extension = e;
    attributes.heuristic.strategy = wf_heuristic_none;

    wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(
        &attributes
    );

    wavefront_align(wf_aligner, pattern, plen, text, tlen);

    cigar_print_pretty(stdout,
      pattern,strlen(pattern),text,strlen(text),
      &wf_aligner->cigar,wf_aligner->mm_allocator);

    wavefront_aligner_delete(wf_aligner);
}
