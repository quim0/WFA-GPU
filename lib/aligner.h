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

#ifndef ALIGNER_H
#define ALIGNER_H

#include <inttypes.h>
#include "utils/sequences.h"
#include "alignment_parameters.h"
#include "alignment_results.h"

#define WFA_ALIGN_32_BITS(x) ((x) + (4 - ((x) % 4)))

typedef char wfagpu_seqbuf_t;

typedef struct {
    wfagpu_seqbuf_t* sequences_buffer;
    size_t sequences_buffer_len;
    sequence_pair_t* sequences_metadata;
    size_t sequences_metadata_len;
    size_t num_sequence_pairs;
    wfa_alignment_result_t* results;
    int64_t last_sequence_pair_idx;
    wfa_alignment_options_t alignment_options;
} wfagpu_aligner_t;

#if __cplusplus
extern "C" {
#endif

bool wfagpu_add_sequences (wfagpu_aligner_t* aligner,
                    const char* query,
                    const char* target);

bool wfagpu_initialize_aligner (wfagpu_aligner_t* aligner);

bool wfagpu_initialize_parameters (wfagpu_aligner_t* aligner,
                                   affine_penalties_t penalties);

bool wfagpu_set_batch_size (wfagpu_aligner_t* aligner, size_t batch_size);

void wfagpu_destroy_aligner (wfagpu_aligner_t* aligner);

bool wfagpu_align (wfagpu_aligner_t* aligner);

#if __cplusplus // end of extern "C"
}
#endif

#endif // ALIGNER_H
