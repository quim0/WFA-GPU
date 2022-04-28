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

#include <string.h>
#include "aligner.h"
#include "batch_async.cuh"
#include "utils/logger.h"

// 1MiB
#define DEFAULT_SEQ_BUF_SIZE (1 << 20)
#define DEFAULT_SEQ_METADATA_SIZE (10000)
#define INITIAL_CIGAR_LEN (50)

static bool initialize_sequences_buffer (wfagpu_aligner_t* aligner) {
    if (aligner->sequences_buffer != NULL) {
        LOG_WARN("Sequence buffer pointer is not NULL, not initializing.");
        return false;
    }
    aligner->sequences_buffer = (wfagpu_seqbuf_t*)calloc(DEFAULT_SEQ_BUF_SIZE, sizeof(wfagpu_seqbuf_t));
    if (aligner->sequences_buffer == NULL) {
        LOG_ERROR("Can not initialize sequences buffer.");
        return false;
    }
    aligner->sequences_buffer_len = DEFAULT_SEQ_BUF_SIZE;
    return true;
}

static bool grow_sequences_buffer (wfagpu_aligner_t* aligner) {
    if (aligner->sequences_buffer == NULL) {
        LOG_WARN("Sequence buffer not initialized, not resizing.");
        return false;
    }

    size_t new_size = (aligner->sequences_buffer_len + DEFAULT_SEQ_BUF_SIZE) * sizeof(wfagpu_seqbuf_t);
    wfagpu_seqbuf_t* new_buf = (wfagpu_seqbuf_t*)realloc(aligner->sequences_buffer, new_size);
    if (new_buf == NULL) {
        LOG_ERROR("Can now grow sequences buffer (realloc failed).");
        return false;
    }
    aligner->sequences_buffer = new_buf;
    aligner->sequences_buffer_len = new_size;
    return true;
}

static bool initialize_sequences_metadata (wfagpu_aligner_t* aligner) {
    if (aligner->sequences_metadata != NULL) {
        LOG_WARN("Sequence metadata pointer is not NULL, not initializing.");
        return false;
    }
    aligner->sequences_metadata = (sequence_pair_t*)calloc(DEFAULT_SEQ_METADATA_SIZE, sizeof(sequence_pair_t));
    if (aligner->sequences_metadata == NULL) {
        LOG_ERROR("Can not initialize sequences metadata.");
        return false;
    }
    aligner->sequences_metadata_len = DEFAULT_SEQ_METADATA_SIZE;
    return true;
}

static bool grow_sequences_metadata (wfagpu_aligner_t* aligner) {
    if (aligner->sequences_metadata == NULL) {
        LOG_WARN("Sequence metadata buffer not initialized, not resizing.");
        return false;
    }

    size_t new_size = (aligner->sequences_metadata_len + DEFAULT_SEQ_METADATA_SIZE) * sizeof(sequence_pair_t);
    sequence_pair_t* new_buf = (sequence_pair_t*)realloc(aligner->sequences_metadata, new_size);
    if (new_buf == NULL) {
        LOG_ERROR("Can now grow sequences metadata buffer (realloc failed).");
        return false;
    }
    aligner->sequences_metadata = new_buf;
    aligner->sequences_metadata_len = new_size;
    return true;
}

bool wfagpu_add_sequences (wfagpu_aligner_t* aligner,
                           const char* query,
                           const char* target) {
    // +1 for the nullbyte
    size_t pattern_offset;
    if (aligner->last_sequence_pair_idx == -1) {
        pattern_offset = 0;
    } else {
        sequence_pair_t last_pair_metadata = aligner->sequences_metadata[aligner->last_sequence_pair_idx];
        pattern_offset = WFA_ALIGN_32_BITS(last_pair_metadata.text_offset + last_pair_metadata.text_len + 1);
    }
    const size_t pattern_len = strlen(query);
    const size_t text_offset = WFA_ALIGN_32_BITS(pattern_offset + pattern_len + 1);
    const size_t text_len = strlen(target);

    while ((text_offset + text_len + 1) >= aligner->sequences_buffer_len) {
        if (!grow_sequences_buffer(aligner)) {
            LOG_ERROR("Sequences do not fit in memory. Aborting.");
            return false;
        }
    }

    if ((aligner->last_sequence_pair_idx + 1) >= aligner->sequences_metadata_len) {
        if (!grow_sequences_metadata(aligner)) {
            LOG_ERROR("Can not resize sequence metadata buffer. Aborting.");
            return false;
        }
    }

    const size_t curr_seq_idx = (aligner->last_sequence_pair_idx) + 1;

    strcpy(aligner->sequences_buffer + pattern_offset, query);
    strcpy(aligner->sequences_buffer + text_offset, target);

    aligner->sequences_metadata[curr_seq_idx].pattern_offset = pattern_offset;
    aligner->sequences_metadata[curr_seq_idx].pattern_len = pattern_len;
    aligner->sequences_metadata[curr_seq_idx].text_offset = text_offset;
    aligner->sequences_metadata[curr_seq_idx].text_len = text_len;

    (aligner->last_sequence_pair_idx)++;
    (aligner->num_sequence_pairs)++;

    return true;
}

void wfagpu_initialize_aligner (wfagpu_aligner_t* aligner) {
    memset(aligner, 0, sizeof(wfagpu_aligner_t));
    aligner->last_sequence_pair_idx = -1;
    initialize_sequences_buffer(aligner);
    initialize_sequences_metadata(aligner);
}

void wfagpu_initialize_parameters (wfagpu_aligner_t* aligner,
                                   affine_penalties_t penalties) {
    wfagpu_set_default_options(&(aligner->alignment_options),
                               aligner->sequences_metadata,
                               penalties,
                               aligner->num_sequence_pairs);
    initialize_wfa_results(&(aligner->results),
                           aligner->num_sequence_pairs,
                           INITIAL_CIGAR_LEN);
}

void wfagpu_set_batch_size (wfagpu_aligner_t* aligner, size_t batch_size) {
    aligner->alignment_options.batch_size = batch_size;
}

void wfagpu_destroy_aligner (wfagpu_aligner_t* aligner) {
    if (aligner->sequences_buffer != NULL) free(aligner->sequences_buffer);
    if (aligner->sequences_metadata != NULL) free(aligner->sequences_metadata);
    if (aligner->results != NULL) destroy_wfa_results(aligner->results, aligner->num_sequence_pairs);
}

void wfagpu_align (wfagpu_aligner_t* aligner) {
    launch_alignments(
        aligner->sequences_buffer,
        aligner->sequences_buffer_len,
        aligner->sequences_metadata,
        aligner->results,
        aligner->alignment_options,
        false // Check if results are correct
    );
}
