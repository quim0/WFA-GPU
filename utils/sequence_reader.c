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

#include <string.h>
#include "sequence_reader.h"
#include "logger.h"
#include "../lib/wfa_types.h"

static bool grow_sequence_buffer (char** sequences_buffer, size_t* sequences_buffer_size) {
    // Grow 256MiB of memory each call
    const size_t grow_size = 1L << 28;
    char* new_mem_chunk = realloc(*sequences_buffer,
                                  *sequences_buffer_size + grow_size);
    if (new_mem_chunk == NULL) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }

    *sequences_buffer = new_mem_chunk;
    memset(*sequences_buffer + *sequences_buffer_size,
           0,
           grow_size);
    *sequences_buffer_size += grow_size;

    return true;
}

static bool grow_metadata_array (sequence_pair_t** sequences_metadata, size_t* sequences_metadata_size) {
    const size_t alignments_to_grow = 50000;
    sequence_pair_t* new_mem_chunk = realloc(
            *sequences_metadata,
            (*sequences_metadata_size + alignments_to_grow) * sizeof(sequence_pair_t)
            );

    if (new_mem_chunk == NULL) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }

    *sequences_metadata = new_mem_chunk;
    memset(*sequences_metadata + *sequences_metadata_size,
           0,
           alignments_to_grow * sizeof(sequence_pair_t));
    *sequences_metadata_size += alignments_to_grow;
    return true;
}

bool init_sequence_reader (sequence_reader_t* reader, char* seq_file) {
    reader->fp = fopen(seq_file, "r");
    if (!reader->fp) {
        LOG_ERROR("Could not open file %s", seq_file);
        return false;
    }

    // TODO: CudaMallocHost instead of calloc (?)

    // Sequences buffer allocation
    // Allocate a 256MiB chunk, then this will increase as necessary
    size_t seq_size_to_alloc = 1L << 28;
    reader->sequences_buffer = calloc(seq_size_to_alloc, 1);
    if (!reader->sequences_buffer) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }
    reader->sequences_buffer_size = seq_size_to_alloc;

    // Sequences metadata initial allocation
    size_t init_num_alignments = 50000;
    reader->sequences_metadata = calloc(init_num_alignments,
                                        sizeof(sequence_pair_t));
    if (!reader->sequences_metadata) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }
    reader->sequences_metadata_size = init_num_alignments;

    reader->num_sequences_read = 0;
    return true;
}

bool init_sequence_reader_fasta (sequence_reader_fasta_t* reader, char* seq_file_target, char* seq_file_query) {
    reader->fp_target = fopen(seq_file_target, "r");
    if (!reader->fp_target) {
        LOG_ERROR("Could not open file %s", seq_file_target);
        return false;
    }

    reader->fp_query = fopen(seq_file_query, "r");
    if (!reader->fp_query) {
        LOG_ERROR("Could not open file %s", seq_file_query);
        return false;
    }

    // Sequences buffer allocation
    // Allocate a 256MiB chunk, then this will increase as necessary
    size_t seq_size_to_alloc = 1L << 28;
    reader->sequences_buffer = calloc(seq_size_to_alloc, 1);
    if (!reader->sequences_buffer) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }
    reader->sequences_buffer_size = seq_size_to_alloc;

    // Sequences metadata initial allocation
    size_t init_num_alignments = 50000;
    reader->sequences_metadata = calloc(init_num_alignments,
                                        sizeof(sequence_pair_t));
    if (!reader->sequences_metadata) {
        LOG_ERROR("Could not allocate memory.");
        return false;
    }
    reader->sequences_metadata_size = init_num_alignments;

    reader->num_sequences_read = 0;
    return true;
}

// Reads at most n sequences from a .seq file, if n=0, read all file
bool read_n_sequences (sequence_reader_t* reader, size_t* n) {
    if (!reader->fp || !reader->sequences_buffer || !reader->sequences_metadata) {
        LOG_ERROR("Sequence reader not initialized.");
        return false;
    }

    char* lineptr = NULL;
    size_t line_size = 0;
    size_t read_bytes = 0;
    size_t curr_sequence_idx = 0;
    while ((*n == 0) || (curr_sequence_idx < *n)) {
        ssize_t curr_read_size = getline(&lineptr, &line_size, reader->fp);
        if (curr_read_size == -1) {
            break;
        }

        if (lineptr == NULL) {
            LOG_ERROR("getline could not allocate memory.");
            return false;
        }

        // Only the delimiter character has been read (empty line)
        if (curr_read_size == 1) continue;

        // Add padding so that sequences are 32 bits aligned
        // curr_read_size - 1 to remove the initial '>' or '<', the newline is
        // not substracted as it's compensated with the nullbyte needed to be
        // added at the end.
        size_t curr_seq_size_padded = (curr_read_size - 1)
                                      + (4 - ((curr_read_size - 1) % 4));

        // Be sure that there is enough space in the sequence buffer to allocate
        // the current sequence.
        while ((read_bytes + curr_seq_size_padded) > reader->sequences_buffer_size) {
            if (!grow_sequence_buffer(&(reader->sequences_buffer), &(reader->sequences_buffer_size))) {
                LOG_ERROR("Could not allocate memory for the sequence buffer.");
                return false;
            }
        }

        // Be sure there is enough space in the metadata array
        if ((curr_sequence_idx / 2) >= reader->sequences_metadata_size) {
            if (!grow_metadata_array(&(reader->sequences_metadata), &(reader->sequences_metadata_size))) {
                LOG_ERROR("Could not allocate memory for the sequence metadata"
                          " array.");
                return false;
            }
        }

        char* curr_seq_ptr = reader->sequences_buffer + read_bytes;

        sequence_pair_t* curr_alignment = \
                    &(reader->sequences_metadata[curr_sequence_idx / 2]);

        curr_alignment->has_N = false;

        if ((curr_sequence_idx % 2) == 0) {
            // Next alignment, read PATTERN
            if (curr_read_size <= 2 || lineptr[0] != '>') {
                LOG_ERROR("Invalid file format. Could not read pattern in "
                          "line %zu",
                          curr_sequence_idx);
                free(lineptr);
                return false;
            }

            // lineptr + 1 to remove the initial '>' of the sequence in the file
            // size - 2 to remove the initial '>' and the final newline
            memcpy(curr_seq_ptr, lineptr + 1, curr_read_size - 2);
            curr_alignment->pattern_offset = read_bytes;
            curr_alignment->pattern_len = curr_read_size - 2;

            read_bytes += curr_seq_size_padded;
        } else {
            // Read TEXT
            if (curr_read_size <= 2 || lineptr[0] != '<') {
                LOG_ERROR("Invalid file format. Could not read pattern in "
                          "line %zu",
                          curr_sequence_idx);
                free(lineptr);
                return false;
            }

            // lineptr + 1 to remove the initial '<' of the sequence in the file
            // size - 2 to remove the initial '<' and the final newline
            memcpy(curr_seq_ptr, lineptr + 1, curr_read_size - 2);
            curr_alignment->text_offset = read_bytes;
            curr_alignment->text_len = curr_read_size - 2;

            read_bytes += curr_seq_size_padded;
        }

        curr_sequence_idx++;
    }

    free(lineptr);
    *n = curr_sequence_idx;
    reader->num_sequences_read = curr_sequence_idx;

    LOG_DEBUG("Read %zu sequences.", *n);
    return true;
}

// Reads at most n sequences from a .seq file, if n=0, read all file
bool read_n_sequences_fasta (sequence_reader_fasta_t* reader, size_t* n) {
    if (!reader->fp_target || !reader->fp_query || !reader->sequences_buffer || !reader->sequences_metadata) {
        LOG_ERROR("Sequence reader not initialized.");
        return false;
    }

    size_t line_size = 0;
    size_t read_bytes = 0;
    size_t curr_sequence_idx = 0;

    // A sequence can be split along multiple lines
    char partial_sequence[MAX_SEQ_LEN] = {0};

    // Ignore first line as it should be sequence information and start with '>'
    char* lineptr_query = NULL;
    char* lineptr_target = NULL;
    int rsize_query = getline(&lineptr_query, &line_size, reader->fp_query);
    int rsize_target = getline(&lineptr_target, &line_size, reader->fp_target);

    // EOF
    if (rsize_target == -1 || rsize_query == -1) {
        LOG_ERROR("Empty FASTA file.");
        return false;
    }

    if (lineptr_query == NULL || lineptr_target == NULL) {
        LOG_ERROR("getline could not allocate memory.");
        return false;
    }

    char first_char = lineptr_query[0];
    for (int i=1; first_char == ' '; i++) {
        first_char = lineptr_query[i];
    }
    if (first_char != '>') {
        LOG_ERROR("Invalid FASTA query file. '>' character not found.");
        return false;
    }

    first_char = lineptr_target[0];
    for (int i=1; first_char == ' '; i++) {
        first_char = lineptr_target[i];
    }
    if (first_char != '>') {
        LOG_ERROR("Invalid FASTA target file. '>' character not found.");
        return false;
    }

    free(lineptr_query);
    free(lineptr_target);

    int finished_files = 0;

    while ((finished_files < 2) && ((*n == 0) || (curr_sequence_idx < *n))) {
        size_t partial_sequence_length = 0;

        while (1) {
            ssize_t curr_read_size;
            char* lineptr = NULL;
            if ((curr_sequence_idx % 2) == 0) // read line from query (pattern) file
                curr_read_size = getline(&lineptr, &line_size, reader->fp_query);
            else // read line from target (text) file
                curr_read_size = getline(&lineptr, &line_size, reader->fp_target);

            // EOF
            if (curr_read_size == -1) {
                finished_files++;
                break;
            }

            if (lineptr == NULL) {
                LOG_ERROR("getline could not allocate memory.");
                return false;
            }

            // Only the delimiter character has been read (empty line)
            // Ignore empty lines
            if (curr_read_size == 1) continue;

            // Use ">" as a delimiter, ignore spaces
            char first_char = lineptr[0];
            for (int i=1; first_char == ' '; i++) {
                first_char = lineptr[i];
            }
            if (first_char == '>') {
                break;
            } else {
                if ((partial_sequence_length + curr_read_size - 1) >= MAX_SEQ_LEN) {
                    LOG_ERROR("Sequence longer than %ld characters detected, aborting.", MAX_SEQ_LEN);
                    return false;
                }
                // read_size - 1 to remove the final newline
                memcpy(&partial_sequence[0] + partial_sequence_length,
                       lineptr,
                       curr_read_size - 1);
                partial_sequence_length += curr_read_size - 1;
            }
            free(lineptr);
            lineptr = NULL;
        }

        // Add padding so that sequences are 32 bits aligned
        // The newline is not substracted as it's compensated with the nullbyte
        // needed to be added at the end.
        size_t curr_seq_size_padded = partial_sequence_length + (4 - (partial_sequence_length % 4));

        // Be sure that there is enough space in the sequence buffer to allocate
        // the current sequence.
        while ((read_bytes + curr_seq_size_padded) > reader->sequences_buffer_size) {
            if (!grow_sequence_buffer(&(reader->sequences_buffer), &(reader->sequences_buffer_size))) {
                LOG_ERROR("Could not allocate memory for the sequence buffer.");
                return false;
            }
        }

        // Be sure there is enough space in the metadata array
        if ((curr_sequence_idx / 2) >= reader->sequences_metadata_size) {
            if (!grow_metadata_array(&(reader->sequences_metadata), &(reader->sequences_metadata_size))) {
                LOG_ERROR("Could not allocate memory for the sequence metadata"
                          " array.");
                return false;
            }
        }

        char* curr_seq_ptr = reader->sequences_buffer + read_bytes;

        sequence_pair_t* curr_alignment = \
                    &(reader->sequences_metadata[curr_sequence_idx / 2]);

        curr_alignment->has_N = false;

        if ((curr_sequence_idx % 2) == 0) {
            // Save sequence data and metadata (pattern)
            memcpy(curr_seq_ptr, partial_sequence, partial_sequence_length);
            curr_alignment->pattern_offset = read_bytes;
            curr_alignment->pattern_len = partial_sequence_length;
        } else {
            memcpy(curr_seq_ptr, partial_sequence, partial_sequence_length);
            curr_alignment->text_offset = read_bytes;
            curr_alignment->text_len = partial_sequence_length;
        }

        read_bytes += curr_seq_size_padded;
        curr_sequence_idx++;
    }

    *n = curr_sequence_idx;
    reader->num_sequences_read = curr_sequence_idx;

    LOG_DEBUG("Read %zu sequences.", *n);
    return true;
}

void destroy_reader (sequence_reader_t* reader) {
    if (reader->sequences_buffer) free(reader->sequences_buffer);
    if (reader->sequences_metadata) free(reader->sequences_metadata);
}
