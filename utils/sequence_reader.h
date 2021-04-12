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


#ifndef SEQUENCE_READER_H
#define SEQUENCE_READER_H

#include <stdbool.h>

typedef struct {
    size_t text_offset;
    size_t pattern_offset;
    unsigned int text_len;
    unsigned int pattern_len;
} sequence_pair_t;

typedef struct {
    FILE* fp;
    char* sequences_buffer;
    size_t sequences_buffer_size;
    sequence_pair_t* sequences_metadata;
    size_t sequences_metadata_size;
    size_t num_sequences_read;
} sequence_reader_t;

bool grow_sequence_buffer (sequence_reader_t* reader);
bool grow_metadata_array (sequence_reader_t* reader);
bool init_sequence_reader (sequence_reader_t* reader, char* seq_file);
bool read_n_sequences (sequence_reader_t* reader, size_t n);

// TODO: Destroy reader

#endif
