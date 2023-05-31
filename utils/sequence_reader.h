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
#include <stdlib.h>
#include <stdio.h>

#include "sequences.h"

typedef struct {
    FILE* fp;
    char* sequences_buffer;
    size_t sequences_buffer_size;
    sequence_pair_t* sequences_metadata;
    size_t sequences_metadata_size;
    size_t num_sequences_read;
} sequence_reader_t;

typedef struct {
    FILE* fp_target;
    FILE* fp_query;
    char* sequences_buffer;
    size_t sequences_buffer_size;
    sequence_pair_t* sequences_metadata;
    size_t sequences_metadata_size;
    size_t num_sequences_read;
} sequence_reader_fasta_t;

bool init_sequence_reader (sequence_reader_t* reader, char* seq_file);
bool init_sequence_reader_fasta (sequence_reader_fasta_t* reader, char* seq_file_target, char* seq_file_query);
// Reads at most n sequences, if n=0, read all file. N will be updated with the
// real number of sequences read.
bool read_n_sequences (sequence_reader_t* reader, size_t* n);
bool read_n_sequences_fasta (sequence_reader_fasta_t* reader, size_t* n);
void destroy_reader (sequence_reader_t* reader);

#endif
