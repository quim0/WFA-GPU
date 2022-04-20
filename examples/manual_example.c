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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/wfa_gpu.h"

void fill_sequences (char* sequences_buffer, sequence_pair_t* sequence_metadata) {
    // Copy all sequence pairs to the sequences_buffer, with a nullbyte
    // separating the sequences (automatically added by strcpy in this case).
    // **All sequences need to be 32-bit aligned.**
    //
    // The buffer should look like:
    // QUERY1\0TEXT1\0QUERY2\0TEXT2\0QUERY3\0TEXT3\0 ...
    //
    // Add the sequences information in the sequence_metadata array.
    // Pattern = query, Text = text

    // Alignment 1
    int pattern_offset = 0;
    strcpy(
        sequences_buffer,
        "CCTAACCCTAACCCTAACCCTAAACCCTAAACCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCTAACCCCAACCCTAACCCCAACCCCACCTATACCCTAACCGCACCCCAACCCAAACCCCAA");
    sequence_metadata[0].pattern_offset = 0;
    sequence_metadata[0].pattern_len = strlen(sequences_buffer);
    int text_offset = WFA_ALIGN_32_BITS(pattern_offset + sequence_metadata[0].pattern_len + 1);
    strcpy(
        sequences_buffer + text_offset,
        "CCTAACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCC");
    sequence_metadata[0].text_offset = text_offset; // +1 for the nullbyte
    sequence_metadata[0].text_len = strlen(sequences_buffer + sequence_metadata[0].text_offset);

    // Alignment 2
    pattern_offset = WFA_ALIGN_32_BITS(sequence_metadata[0].text_offset + sequence_metadata[0].text_len + 1);
    strcpy(
        sequences_buffer + pattern_offset,
        "TAACCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAAACCTAACCCTACCCTAACCCAAACCCTAACCCCAACCCCAACCCCAACCC");
    sequence_metadata[1].pattern_offset = pattern_offset;
    sequence_metadata[1].pattern_len = strlen(sequences_buffer + pattern_offset);

    text_offset = WFA_ALIGN_32_BITS(pattern_offset + sequence_metadata[1].pattern_len + 1);
    strcpy(
        sequences_buffer + text_offset,
        "TAACCCTACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTTAACCCTAACCCTTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCC");
    sequence_metadata[1].text_offset = text_offset;
    sequence_metadata[1].text_len = strlen(sequences_buffer + text_offset);

    // Alignment 3
    pattern_offset = WFA_ALIGN_32_BITS(sequence_metadata[1].text_offset + sequence_metadata[1].text_len + 1);
    strcpy(
        sequences_buffer + pattern_offset,
        "GGTGAGGGTGAGGGTTAGGGTTAGGGTGAGGGTTAGGGTGAGGGTGAGGGTGAGGGTGAGGGTAGGGGTAGGGGGTAGGGGTGGGGGTGGGTGAGGGGTAGGGGTAGGGAGAGGGGTAGGGTTAGGGGTGGGGGGAGGGTGAGGGTTA");
    sequence_metadata[2].pattern_offset = pattern_offset;
    sequence_metadata[2].pattern_len = strlen(sequences_buffer + pattern_offset);

    text_offset = WFA_ALIGN_32_BITS(pattern_offset + sequence_metadata[2].pattern_len + 1);
    strcpy(
        sequences_buffer + text_offset,
        "GGTGAGGGTGAGGGTTAGGGTTAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTTAGGGTGTTAGAGGGTTAGGGTTAGGGTTAGGGTTAGGGTTAGGGGGTTAGGGGGTTAGGGGGTTAGGGGGTTAGGGTGAGGGTGAGGGT");
    sequence_metadata[2].text_offset = text_offset;
    sequence_metadata[2].text_len = strlen(sequences_buffer + text_offset);

    // Alignment 4
    pattern_offset = WFA_ALIGN_32_BITS(sequence_metadata[2].text_offset + sequence_metadata[2].text_len + 1);
    strcpy(
        sequences_buffer + pattern_offset,
        "CCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCTAACCCTCACCTAACCCCAACCCTAACCCCAACCCCAACCCACACCCTAC");
    sequence_metadata[3].pattern_offset = pattern_offset;
    sequence_metadata[3].pattern_len = strlen(sequences_buffer + pattern_offset);

    text_offset = WFA_ALIGN_32_BITS(pattern_offset + sequence_metadata[3].pattern_len + 1);
    strcpy(
        sequences_buffer + text_offset,
        "CCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCCTAACCCTAACCCTAACCCTAC");
    sequence_metadata[3].text_offset = text_offset;
    sequence_metadata[3].text_len = strlen(sequences_buffer + text_offset);
}

int main() {
    const int num_alignments = 4;
    const int batch_size = 2;

    // Initialize sequences buffer and metadata structures
    // Allocate 2KiB to store the sequences
    const int sequences_buffer_size = 2048;
    char* sequences_buffer = (char*)calloc(sequences_buffer_size, 1);
    // Allocate space for 4 alignments
    sequence_pair_t* sequences_metadata = (sequence_pair_t*)calloc(num_alignments, sizeof(sequence_pair_t));
    fill_sequences(sequences_buffer, sequences_metadata);

    for (int i=0; i<num_alignments; i++) {
        printf("Alignment %d:\n", i);
        printf("\tQuery (offset=%zu, len=%d): %s\n", sequences_metadata[i].pattern_offset, sequences_metadata[i].pattern_len, &sequences_buffer[sequences_metadata[i].pattern_offset]);
        printf("\tText (offset=%zu, len=%d): %s\n", sequences_metadata[i].text_offset, sequences_metadata[i].text_len, &sequences_buffer[sequences_metadata[i].text_offset]);
    }
    printf("--------------------------------------------------------------------------------\n");

    // Initialize structures to store the results (CIGARS + scores)
    wfa_alignment_result_t* results;
    // Assign an initial CIGAR length (in bytes). The CIGARs buffers are resized
    // if they need more space, but having an adecuate initial CIGAR size can
    // improve performance.
    const int cigar_len = 50;
    if (!initialize_wfa_results(&results, num_alignments, cigar_len)) {
        fprintf(stderr, "Can not initialize CIGAR buffer.");
        exit(-1);
    }

    // Initialize alignment options
    wfa_alignment_options_t wfa_options = {0};

    // Maximum error (score) supported by the kernel. This is a very important
    // parameter to achieve good performance. Try to keep this parameter as low
    // as possible. It is better to set it for majority of the dataset, letting
    // the corner cases with higher error to be offloaded to the CPU.
    const size_t max_error = 100;
    wfa_options.max_error = max_error;

    // Number of CUDA threads per alignment. Some performance considerations:
    // - It needs to be a multiple of the warp size (i.e. multiple of 32)
    // - For most GPU architectures, it must be <= 1024
    // - It should be <= than the maxium WF size it can achieved with the
    //   supported maximum error. maximum WF size = 2 * max_error + 1
    //   e.g. for a maximum error of 200, the maximum WF size is 401, the number
    //   of threads per alignment should be <= 384 (closest multiple of 32)
    //   It is a good idea not to give the maximum number of thread per
    //   alignment, as some of those threads will never be used until the last
    //   iterations.
    //const int threads_per_block = get_threads_per_alignment(max_error);
    //wfa_options.threads_per_block = threads_per_block;
    wfa_options.threads_per_block = 128;

    // Number of GPU "workers" (alignmers). This parameter depends on the GPU
    // architecture, and the number of threads.
    wfa_options.num_workers = get_num_workers(wfa_options.threads_per_block);

    // Set to BAND_NONE to not use adaptative band and get exact alignments.
    // Otherwise, set the maximum diagonal that will be initially computed.
    wfa_options.band = BAND_NONE;


    wfa_options.batch_size = batch_size;
    wfa_options.num_alignments = num_alignments;

    // Penalties need to be positive, match score is always 0
    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    wfa_options.penalties = penalties;

    launch_alignments(
        sequences_buffer,
        sequences_buffer_size,
        sequences_metadata,
        results,
        wfa_options,
        false // Check if results are correct
    );

    // Read the results
    for (int i=0; i<wfa_options.num_alignments; i++) {
        printf("Alignment %d:\n", i);
        printf("\tScore: %d\n", results[i].error);
        printf("\tCIGAR: %s\n", results[i].cigar.buffer);
    }

    free(sequences_buffer);
    free(sequences_metadata);
}
