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
#include "include/wfa_gpu.h"

int main() {
    // Create and initialize the aligner structure
    wfagpu_aligner_t aligner = {0};
    bool ok = wfagpu_initialize_aligner(&aligner);
    if (!ok) {
        return 1;
    }

    // Add the sequences to align: wfagpu_add_sequences(aligner, query, target)
    wfagpu_add_sequences(&aligner,
                  "CCTAACCCTAACCCTAACCCTAAACCCTAAACCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCTAACCCCAACCCTAACCCCAACCCCACCTATACCCTAACCGCACCCCAACCCAAACCCCAA",
                  "CCTAACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCC");
    wfagpu_add_sequences(&aligner,
                         "TAACCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAAACCTAACCCTACCCTAACCCAAACCCTAACCCCAACCCCAACCCCAACCC",
                         "TAACCCTACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTTAACCCTAACCCTTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCC");
    wfagpu_add_sequences(&aligner,
                         "GGTGAGGGTGAGGGTTAGGGTTAGGGTGAGGGTTAGGGTGAGGGTGAGGGTGAGGGTGAGGGTAGGGGTAGGGGGTAGGGGTGGGGGTGGGTGAGGGGTAGGGGTAGGGAGAGGGGTAGGGTTAGGGGTGGGGGGAGGGTGAGGGTTA",
                         "GGTGAGGGTGAGGGTTAGGGTTAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTGAGGGTTAGGGTGTTAGAGGGTTAGGGTTAGGGTTAGGGTTAGGGTTAGGGGGTTAGGGGGTTAGGGGGTTAGGGGGTTAGGGTGAGGGTGAGGGT");
    wfagpu_add_sequences(&aligner,
                         "CCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCTAACCCTCACCTAACCCCAACCCTAACCCCAACCCCAACCCACACCCTAC",
                         "CCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCCTAACCCTAACCCTAACCCTAC");

    // Initialize the alignment parameters (*after* the sequences have been
    // added)
    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    ok = wfagpu_initialize_parameters(&aligner, penalties);
    if (!ok)
        return 1;

    // Optionally set batch size, in this case (we have 4 alginments), setting a
    // batch size of 2 will execute two batches
    ok = wfagpu_set_batch_size(&aligner, 2);
    if (!ok)
        return 1;

    // Compute the optimal alignment path (CIGAR) or only the distance
    aligner.alignment_options.compute_cigar = true;

    // Align all sequence pairs
    ok = wfagpu_align(&aligner);
    if (!ok)
        return 1;

    // Read the results
    for (int i=0; i<aligner.alignment_options.num_alignments; i++) {
        printf("Alignment %d:\n", i);
        printf("\tScore: %d\n", aligner.results[i].error);
        printf("\tCIGAR: %s\n", aligner.results[i].cigar.buffer);
    }
    // Free internal structures memory
    wfagpu_destroy_aligner(&aligner);
}
