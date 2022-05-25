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
    wfagpu_initialize_aligner(&aligner);

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
    wfagpu_initialize_parameters(&aligner, penalties);

    // Optionally set batch size, in this case (we have 4 alginments), setting a
    // batch size of 2 will execute two batches
    wfagpu_set_batch_size(&aligner, 2);

    // With wfagpu_initialize_parameters all the alignment parameters are set to
    // its default, but it is possible to change them to obtain a better
    // performance depending on the dataset.

    // Maximum error (score) supported by the kernel. This is a very important
    // parameter to achieve good performance. Try to keep this parameter as low
    // as possible. It is better to set it for majority of the dataset, letting
    // the corner cases with higher error to be offloaded to the CPU.
    aligner.alignment_options.max_error = 100;

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
    aligner.alignment_options.threads_per_block = 128;

    // Number of GPU "workers" (alignmers). This parameter depends on the GPU
    // architecture, and the number of threads. Only set this if you really
    // understand what you are doing.
    aligner.alignment_options.num_workers = get_num_workers(aligner.alignment_options.threads_per_block);

    // Set to BAND_NONE to not use adaptative band and get exact alignments.
    // Otherwise, set the maximum diagonal that will be initially computed.
    aligner.alignment_options.band = BAND_NONE;

    // Compute the optimal alignment path (CIGAR) or only the distance
    aligner.alignment_options.compute_cigar = true;

    // Align all sequence pairs
    wfagpu_align(&aligner);

    // Read the results
    for (int i=0; i<aligner.alignment_options.num_alignments; i++) {
        printf("Alignment %d:\n", i);
        printf("\tScore: %d\n", aligner.results[i].error);
        printf("\tCIGAR: %s\n", aligner.results[i].cigar.buffer);
    }
    // Free internal structures memory
    wfagpu_destroy_aligner(&aligner);
}
