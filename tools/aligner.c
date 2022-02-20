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

#include "utils/logger.h"
#include "utils/wf_clock.h"
#include "utils/arg_handler.h"
#include "utils/sequence_reader.h"
#include "utils/verification.h"
#include "utils/device_query.cuh"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "wfa_types.h"
#include "batch_async.cuh"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define NUM_ARGUMENTS 8

int main(int argc, char** argv) {

    option_t options_arr[NUM_ARGUMENTS] = {
        // 0
        {.name = "Sequences file",
         .description = "File containing the sequences to align.",
         .short_arg = 'f',
         .long_arg = "file",
         .required = true,
         .type = ARG_STR
         },
        // 1
        {.name = "Number of alignments",
         .description = "Number of alignments to read from the file (default=all"
                        " alignments)",
         .short_arg = 'n',
         .long_arg = "num-alignments",
         .required = false,
         .type = ARG_INT
         },
        // 2
        {.name = "Affine penalties",
         .description = "Gap-affine penalties for the alignment, in format x,o,e",
         .short_arg = 'g',
         .long_arg = "affine-penalties",
         .required = true,
         .type = ARG_STR
         },
        // 3
        {.name = "Check",
         .description = "Check for alignment correctness",
         .short_arg = 'c',
         .long_arg = "check",
         .required = false,
         .type = ARG_NO_VALUE
         },
        // 4
        {.name = "Maximum distance allowed",
         .description = "Maximum distance that the kernel will be able to "
                        "compute (default = maximum distance of first alignment)",
         .short_arg = 'd',
         .long_arg = "max-distance",
         .required = false,
         .type = ARG_INT
         },
        // 5
        {.name = "Number of threads per alginment",
         .description = "Number of threads per block, each block computes one"
                        " alignment",
         .short_arg = 't',
         .long_arg = "threads",
         .required = false,
         .type = ARG_INT
         },
         // 6
        {.name = "Batch size",
         .description = "Number of alignments per batch.",
         .short_arg = 'b',
         .long_arg = "batch-size",
         .required = false,
         .type = ARG_INT
         },
         // 7
        {.name = "GPU workers",
         .description = "Number of blocks ('workers') to be running on the GPU.",
         .short_arg = 'w',
         .long_arg = "workers",
         .required = false,
         .type = ARG_INT
         },
    };

    int cuda_devices = 0;
    get_num_cuda_devices(&cuda_devices);
    if (cuda_devices == 0) {
        LOG_ERROR("No CUDA devices detected.")
        exit(-1);
    }

    int major, minor;
    get_cuda_capability(0, &major, &minor);

    char* device_name = get_cuda_dev_name(0);

    LOG_INFO("Using CUDA device \"%s\" with capability %d.%d",
             device_name, major, minor)

    free(device_name);

    options_t options = {options_arr, NUM_ARGUMENTS};

    bool success = parse_args(argc, argv, options);
    if (!success) {
        print_usage(options);
        exit(1);
    }

    sequence_reader_t sequence_reader = {0};
    char* sequences_file = options.options[0].value.str_val;
    init_sequence_reader(&sequence_reader, sequences_file);

    size_t sequences_read = 0;
    if (options.options[1].parsed) {
        sequences_read = options.options[1].value.int_val * 2;
    }

    bool check = options.options[3].parsed;

    affine_penalties_t penalties = {0};
    // TODO: This is insecure but works for now, parse it better
    int x, o, e;
    sscanf(options.options[2].value.str_val, "%d,%d,%d", &x, &o, &e);

    penalties.x = x;
    penalties.o = o;
    penalties.e = e;

    DEBUG_CLOCK_INIT()
    DEBUG_CLOCK_START()

    if (!read_n_sequences(&sequence_reader, &sequences_read)) {
        LOG_ERROR("Error reading file: %s (%s).", sequences_file, strerror(errno));
        exit(1);
    }

    DEBUG_CLOCK_STOP("File read.")

    int max_distance;
    if (options.options[4].parsed) {
        max_distance = options.options[4].value.int_val;
    } else {
        max_distance = sequence_reader.sequences_metadata[0].text_len
                       + sequence_reader.sequences_metadata[0].pattern_len;
    }

    // Threads per block
    int threads_per_block;
    if (options.options[5].parsed) {
        threads_per_block = options.options[5].value.int_val;
    } else {
        // TODO: Arbitrary number of threads
        threads_per_block = 512;
    }

    LOG_INFO("Penalties: M=0, X=%d, O=%d, E=%d. Maximum distance: %d",
             penalties.x, penalties.o, penalties.e, max_distance)

    size_t num_alignments = sequence_reader.num_sequences_read / 2;

    int batch_size;
    if (options.options[6].parsed) {
        batch_size = options.options[6].value.int_val;
    } else {
        batch_size = num_alignments;
    }

    if (batch_size <= 0) {
        LOG_ERROR("Incorrect batch size (%d).", batch_size)
        exit(-1);
    }

    LOG_INFO("Batch size = %d.", batch_size)

    int num_blocks;
    if (options.options[7].parsed) {
        num_blocks = options.options[7].value.int_val;
    } else {
        // TODO: Get this from num_threads and GPU capabilities
        num_blocks = 68;
    }

    if (num_blocks <= 0) {
        LOG_ERROR("Incorrect number of workers (%d).", num_blocks)
        exit(-1);
    }

    LOG_INFO("Number of GPU workers = %d.", num_blocks)

    alignment_result_t* results = (alignment_result_t*)calloc(num_alignments, sizeof(alignment_result_t));
    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
    // TODO: * batch_size instead of * num_alignments (?)
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    backtraces_offloaded_elements * num_alignments,
                                                    sizeof(wfa_backtrace_t)
                                                    );

    CLOCK_INIT()
    CLOCK_START()

    launch_alignments_batched(
        sequence_reader.sequences_buffer,
        sequence_reader.sequences_buffer_size,
        sequence_reader.sequences_metadata,
        num_alignments,
        penalties,
        results,
        backtraces,
        max_distance,
        threads_per_block,
        num_blocks,
        batch_size,
        check
    );

    CLOCK_STOP()
    printf("Alignment computed. Wall time: %.3fs (%.3f alignments per second)\n",
           CLOCK_SECONDS, num_alignments / CLOCK_SECONDS);

    /*
    float avg_distance = 0;
    int correct = 0;
    int incorrect = 0;

    if (check) {
        CLOCK_START()
        printf("Checking correctness...\n");
        #pragma omp parallel for reduction(+:avg_distance,correct,incorrect)
        for (int i=0; i<num_alignments; i++) {
            size_t toffset = sequence_reader.sequences_metadata[i].text_offset;
            size_t poffset = sequence_reader.sequences_metadata[i].pattern_offset;

            char* text = &sequence_reader.sequences_buffer[toffset];
            char* pattern = &sequence_reader.sequences_buffer[poffset];

            size_t tlen = sequence_reader.sequences_metadata[i].text_len;
            size_t plen = sequence_reader.sequences_metadata[i].pattern_len;

            int distance = results[i].distance;
            char* cigar = recover_cigar(text, pattern, tlen,
                                        plen,results[i].backtrace,
                                        backtraces + backtraces_offloaded_elements*i,
                                        results[i]);

            bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
            bool correct_affine_d = check_affine_distance(text, pattern, tlen,
                                                          plen, distance,
                                                          penalties, cigar);

            avg_distance += distance;

            if (correct_cigar && correct_affine_d) {
                correct++;
            } else {
                incorrect++;
            }

            free(cigar);
        }

        avg_distance /= num_alignments;
        CLOCK_STOP()
        printf("Correct=%d Incorrect=%d Average score=%f (%.3f alignments per"
               " second checked)\n",
               correct, incorrect, avg_distance, num_alignments / CLOCK_SECONDS);
    }
    */

    free(results);
    free(backtraces);
}
