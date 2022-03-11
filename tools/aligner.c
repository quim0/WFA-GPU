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

#define NUM_ARGUMENTS 9
#define NUM_CATEGORIES 3

typedef enum {
    CAT_IO,
    CAT_ALIGN,
    CAT_SYS
} menu_categories_t;

const char* menu_categories[] = {
    "Input/Output",      // 0
    "Alignment Options", // 1
    "System"             // 2
};

int main(int argc, char** argv) {

    option_t options_arr[NUM_ARGUMENTS] = {
        // 0
        {.name = "Input sequences file",
         .description = "File containing the sequences to align.",
         .category = CAT_IO,
         .short_arg = 'i',
         .long_arg = "input",
         .required = true,
         .type = ARG_STR
         },
        // 1
        {.name = "Number of alignments",
         .description = "Number of alignments to read from the file (default=all"
                        " alignments)",
         .category = CAT_IO,
         .short_arg = 'n',
         .long_arg = "num-alignments",
         .required = false,
         .type = ARG_INT
         },
        // 2
        {.name = "Affine penalties",
         .description = "Gap-affine penalties for the alignment, in format x,o,e",
         .category = CAT_ALIGN,
         .short_arg = 'g',
         .long_arg = "affine-penalties",
         .required = true,
         .type = ARG_STR
         },
        // 3
        {.name = "Check",
         .description = "Check for alignment correctness",
         .category = CAT_SYS,
         .short_arg = 'c',
         .long_arg = "check",
         .required = false,
         .type = ARG_NO_VALUE
         },
        // 4
        {.name = "Maximum error allowed",
         .description = "Maximum error that the kernel will be able to "
                        "compute (default = maximum possible error of first "
                        "alignment)",
         .category = CAT_ALIGN,
         .short_arg = 'e',
         .long_arg = "max-distance",
         .required = false,
         .type = ARG_INT
         },
        // 5
        {.name = "Number of CUDA threads per alginment",
         .description = "Number of CUDA threads per block, each block computes"
                        " one or multiple alignment",
         .category = CAT_SYS,
         .short_arg = 't',
         .long_arg = "threads-per-block",
         .required = false,
         .type = ARG_INT
         },
         // 6
        {.name = "Batch size",
         .description = "Number of alignments per batch.",
         .category = CAT_ALIGN,
         .short_arg = 'b',
         .long_arg = "batch-size",
         .required = false,
         .type = ARG_INT
         },
         // 7
        {.name = "GPU workers",
         .description = "Number of blocks ('workers') to be running on the GPU.",
         .category = CAT_SYS,
         .short_arg = 'w',
         .long_arg = "workers",
         .required = false,
         .type = ARG_INT
         },
         // 8
        {.name = "Band",
         .description = "Wavefront band (highest and lower diagonal that will be computed).",
         .category = CAT_ALIGN,
         .short_arg = 'B',
         .long_arg = "band",
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

    options_t options = {options_arr, NUM_ARGUMENTS, menu_categories, NUM_CATEGORIES};

    bool success = parse_args(argc, argv, options);
    if (!success) {
        print_usage(options);
        exit(1);
    }

    sequence_reader_t sequence_reader = {0};
    char* sequences_file = get_option(options, 'i')->value.str_val;
    init_sequence_reader(&sequence_reader, sequences_file);

    size_t sequences_read = 0;
    option_t* opt_sequences_read = get_option(options, 'n');
    if (opt_sequences_read->parsed) {
        sequences_read = opt_sequences_read->value.int_val * 2;
    }

    bool check = get_option(options, 'c')->parsed;

    affine_penalties_t penalties = {0};
    // TODO: This is insecure but works for now, parse it better
    int x, o, e;
    sscanf(get_option(options, 'g')->value.str_val, "%d,%d,%d", &x, &o, &e);

    if (x < 0) x *= -1;
    if (o < 0) o *= -1;
    if (e < 0) e *= -1;

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
    option_t* opt_max_distance = get_option(options, 'e');
    if (opt_max_distance->parsed) {
        max_distance = opt_max_distance->value.int_val;
    } else {
        max_distance = sequence_reader.sequences_metadata[0].text_len
                       + sequence_reader.sequences_metadata[0].pattern_len;
        max_distance /= 1.5;
        if (max_distance > 10000) {
            LOG_WARN("Automatically genereated maximum error supported by the"
                     " kernel seems to be very high, to avoid running out of "
                     "memory, consider limiting the maximum error with the "
                     "'-e' argument.");
        }
    }

    // Threads per block
    int threads_per_block;
    option_t* opt_tpb = get_option(options, 't');
    if (opt_tpb->parsed) {
        threads_per_block = opt_tpb->value.int_val;
    } else {
        // If it is not provided by the user, use the maximum distance as a
        // hint.
        const size_t max_wf_size = 2 * max_distance + 1;
        if (max_wf_size <= 96)       threads_per_block = 64;
        else if (max_wf_size <= 192) threads_per_block = 128;
        else if (max_wf_size <= 380) threads_per_block = 256;
        else if (max_wf_size <= 768) threads_per_block = 512;
        else                         threads_per_block = 1024;
    }

    LOG_INFO("Penalties: M=0, X=%d, O=%d, E=%d. Maximum distance: %d",
             penalties.x, penalties.o, penalties.e, max_distance)

    size_t num_alignments = sequence_reader.num_sequences_read / 2;

    int batch_size;
    option_t* opt_batch_size = get_option(options, 'b');
    if (opt_batch_size->parsed) {
        batch_size = opt_batch_size->value.int_val;
    } else {
        batch_size = num_alignments;
    }

    if (batch_size <= 0) {
        LOG_ERROR("Incorrect batch size (%d).", batch_size)
        exit(-1);
    }

    LOG_INFO("Batch size = %d.", batch_size)

    int num_blocks;
    option_t* opt_num_blocks = get_option(options, 'w');
    if (opt_num_blocks->parsed) {
        num_blocks = opt_num_blocks->value.int_val;
    } else {
        // TODO: Get this from num_threads and GPU capabilities
        const int num_sm = get_cuda_SM_count(0);
        const int warps_per_block = threads_per_block / 32;
        // Assume we can get an occupancy of 32 warps / block
        const int blocks_per_sm = 32 / warps_per_block;
        num_blocks = num_sm * blocks_per_sm;
    }

    if (num_blocks <= 0) {
        LOG_ERROR("Incorrect number of workers (%d).", num_blocks)
        exit(-1);
    }

    LOG_INFO("Number of GPU workers = %d.", num_blocks)

    int band;
    option_t* opt_band = get_option(options, 'B');
    if (opt_band->parsed) {
        band = opt_band->value.int_val;
        if (band <= 0) {
            LOG_ERROR("Band must positive (band=%d).", band)
            exit(-1);
        }
        LOG_INFO("Banded execution. Max diagonal: %d, Min diagonal: %d", band, -band)
    } else {
        band = -1;
    }

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
        band,
        check
    );

    CLOCK_STOP()
    printf("Alignment computed. Wall time: %.3fs (%.3f alignments per second)\n",
           CLOCK_SECONDS, num_alignments / CLOCK_SECONDS);

    free(results);
    free(backtraces);
}
