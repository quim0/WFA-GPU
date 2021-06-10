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
#include "affine_penalties.h"
#include "batch_async.cuh"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define NUM_ARGUMENTS 3

int main(int argc, char** argv) {

    option_t options_arr[NUM_ARGUMENTS] = {
        {.name = "Sequences file",
         .description = "File containing the sequences to align.",
         .short_arg = 'f',
         .long_arg = "file",
         .required = true,
         .type = ARG_STR
         },
        {.name = "Number of alignments",
         .description = "Number of alignments to read from the file (default=all"
                        " alignments)",
         .short_arg = 'n',
         .long_arg = "num-alignments",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Affine penalties",
         .description = "Gap-affine penalties for the alignment, in format x,o,e",
         .short_arg = 'g',
         .long_arg = "affine-penalties",
         .required = true,
         .type = ARG_STR
         },
    };

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

    CLOCK_INIT()
    CLOCK_START()

    launch_batch_async(
        sequence_reader.sequences_buffer,
        sequence_reader.sequences_buffer_size,
        sequence_reader.sequences_metadata,
        sequence_reader.num_sequences_read / 2,
        penalties
    );

    CLOCK_STOP("Alignment computed")
}
