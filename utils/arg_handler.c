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

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#include "arg_handler.h"

// Check if current string arguent is in options, and return it. Return NULL if
// it's not found.
option_t* check_long_arg (const char* arg, const options_t options) {
    for (int i=0; i<options.len; i++) {
        if (!strcmp(options.options[i].long_arg, arg)) {
            return &options.options[i];
        }
    }
    return NULL;
}

// Check if current string arguent is in options, and return it. Return NULL if
// it's not found.
option_t* check_short_arg (const char* arg, const options_t options) {
    for (int i=0; i<options.len; i++) {
        if (options.options[i].short_arg == arg[0]) {
            return &options.options[i];
        }
    }
    return NULL;
}

bool parse_arg (char* const value, option_t* option) {
    switch (option->type) {
        case ARG_INT:
            option->value.int_val = atoll(value);
            break;
        case ARG_STR:
            option->value.str_val = value;
            break;
        case ARG_CHAR:
            if (strlen(value) == 0) {
                return false;
            }
            option->value.char_val = value[0];
            break;
        case ARG_FLOAT:
            option->value.float_val = atof(value);
            break;
        case ARG_BOOL:
            if (!strcmp(value, "false") || !strcmp(value, "0")) {
                option->value.bool_val = false;
            } else {
                option->value.bool_val = true;
            }
            break;
        default:
            return false;
    }

    return true;
}

void clean_options (options_t* options) {
    for (int i=0; i<options->len; i++) {
        options->options[i].parsed = false;
    }
}

option_t* get_option (const options_t options, char short_arg) {
    for (int i=0; i<options.len; i++) {
        if (options.options[i].short_arg == short_arg) {
            return &(options.options[i]);
        }
    }
    return NULL;
}

bool parse_args (const int argc, char** argv, options_t options) {

    clean_options(&options);

    bool read_value = false;
    option_t* curr_option = NULL;
    for (int i=0; i<argc; i++) {
        if (!read_value) {
            int arg_len = strlen(argv[i]);
            if (argv[i][0] == '-') {
                if (arg_len >= 2 && (argv[i][1] == '-')) {
                    // Long form of the argument e.g. --arg
                    curr_option = check_long_arg(argv[i] + 2, options);
                } else {
                    // Short form of the argument e.g. -a
                    curr_option = check_short_arg(argv[i] + 1, options);
                }
                if (curr_option != NULL) {
                    if (curr_option->type != ARG_NO_VALUE) {
                        // Read value the next iteration
                        read_value = true;
                    } else {
                        // For arguments without value associated, e.g --debug
                        curr_option->value.set = true;
                        curr_option->parsed = true;
                    }
                }
            }
        } else {
            bool success = parse_arg(argv[i], curr_option);
            if (success) {
                curr_option->parsed = true;
            } else {
                fprintf(stderr,
                        "Error parsing argument: %s.", curr_option->name);
                return false;
            }
            read_value = false;
            curr_option = NULL;
        }
    }

    // Check if all required arguments have been parsed
    for (int i=0; i<options.len; i++) {
        option_t curr_option = options.options[i];
        if (curr_option.required && !curr_option.parsed) {
            fprintf(stderr,
                    "Missing required parameter \"%s\"\n", curr_option.long_arg);
            return false;
        }
    }
    return true;
}

void print_usage (const options_t options) {
    fprintf(stderr, "Options:\n");
    for (int j=0; j<options.num_categories; j++) {
        const char* cat_name = options.categories[j];
        fprintf(stderr, "[%s]\n", cat_name);
        for (int i=0; i<options.len; i++) {
            const option_t curr_option = options.options[i];

            if (curr_option.category != j) continue;

            char* type_str;
            switch (curr_option.type) {
                case ARG_INT:
                    type_str = "int";
                    break;
                case ARG_STR:
                    type_str = "string";
                    break;
                case ARG_CHAR:
                    type_str = "char";
                    break;
                case ARG_FLOAT:
                    type_str = "float";
                    break;
                case ARG_BOOL:
                    type_str = "bool";
                    break;
                case ARG_NO_VALUE:
                    type_str = "flag";
                    break;
            }

            if (curr_option.type == ARG_NO_VALUE) {
                if (curr_option.required) {
                    fprintf(stderr, "\t-%c, --%-30s(required) %s: %s\n",
                            curr_option.short_arg, curr_option.long_arg,
                            curr_option.name, curr_option.description);
                } else {
                    fprintf(stderr, "\t-%c, --%-30s%s: %s\n",
                            curr_option.short_arg, curr_option.long_arg,
                            curr_option.name, curr_option.description);
                }
            } else {
                if (curr_option.required) {
                    fprintf(stderr, "\t-%c, --%-30s(%s, required) %s: %s\n",
                            curr_option.short_arg, curr_option.long_arg,
                            type_str, curr_option.name, curr_option.description);
                } else {
                    fprintf(stderr, "\t-%c, --%-30s(%s) %s: %s\n",
                            curr_option.short_arg, curr_option.long_arg,
                            type_str, curr_option.name, curr_option.description);
                }
            }

        }

    }
}
