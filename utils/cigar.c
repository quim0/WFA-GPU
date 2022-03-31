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

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "utils/cigar.h"
#include "utils/wf_clock.h"

#define ITOA_BUFFER_SIZE 20

bool insert_ops (wfa_cigar_t* const cigar,
                 const char op,
                 const unsigned int rep) {
    if (rep == 0) return true;

    char itoa_buffer[ITOA_BUFFER_SIZE] = {0};
    snprintf(itoa_buffer, ITOA_BUFFER_SIZE, "%d", rep);
    const int itoa_len = strnlen(itoa_buffer, ITOA_BUFFER_SIZE);
    const int total_len_to_add = itoa_len + 1;
    const int free_cigar_size = cigar->buffer_size - cigar->last_free_position;

    if (free_cigar_size <= total_len_to_add) {
        // Make CIGAR buffer bigger by 50%
        int new_cigar_buffer_len = cigar->buffer_size * 1.5;
        cigar->buffer = realloc(cigar->buffer, new_cigar_buffer_len);
        if (cigar->buffer == NULL) return false;
        memset(cigar->buffer + cigar->buffer_size,
               0,
               new_cigar_buffer_len - cigar->buffer_size);
        
        // New values
        cigar->buffer_size = new_cigar_buffer_len;
    }

    memcpy(cigar->buffer + cigar->last_free_position,
           itoa_buffer,
           itoa_len);
    *(cigar->buffer + cigar->last_free_position + itoa_len) = op;
    cigar->last_free_position += total_len_to_add;
    return true;
}

static wfa_offset_t extend_wavefront (
        const wfa_offset_t offset_val,
        const int curr_k,
        const char* const pattern,
        const int pattern_length,
        const char* const text,
        const int text_length) {
    int k = curr_k;
    int acc = 0;
    // Fetch pattern/text blocks
    uint64_t* pattern_blocks = (uint64_t*)(pattern+EWAVEFRONT_V(k,offset_val));
    uint64_t* text_blocks = (uint64_t*)(text+EWAVEFRONT_H(k,offset_val));
    // Compare 64-bits blocks
    uint64_t cmp = *pattern_blocks ^ *text_blocks;

    while (__builtin_expect(cmp==0,0)) {
        // Increment offset (full block)
        acc += 8;
        // Next blocks
        ++pattern_blocks;
        ++text_blocks;
        // Compare
        cmp = *pattern_blocks ^ *text_blocks;
    }

    // Count equal characters
    const int equal_right_bits = __builtin_ctzl(cmp);
    const int equal_chars = equal_right_bits/8;
    acc += equal_chars;

    return acc;
}

void recover_cigar_affine (char* text,
                           char* pattern,
                           const size_t tlen,
                           const size_t plen,
                           wfa_backtrace_t final_backtrace,
                           wfa_backtrace_t* offloaded_backtraces_array,
                           alignment_result_t result,
                           wfa_cigar_t *cigar) {
    int k=0;
    wfa_offset_t offset = 0;
    bool extending = false;

    if (result.distance == 0) {
       insert_ops(cigar, 'M', tlen);
       return;
    }

    // Set sentinels
    *(unsigned char*)(pattern + plen) = 0xff;

    int iterations = result.num_bt_blocks;
    affine_op_t op, prev_op;
    unsigned int rep = 0;
    while (iterations > 0) {

        wfa_backtrace_t* backtrace = &offloaded_backtraces_array[iterations - 1];
        uint32_t backtrace_val = backtrace->backtrace;

        int steps = OPS_PER_BT_WORD - (__builtin_clz(backtrace_val) / 2);
        // __builtin_clz(0) is undefined
        if (backtrace_val == 0) steps = 0;

        //op = (affine_op_t)((backtrace_val >> ((steps - 1) * 2)) & 3);
        for (int d=0; d<steps; d++) {
            op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

            if (op != prev_op && rep != 0) {
                insert_ops(cigar, ops_ascii[prev_op], rep);
                rep = 0;
            }

            if (!extending) {
                wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
                if (acc > 0 && rep != 0) {
                    // flush rep
                    insert_ops(cigar, ops_ascii[prev_op], rep);
                    rep = 0;

                }
                insert_ops(cigar, 'M', acc);
                offset += acc;
            }

            switch (op) {
                // k + 1
                case OP_DEL:
                    extending = true;
                    k--;
                    break;
                // k
                case OP_SUB:
                    if (extending) {
                        extending = false;
                        op = OP_NOOP;
                        // Don't need to insert anything here, just the
                        // delimiter
                        rep--;
                    } else {
                        offset++;
                    }
                    break;
                // k - 1
                case OP_INS:
                    extending = true;
                    k++;
                    offset++;
                    break;
            }
            prev_op = op;
            rep++;
        }

        if (op != prev_op && rep != 0) {
            insert_ops(cigar, ops_ascii[op], rep);
            rep = 0;
        }

        if (!extending) {
            // Last exension
            wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
            if (acc > 0 && rep != 0) {
                // flush rep
                insert_ops(cigar, ops_ascii[prev_op], rep);
                rep = 0;
            }
            insert_ops(cigar, 'M', acc);
            offset += acc;
        }

        iterations--;
    }

    // Final round with the final backtrace word that is not in the offlaoded
    // array
    uint32_t backtrace_val = final_backtrace.backtrace;

    int steps = OPS_PER_BT_WORD - (__builtin_clz(backtrace_val) / 2);
    // __builtin_clz(0) is undefined
    if (backtrace_val == 0) steps = 0;

    for (int d=0; d<steps; d++) {
        prev_op = op;
        op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

        if (op != prev_op && rep != 0) {
            insert_ops(cigar, ops_ascii[prev_op], rep);
            rep = 0;
        }

        if (!extending) {
            wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
            if (acc > 0 && rep != 0) {
                // flush rep
                insert_ops(cigar, ops_ascii[prev_op], rep);
                rep = 0;
            }
            insert_ops(cigar, 'M', acc);
            offset += acc;
        }

        switch (op) {
            // k + 1
            case OP_DEL:
                extending = true;
                k--;
                break;
            // k
            case OP_SUB:
                if (extending) {
                    extending = false;
                    op = OP_NOOP;
                    rep--;
                } else {
                    offset++;
                }
                break;
            // k - 1
            case OP_INS:
                extending = true;
                k++;
                offset++;
                break;
        }

        rep++;
    }

    if (rep != 0) {
        insert_ops(cigar, ops_ascii[op], rep);
        rep = 0;
    }

    if (!extending) {
        // Last exension
        wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
        if (acc > 0 && rep != 0) {
            // flush rep
            insert_ops(cigar, ops_ascii[op], rep);
            rep = 0;
        }
        insert_ops(cigar, 'M', acc);
        offset += acc;
    }

    // Remove sentinels
    *(unsigned char*)(pattern + plen) = 0;
}
