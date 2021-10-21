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

#include "utils/verification.h"
#include "utils/logger.h"
#include <string.h>

bool check_cigar_edit (const char* text,
                           const char* pattern,
                           const int tlen,
                           const int plen,
                           const char* curr_cigar) {
	int text_pos = 0, pattern_pos = 0;

	if (!curr_cigar)
		return false;

	const size_t cigar_len = strnlen(curr_cigar, tlen + plen);

	for (int i=0; i<cigar_len; i++) {
		char curr_cigar_element = curr_cigar[i];
		switch (curr_cigar_element) {
			case 'M':
				if (pattern[pattern_pos] != text[text_pos]) {
					LOG_DEBUG("Alignment not matching at CCIGAR index %d"
						  " (pattern[%d] = %c != text[%d] = %c)\n",
						  i, pattern_pos, pattern[pattern_pos],
						  text_pos, text[text_pos]);
					return false;
				}
				++pattern_pos;
				++text_pos;
				break;
			case 'I':
				++text_pos;
				break;
			case 'D':
				++pattern_pos;
				break;
			case 'X':
				if (pattern[pattern_pos] == text[text_pos]) {
					LOG_DEBUG("Alignment not mismatching at CCIGAR index %d"
						  " (pattern[%d] = %c == text[%d] = %c)\n",
						  i, pattern_pos, pattern[pattern_pos],
						  text_pos, text[text_pos]);
					return false;
				}
				++pattern_pos;
				++text_pos;
				break;
			default:
				break;
		}
	}

	if (pattern_pos != plen) {
		LOG_DEBUG("Alignment incorrect length, pattern-aligned: %d, "
			  "pattern-length: %d.\n", pattern_pos, plen);
		return false;
	}

	if (text_pos != tlen) {
		LOG_DEBUG("Alignment incorrect length, text-aligned: %d, "
			  "text-length: %d\n", text_pos, tlen);
		return false;
	}

	return true;
}

bool check_affine_distance (const char* text,
                                     const char* pattern,
                                     const int tlen,
                                     const int plen,
                                     const int distance,
                                     const affine_penalties_t penalties,
                                     const char* cigar) {
    bool extending_I = false, extending_D = false;
    int cigar_len = strnlen(cigar, tlen + plen);
    int result_distance = 0;

    for (int i=0; i<cigar_len; i++) {
        char curr_op = cigar[i];

        switch (curr_op) {
            case 'M':
                if (extending_D) extending_D = false;
                if (extending_I) extending_I = false;
                break;
            case 'I':
                if (extending_D) {
                    extending_D = false;
                    extending_I = true;
                    result_distance += penalties.o + penalties.e;
                } else if (extending_I) {
                    result_distance += penalties.e;
                } else {
                    extending_I = true;
                    result_distance += penalties.o + penalties.e;
                }
                break;
            case 'D':
                if (extending_I) {
                    extending_D = true;
                    extending_I = false;
                    result_distance += penalties.o + penalties.e;
                } else if (extending_D) {
                    result_distance += penalties.e;
                } else {
                    extending_D = true;
                    result_distance += penalties.o + penalties.e;
                }
                break;
            case 'X':
                if (extending_D) extending_D = false;
                if (extending_I) extending_I = false;
                result_distance += penalties.x;
                break;
            default:
                LOG_ERROR("Incorrect cigar generated")
        }
    }

    return result_distance == distance;
}


wfa_offset_t extend_wavefront (
        const wfa_offset_t offset_val,
        const int curr_k,
        const char* const pattern,
        const int pattern_length,
        const char* const text,
        const int text_length) {
    // Parameters
    int v = EWAVEFRONT_V(curr_k, offset_val);
    int h = EWAVEFRONT_H(curr_k, offset_val);
    wfa_offset_t acc = 0;
    while (v<pattern_length && h<text_length && pattern[v++]==text[h++]) {
      acc++;
    }
    return acc;
}

// Backtraces lis gets reversed, and final backtrace gets updated to it points
// to new list head
/*
int reverse_backtraces_list (wfa_backtrace_t* offloaded_backtraces,
                             wfa_backtrace_t* final_bactrace) {
    int prev = 0;
    int curr = final_bactrace->prev;
    int next = 0;
    wfa_backtrace_t* curr_bt = &offloaded_backtraces[curr];

    while (curr != 0) {
        // In the backtrace structure, "prev" is the equivalent to the next in a
        // normal linked list
        // Get the previous backtrace offset
        next = curr_bt->prev;

        // Update the current node
        curr_bt->prev = prev;

        // Move offsets to point one position down the list
        prev = curr;
        curr = next;
        curr_bt = &offloaded_backtraces[next];
    }

    // The offset of the new head of the backtraces list is on prev
    return prev;
}
*/

char* recover_cigar (const char* text,
                     const char* pattern,
                     const size_t tlen,
                     const size_t plen,
                     wfa_backtrace_t final_backtrace,
                     wfa_backtrace_t* offloaded_backtraces_array,
                     alignment_result_t result) {
    char* cigar_ascii = (char*)calloc(tlen + plen, 1);
    char* cigar_ptr = cigar_ascii;

    int k=0;
    wfa_offset_t offset = 0;
    bool extending = false;

    int iterations = result.num_bt_blocks;
    while (iterations > 0) {

        wfa_backtrace_t* backtrace = &offloaded_backtraces_array[iterations - 1];
        wfa_bt_vector_t backtrace_val = backtrace->backtrace;

        int steps = OPS_PER_BT_WORD - (__builtin_clz(backtrace_val) / 2);

        for (int d=0; d<steps; d++) {
            if (!extending) {
                wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
                for (int j=0; j<acc; j++) {
                    *cigar_ptr = 'M';
                    cigar_ptr++;
                }

                offset += acc;
            }

            affine_op_t op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

            switch (op) {
                // k + 1
                case OP_DEL:
                    *cigar_ptr = 'D';
                    extending = true;
                    k--;
                    cigar_ptr++;
                    break;
                // k
                case OP_SUB:
                    if (extending) {
                        extending = false;
                    } else {
                        *cigar_ptr = 'X';
                        offset++;
                        cigar_ptr++;
                    }
                    break;
                // k - 1
                case OP_INS:
                    *cigar_ptr = 'I';
                    extending = true;
                    k++;
                    offset++;
                    cigar_ptr++;
                    break;
            }
        }

        if (!extending) {
            // Last exension
            wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
            for (int j=0; j<acc; j++) {
                *cigar_ptr = 'M';
                cigar_ptr++;
            }

            offset += acc;
        }

        iterations--;
    }

    // Final round with the final backtrace word that is not in the offlaoded
    // array
    wfa_bt_vector_t backtrace_val = final_backtrace.backtrace;

    int steps = OPS_PER_BT_WORD - (__builtin_clz(backtrace_val) / 2);

    for (int d=0; d<steps; d++) {
        if (!extending) {
            wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
            for (int j=0; j<acc; j++) {
                *cigar_ptr = 'M';
                cigar_ptr++;
            }

            offset += acc;
        }

        affine_op_t op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

        switch (op) {
            // k + 1
            case OP_DEL:
                *cigar_ptr = 'D';
                extending = true;
                k--;
                cigar_ptr++;
                break;
            // k
            case OP_SUB:
                if (extending) {
                    extending = false;
                } else {
                    *cigar_ptr = 'X';
                    offset++;
                    cigar_ptr++;
                }
                break;
            // k - 1
            case OP_INS:
                *cigar_ptr = 'I';
                extending = true;
                k++;
                offset++;
                cigar_ptr++;
                break;
        }
    }

    if (!extending) {
        // Last exension
        wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
        for (int j=0; j<acc; j++) {
            *cigar_ptr = 'M';
            cigar_ptr++;
        }

        offset += acc;
    }

    return cigar_ascii;
}
