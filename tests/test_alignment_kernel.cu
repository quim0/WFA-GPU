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

#include <stdint.h>

#include "utils/logger.h"
#include "utils/sequences.h"
#include "sequence_packing.cuh"
#include "sequence_alignment.cuh"
#include "wfa_types.h"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "tests/test.h"

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)

#define OPS_PER_BT_WORD 8

SET_TEST_NAME("ALIGNMENT KERNEL")

__host__ bool check_cigar_edit (const char* text,
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
					printf("Alignment not matching at CCIGAR index %d"
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
					printf("Alignment not mismatching at CCIGAR index %d"
						  " (pattern[%d] = %c == text[%d] = %c)\n",
						  i, pattern_pos, pattern[pattern_pos],
						  text_pos, text[text_pos]);
					return false;
				}
				++pattern_pos;
				++text_pos;
				break;
			default:
				TEST_FAIL("Invalid CIGAR generated.\n");
				break;
		}
	}

	if (pattern_pos != plen) {
		printf("Alignment incorrect length, pattern-aligned: %d, "
			  "pattern-length: %d.\n", pattern_pos, plen);
		return false;
	}

	if (text_pos != tlen) {
		printf("Alignment incorrect length, text-aligned: %d, "
			  "text-length: %d\n", text_pos, tlen);
		return false;
	}

	return true;
}

__host__ bool check_affine_distance (const char* text,
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
                TEST_ASSERT(false);
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

char* recover_cigar (const char* text,
                     const char* pattern,
                     const size_t tlen,
                     const size_t plen,
                     wfa_backtrace_t final_backtrace,
                     wfa_backtrace_t* offloaded_backtraces_array) {
    char* cigar_ascii = (char*)calloc(tlen + plen, 1);
    char* cigar_ptr = cigar_ascii;

    // TODO: Reverse linked list instead of doing this
    // Max possible distance / 4 as there are 4 ops per byte
    const int max_words = (tlen + plen) / 4;
    uint16_t* bt_indexes = (uint16_t*)calloc(max_words, sizeof(uint16_t));

    wfa_backtrace_t curr_bt = final_backtrace;
    uint16_t* curr_bt_index = bt_indexes;
    while (curr_bt.prev != 0) {
        *curr_bt_index++ = curr_bt.prev;
        curr_bt = offloaded_backtraces_array[curr_bt.prev];
    }

    int k=0;
    wfa_offset_t offset = 0;
    bool extending = false;
    while (curr_bt_index-- != bt_indexes) {

        wfa_backtrace_t backtrace = offloaded_backtraces_array[*curr_bt_index];
        uint16_t backtrace_val = backtrace.backtrace;

        // Substract 16 to builtin_clz because the function gets a 32 bits
        // value, and we pass a 16 bit value (that gets automatically converted
        // to 32 bit)
        int steps = OPS_PER_BT_WORD - ((__builtin_clz(backtrace_val) - 16) / 2);

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

    }

    // Final round with last backtrace
    wfa_backtrace_t backtrace = final_backtrace;
    uint16_t backtrace_val = backtrace.backtrace;

    int steps = OPS_PER_BT_WORD - ((__builtin_clz(backtrace_val) - 16) / 2);

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
    }

    free(bt_indexes);

    return cigar_ascii;
}

void test_one_alignment() {
    // One sequence test
    size_t seq_buf_size = 32;
    char* sequence_unpacked = (char*)calloc(seq_buf_size, 1);
    sequence_pair_t* sequence_metadata = (sequence_pair_t*)calloc(1, sizeof(sequence_pair_t));
    if (!sequence_unpacked || !sequence_metadata) {
        LOG_ERROR("Can not allocate memory");
        exit(-1);
    }

    sequence_metadata[0].pattern_offset = 0;
    sequence_metadata[0].pattern_len = 7;
    strcpy(sequence_unpacked, "GATTACA");

    sequence_metadata[0].text_offset = 12;
    sequence_metadata[0].text_len = 5;
    strcpy(sequence_unpacked + sequence_metadata[0].text_offset, "GAATA");
    size_t num_alignments = 1;

    char* d_seq_buf_unpacked = NULL;
    char* d_seq_buf_packed = NULL;
    size_t d_seq_buf_packed_size = 0;
    sequence_pair_t* d_seq_metadata = NULL;

    prepare_pack_sequences_gpu_async(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        &d_seq_buf_unpacked,
        &d_seq_buf_packed,
        &d_seq_buf_packed_size,
        &d_seq_metadata,
        0
    );

    pack_sequences_gpu_async(
        d_seq_buf_unpacked,
        d_seq_buf_packed,
        seq_buf_size,
        d_seq_buf_packed_size,
        d_seq_metadata,
        num_alignments,
        0
    );


    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    //// Only one sequence in this test
    alignment_result_t results = {0};

    // TODO: Move max steps outside launch_alignments_async function
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    BT_OFFLOADED_ELEMENTS(256),
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 7)

    penalties = {.x = 1, .o = 0, .e = 1};

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 3)

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);
    free(backtraces);

    cudaDeviceSynchronize();
}

void test_multiple_alignments_affine () {
    size_t seq_buf_size = 6144;
    char* sequence_unpacked = (char*)calloc(seq_buf_size, 1);
    sequence_pair_t* sequence_metadata = (sequence_pair_t*)calloc(4, sizeof(sequence_pair_t));
    if (!sequence_unpacked || !sequence_metadata) {
        LOG_ERROR("Can not allocate memory");
        exit(-1);
    }

    sequence_metadata[0].pattern_offset = 0;
    sequence_metadata[0].pattern_len = 150;
    strcpy(sequence_unpacked, "TGTGAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG");

    sequence_metadata[0].text_offset = 152;
    sequence_metadata[0].text_len = 151;
    strcpy(sequence_unpacked + sequence_metadata[0].text_offset, "TGTAAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG");


    sequence_metadata[1].pattern_offset = 308;
    sequence_metadata[1].pattern_len = 150;
    strcpy(sequence_unpacked + sequence_metadata[1].pattern_offset, "ACGGGCGTGCATCACAACCCGTGATGATCGCCATAGAGCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGTCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA");

    sequence_metadata[1].text_offset = 460;
    sequence_metadata[1].text_len = 151;
    strcpy(sequence_unpacked + sequence_metadata[1].text_offset, "ACGGGCGTGCATCACAACCCGGATGATCGCCATAGAGCCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGATCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA");

    sequence_metadata[2].pattern_offset = 616;
    sequence_metadata[2].pattern_len = 13;
    strcpy(sequence_unpacked + sequence_metadata[2].pattern_offset,
    "TTTTGGAGGAAAA");

    sequence_metadata[2].text_offset = 1620;
    sequence_metadata[2].text_len = 8;
    strcpy(sequence_unpacked + sequence_metadata[2].text_offset,
    "TTTTAAAA");

    sequence_metadata[3].pattern_offset = 1632;
    sequence_metadata[3].pattern_len = 300;
    strcpy(sequence_unpacked + sequence_metadata[3].pattern_offset,
    "TACAAATGTACACGGCAATGAGCTATCCAACAATAATTTTACAGTTTTTGGAATACGGTTGATGTTTTTGGAAGGTCCTCACGCAGTTAGGGTGCGCCGCAAGATCTCTTGAAACATAATTGGGAACGGTAGTTGTAGAACGAGTGGGGGGGCCAGGCAAACGAACTCAACCGCTGTGCGCAAGGAAAGCATGTTTATAATCGGTCCGATCCTCACGCCCTGAGCACCTGGTTAGTGACGTGAGACATGGACCATGACAATGATGTGCTATGTACTCGTTATCCACACGACGTGCGCTTC");

    sequence_metadata[3].text_offset = 1936;
    sequence_metadata[3].text_len = 301;
    strcpy(sequence_unpacked + sequence_metadata[3].text_offset,
    "TACAAATGTACACGGCAATGAGCTATCCAACAATAATTTTACAGTTTTTGGAATATCGGTTGATGTTTTTGGAGTGTCCTCACGCAGTTAGGGTGCGCCGCAAGATCTCTTGAAACATAGTTGGGAACGGTAGTTGTAGACGAGGGGGGGGCCAGGCAAACGATCTCACCGCGTGCGCAAGGAAAGCATGGTTTATAATCGGTCCCGATCCTCACGCCCTGAGCACCTGTTAGTGACGTTGAGATCATGGACCATGACAATGATGTGCTACTGTACTCGTTATCCACACGACGTGCGCTTC");
    size_t num_alignments = 4;

    char* d_seq_buf_unpacked = NULL;
    char* d_seq_buf_packed = NULL;
    size_t d_seq_buf_packed_size = 0;
    sequence_pair_t* d_seq_metadata = NULL;

    prepare_pack_sequences_gpu_async(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        &d_seq_buf_unpacked,
        &d_seq_buf_packed,
        &d_seq_buf_packed_size,
        &d_seq_metadata,
        0
    );

    pack_sequences_gpu_async(
        d_seq_buf_unpacked,
        d_seq_buf_packed,
        seq_buf_size,
        d_seq_buf_packed_size,
        d_seq_metadata,
        num_alignments,
        0
    );

    cudaDeviceSynchronize();

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    alignment_result_t* results = (alignment_result_t*)calloc(num_alignments,
                                                              sizeof(alignment_result_t));

    // TODO: Move max steps outside launch_alignments_async function
    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_ELEMENTS(256);
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    backtraces_offloaded_elements * num_alignments,
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        results,
        backtraces
    );

    cudaDeviceSynchronize();

    const int correct_results[4] = {6, 12, 8, 52};
    for (int i=0; i<num_alignments; i++) {
        // TODO
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace,
                                    backtraces + backtraces_offloaded_elements*i);

        bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
        TEST_ASSERT(correct_cigar)
        bool correct_affine_d = check_affine_distance(text, pattern, tlen, plen,
                                                      distance, penalties,
                                                      cigar);
        TEST_ASSERT(correct_affine_d)
        TEST_ASSERT(distance == correct_results[i])
    }

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);
    free(results);
    free(backtraces);

    cudaDeviceSynchronize();
}

void test_multiple_alignments_edit () {
    size_t seq_buf_size = 4096;
    char* sequence_unpacked = (char*)calloc(seq_buf_size, 1);
    sequence_pair_t* sequence_metadata = (sequence_pair_t*)calloc(3, sizeof(sequence_pair_t));
    if (!sequence_unpacked || !sequence_metadata) {
        LOG_ERROR("Can not allocate memory");
        exit(-1);
    }

    sequence_metadata[0].pattern_offset = 0;
    sequence_metadata[0].pattern_len = 150;
    strcpy(sequence_unpacked,
    "AAGAGCAACCACTGGCGCTAGGGGTTTGTTTATCCCTCGAGGGGCCTTGTAACGTCCTACGTGCCTTAACCTATGCCGCTCCATTTACTCTCACTCCGGGAACATAGCGTAAACTACACACCCCGATATCGAGTACATGGGTGGCGTGGC");

    sequence_metadata[0].text_offset = 152;
    sequence_metadata[0].text_len = 152;
    strcpy(sequence_unpacked + sequence_metadata[0].text_offset,
    "AAGAGGCACCACTGGCGCTAGGGGTTTGTTTATCCCTCGAGGGGCCTTGTAACGTCCATGTGCCTTAACCCGTATCCCGCTCCATTTACTCTCACTCCGGGAACATATGCGTAAACTACACACCCCGATATCGAGTACATGGGTGGCGTGGC");

    sequence_metadata[1].pattern_offset = 308;
    sequence_metadata[1].pattern_len = 150;
    strcpy(sequence_unpacked + sequence_metadata[1].pattern_offset,
    "GCGATCCCCAGCCACGTTCGTCTTTCCGATATCTAAAGGGGCTAGATCTATTGTGCAATCTACATCCATAGGCGTTGGAGGAGCTAGAAGGAGTCGAGGTGCGATCTGTAAGCGTATGCTCTATGCCTAAGGCCTCGGTGTTCAGACCTT");

    sequence_metadata[1].text_offset = 460;
    sequence_metadata[1].text_len = 150;
    strcpy(sequence_unpacked + sequence_metadata[1].text_offset,
    "GCATCCCCAGCCACCGTCAGTCTTTCCGATACTAAGGCGCTAGATCTATTGTGGAATCTACATCCATAGGACGTTGGAGGAGCTAGAAGGAGTCGAGGTGCGATCTGTAAAGCGTATGCTCTATGCTGAGGCCTCGCTGTTCAGAGCCTT");

    sequence_metadata[2].pattern_offset = 616;
    sequence_metadata[2].pattern_len = 1000;
    strcpy(sequence_unpacked + sequence_metadata[2].pattern_offset,
    "GAACAAAGGGTAAATACCCCAAGTCACTGCCCGGGGGTCCCACGCCTGGATTCGGGTACGGTTAGGTCGAGACAGTCCCACCTGAGATTGGCGGCACACTCATTACGGTAGACGTGCAGCTAGCGTGTAAGGACCATATGACTGATAGAGTTTCCCCACGAAAAGGCCTAATCAGGATCAGATCGCTACCGCCTCTGGCCTCCCGATGTCGGCGTAATCCAATGTCCGAGATACAGGTCCAAAGGTTGTAAAATGAATTAGTTGCTCTTGCAGCTCCTAATAAAATCATACCTCTAACTATCCGGATTTTATAATAAACTAGAAAAAACACCCGATTTTGTTGTACAAGCTTAATAAACGATAGCAGAACTTACTCTTCCCCCCCAGGCACCTCAACGCCACCATAGGATACACTGGGCTGGCCTGGCAATACACACTTCTTTACTTACAATGGCCATTCTGCCTTGATCGTCCTGAGTGGTCACTGGTGTGAAATAGAAAGTCTGAACTGTTAACTTTGCGCGTGGTAGTCATGACTATGGGGTTTGTGCCAGTTAGTCATGCGCCAAGTTCGCAGATCTATTTGGAAGGCCAGTGATTGTGTCATTCGCATATGTGGAAACCCACAACTGACGGGCCTATTTTGGTCCTCCATTAATCCGAGAGAGACCACAACTTAAATGCACCCCAGTTGCAACGCTACACGCACCCGCTACACGGGACCTGCAAACTATAGCCTTTACAACCCTTTCACTTACTACCTGAACGCACCATCCCTGATGGTTCTTGTTAATTCTATCCAGGAATCACGTAATTGTGATGCTGCACGATTCGCCGCTGTCGTGCGACCCAATTGAAATCTGGCATAGTTCACCTTTAACCACCAAGCACGTAACCTCTGCCTGGTCGTCTCGCGGCCTCGCCTGTACAAACCAATAGCTACCGTAAACAGTGATATTGATTGAAGAAAGTCACTTCAAGAGGTTCTGCGGACACCTGC");

    sequence_metadata[2].text_offset = 1620;
    sequence_metadata[2].text_len = 992;
    strcpy(sequence_unpacked + sequence_metadata[2].text_offset,
    "GAACAAAGGGTAAATACCCCAAGATCACTACCCGGGGGTCCCACGCCTGGATTCGGGTACGGTTAGGTCGAGACAGTCCCACCTGAGATGGCGGCACACTGTTACGGAAGACAGTGCAGCTAGCGTGTAAGGACCCATACGCTGATAGAGTTTCCCCACGCAAAGGCCTAATGAGATTAGATCGCTACCCACTCTGGCCTCCCCAATGTACGGCGTAATCCAATCCGAGATACAGGTCCATAGGTTGTAAAATGATTATAGTTGCTCTTGCAGTCTTAATAAAAATCATACCCTAACTTCCGGATTTATATTAAACTAGAAAAAACACCCCGATTTTGTTGATACAACCTTATTAACGATAGACAGAACTTACATCTTCCCCCCCAGGGACCTCAACGCCACCATAGATACACTGGGCTGGCCTGGCAATAACACTTCTTTACATTACAAGGCCATTCTGCCTTGATCGTCCTGAGTGGTCACTGGTGTGAAAATAGAAAGTCTGAACTGTAACTTTGCGCGTGGTAGCATGACTATGGGGTTTGTGCCAGGAAGTCATGCGCCAAGTTCGCAGATCTATTTGGAAGGCCATTGATTGTTCTGCGCATATGTGAAATCCAAACTGAGGGCCTATTTTGGTCCCCATTAATCCGACGAGAGACCCGACAACTTAAATGCACCCCAGTTGCAAGGCTACACGCACCCGCTACACGGGACCTGCAAACTATAGCCGTTACAACCCTTTCACTTACTCCTGAACGCACCATACCTGATGGTTCTTGTTAATTCTATCCAGGGAATCACGTAATTGTGCGCGCGCACGATTCGCCGCTGTCGTGCGACCCAATGAAATCTGGCATAGTTGACCTTTAACCACCAAAGCCCGTAACCTCTGCCTGTTTGTTCGCTGCCTCGCCTGTACAAACCAATAGCTACCGTAAACAGTGATATTGATGAAGAAGTTACTTCAAGACGTTCATCCGGACACCTGC");
    size_t num_alignments = 3;

    char* d_seq_buf_unpacked = NULL;
    char* d_seq_buf_packed = NULL;
    size_t d_seq_buf_packed_size = 0;
    sequence_pair_t* d_seq_metadata = NULL;

    prepare_pack_sequences_gpu_async(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        &d_seq_buf_unpacked,
        &d_seq_buf_packed,
        &d_seq_buf_packed_size,
        &d_seq_metadata,
        0
    );

    pack_sequences_gpu_async(
        d_seq_buf_unpacked,
        d_seq_buf_packed,
        seq_buf_size,
        d_seq_buf_packed_size,
        d_seq_metadata,
        num_alignments,
        0
    );

    cudaDeviceSynchronize();

    affine_penalties_t penalties = {.x = 1, .o = 0, .e = 1};
    alignment_result_t* results = (alignment_result_t*)calloc(num_alignments,
                                                              sizeof(alignment_result_t));

    // TODO: Move max steps outside launch_alignments_async function
    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_ELEMENTS(256);
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    backtraces_offloaded_elements * num_alignments,
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        results,
        backtraces
    );

    cudaDeviceSynchronize();

    const int correct_results[3] = {8, 14, 83};

    for (int i=0; i<num_alignments; i++) {
        // TODO
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace,
                                    backtraces + backtraces_offloaded_elements*i);

        bool correct = check_cigar_edit(text, pattern, tlen, plen, cigar);
        TEST_ASSERT(correct)
        TEST_ASSERT(distance == correct_results[i])
    }

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);
    free(results);
    free(backtraces);

    cudaDeviceSynchronize();
}

int main () {

    test_one_alignment();
    test_multiple_alignments_edit();
    test_multiple_alignments_affine();

    TEST_OK
    return 0;
}
