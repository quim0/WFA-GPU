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
        //printf("\n");
    while (curr_bt_index-- != bt_indexes) {

        wfa_backtrace_t backtrace = offloaded_backtraces_array[*curr_bt_index];
        uint16_t backtrace_val = backtrace.backtrace;

        // Substract 16 to builtin_clz because the function gets a 32 bits
        // value, and we pass a 16 bit value (that gets automatically converted
        // to 32 bit)
        int steps = OPS_PER_BT_WORD - ((__builtin_clz(backtrace_val) - 16) / 2);

        for (int d=0; d<steps; d++) {
            wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
            for (int j=0; j<acc; j++) {
                *cigar_ptr = 'M';
                cigar_ptr++;
            }

            offset += acc;

            affine_op_t op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

            switch (op) {
                // k + 1
                case OP_DEL:
                    //printf("D");
                    *cigar_ptr = 'D';
                    k--;
                    break;
                // k
                case OP_SUB:
                    //printf("X");
                    *cigar_ptr = 'X';
                    offset++;
                    break;
                // k - 1
                case OP_INS:
                    //printf("I");
                    *cigar_ptr = 'I';
                    k++;
                    offset++;
                    break;
            }
            cigar_ptr++;
        }

        // Last exension
        wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
        for (int j=0; j<acc; j++) {
            *cigar_ptr = 'M';
            cigar_ptr++;
        }

        offset += acc;

        //printf("   (0x%hx)\n", backtrace_val);
    }

    // Final round with last backtrace
    wfa_backtrace_t backtrace = final_backtrace;
    uint16_t backtrace_val = backtrace.backtrace;

    int steps = OPS_PER_BT_WORD - ((__builtin_clz(backtrace_val) - 16) / 2);

    for (int d=0; d<steps; d++) {
        wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
        for (int j=0; j<acc; j++) {
            *cigar_ptr = 'M';
            cigar_ptr++;
        }

        offset += acc;

        affine_op_t op = (affine_op_t)((backtrace_val >> ((steps - d - 1) * 2)) & 3);

        switch (op) {
            // k + 1
            case OP_DEL:
                //printf("D");
                *cigar_ptr = 'D';
                k--;
                break;
            // k
            case OP_SUB:
                //printf("X");
                *cigar_ptr = 'X';
                offset++;
                break;
            // k - 1
            case OP_INS:
                //printf("I");
                *cigar_ptr = 'I';
                k++;
                offset++;
                break;
        }
        cigar_ptr++;
    }

    //printf("\n");
    // Last exension
    wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
    for (int j=0; j<acc; j++) {
        *cigar_ptr = 'M';
        cigar_ptr++;
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
    // >TGTGAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // <TGTAAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // >ACGGGCGTGCATCACAACCCGTGATGATCGCCATAGAGCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGTCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // <ACGGGCGTGCATCACAACCCGGATGATCGCCATAGAGCCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGATCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // >ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG
    // <ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG

    size_t seq_buf_size = 4096;
    char* sequence_unpacked = (char*)calloc(seq_buf_size, 1);
    sequence_pair_t* sequence_metadata = (sequence_pair_t*)calloc(3, sizeof(sequence_pair_t));
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

    const int correct_results[3] = {6, 12, 8};
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

        if (i == 2) printf("I: %d, score: %d, cigar: %s\n", i, distance, cigar);

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

    //affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
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
