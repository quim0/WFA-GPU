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

SET_TEST_NAME("ALIGNMENT KERNEL")

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
                     wfa_backtrace_t backtrace) {

    char* cigar_ascii = (char*)calloc(tlen + plen, 1);
    char* cigar_ptr = cigar_ascii;

    int steps = 16 - (__builtin_clz(backtrace) / 2);

    int k=0;
    wfa_offset_t offset = 0;
    for (int d=0; d<steps; d++) {
        wfa_offset_t acc = extend_wavefront(offset, k, pattern, plen, text, tlen);
        for (int j=0; j<acc; j++) {
            *cigar_ptr = 'M';
            cigar_ptr++;
        }

        offset += acc;

        affine_op_t op = (affine_op_t)((backtrace >> ((steps - d - 1) * 2)) & 3);

        switch (op) {
            // k + 1
            case OP_DEL:
                *cigar_ptr = 'D';
                k--;
                break;
            // k
            case OP_SUB:
                *cigar_ptr = 'X';
                offset++;
                break;
            // k - 1
            case OP_INS:
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

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        &results
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 7)

    penalties = {.x = 1, .o = 0, .e = 1};

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        &results
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 3)

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);

    cudaDeviceSynchronize();
}

void test_multiple_alignments_affine () {
    // >TGTGAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // <TGTAAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // >ACGGGCGTGCATCACAACCCGTGATGATCGCCATAGAGCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGTCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // <ACGGGCGTGCATCACAACCCGGATGATCGCCATAGAGCCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGATCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // >ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG
    // <ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG

    size_t seq_buf_size = 1024;
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
    sequence_metadata[2].pattern_len = 150;
    strcpy(sequence_unpacked + sequence_metadata[2].pattern_offset, "ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG");

    sequence_metadata[2].text_offset = 768;
    sequence_metadata[2].text_len = 148;
    strcpy(sequence_unpacked + sequence_metadata[2].text_offset, "ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG");
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

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        results
    );

    cudaDeviceSynchronize();

    const int correct_results[3] = {6, 12, 10};
    const char* correct_cigars[3] = {
        "MMMXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
        "MMMMMMMMMMMMMMMMMMMMMDMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
        "MMDMMMXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMDMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"
    };

    for (int i=0; i<num_alignments; i++) {
        // TODO
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace);

        TEST_ASSERT(distance == correct_results[i])
        TEST_ASSERT(!strcmp(cigar, correct_cigars[i]))
    }

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);
    free(results);

    cudaDeviceSynchronize();
}

void test_multiple_alignments_edit () {
    // >TGTGAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // <TGTAAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
    // >ACGGGCGTGCATCACAACCCGTGATGATCGCCATAGAGCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGTCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // <ACGGGCGTGCATCACAACCCGGATGATCGCCATAGAGCCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGATCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
    // >ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG
    // <ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG

    size_t seq_buf_size = 1024;
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
    sequence_metadata[2].pattern_len = 150;
    strcpy(sequence_unpacked + sequence_metadata[2].pattern_offset, "ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG");

    sequence_metadata[2].text_offset = 768;
    sequence_metadata[2].text_len = 148;
    strcpy(sequence_unpacked + sequence_metadata[2].text_offset, "ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG");
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

    launch_alignments_async(
        d_seq_buf_packed,
        d_seq_metadata,
        num_alignments,
        penalties,
        results
    );

    cudaDeviceSynchronize();

    const int correct_results[3] = {2, 3, 3};
    const char* correct_cigars[3] = {
        "MMMXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
        "MMMMMMMMMMMMMMMMMMMMMDMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
        "MMDMMMXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMDMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"
    };

    for (int i=0; i<num_alignments; i++) {
        // TODO
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace);

        TEST_ASSERT(distance == correct_results[i])
        TEST_ASSERT(!strcmp(cigar, correct_cigars[i]))
    }

    cudaFree(d_seq_buf_unpacked);
    cudaFree(d_seq_buf_packed);
    cudaFree(d_seq_metadata);
    free(sequence_unpacked);
    free(sequence_metadata);
    free(results);

    cudaDeviceSynchronize();
}

int main () {

    test_one_alignment();
    test_multiple_alignments_edit();
    test_multiple_alignments_affine();

    TEST_OK
    return 0;
}
