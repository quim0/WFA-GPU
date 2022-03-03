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
#include "utils/verification.cuh"
#include "batch_async.cuh"
#include "wfa_types.h"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "tests/test.h"

#define MAX_STEPS 512
#define THREADS_PER_BLOCK 128


SET_TEST_NAME("ALIGNMENT KERNEL")

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

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    // Only one sequence in this test
    alignment_result_t results = {0};

    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    BT_OFFLOADED_RESULT_ELEMENTS(MAX_STEPS),
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0, // Band
        false // check correctness
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 7)

    penalties = {.x = 1, .o = 0, .e = 1};

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0,
        false // check correctness
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 3)

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

    cudaDeviceSynchronize();

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    alignment_result_t* results = (alignment_result_t*)calloc(num_alignments,
                                                              sizeof(alignment_result_t));

    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(MAX_STEPS);
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    backtraces_offloaded_elements * num_alignments,
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0, // Band
        false // check correctness
    );

    cudaDeviceSynchronize();

    const int correct_results[4] = {6, 12, 8, 52};
    for (int i=0; i<num_alignments; i++) {
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace,
                                    backtraces + backtraces_offloaded_elements*i,
                                    results[i]);

        bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
        TEST_ASSERT(correct_cigar)
        bool correct_affine_d = check_affine_distance(text, pattern, tlen, plen,
                                                      distance, penalties,
                                                      cigar);
        TEST_ASSERT(correct_affine_d)
        TEST_ASSERT(distance == correct_results[i])
        free(cigar);
    }

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

    cudaDeviceSynchronize();

    affine_penalties_t penalties = {.x = 1, .o = 0, .e = 1};
    alignment_result_t* results = (alignment_result_t*)calloc(num_alignments,
                                                              sizeof(alignment_result_t));

    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(MAX_STEPS);
    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    backtraces_offloaded_elements * num_alignments,
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0, // Band
        false // check correctness
    );

    cudaDeviceSynchronize();

    const int correct_results[3] = {8, 14, 83};

    for (int i=0; i<num_alignments; i++) {
        char* text = &sequence_unpacked[sequence_metadata[i].text_offset];
        char* pattern = &sequence_unpacked[sequence_metadata[i].pattern_offset];
        size_t tlen = sequence_metadata[i].text_len;
        size_t plen = sequence_metadata[i].pattern_len;
        int distance = results[i].distance;
        char* cigar = recover_cigar(text, pattern, tlen,
                                    plen,results[i].backtrace,
                                    backtraces + backtraces_offloaded_elements*i,
                                    results[i]);

        bool correct = check_cigar_edit(text, pattern, tlen, plen, cigar);
        TEST_ASSERT(correct)
        TEST_ASSERT(distance == correct_results[i])
    }

    free(sequence_unpacked);
    free(sequence_metadata);
    free(results);
    free(backtraces);

    cudaDeviceSynchronize();
}

void test_distance_zero() {
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
    sequence_metadata[0].text_len = 7;
    strcpy(sequence_unpacked + sequence_metadata[0].text_offset, "GATTACA");
    size_t num_alignments = 1;

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    // Only one sequence in this test
    alignment_result_t results = {0};

    wfa_backtrace_t* backtraces = (wfa_backtrace_t*)calloc(
                                                    BT_OFFLOADED_RESULT_ELEMENTS(MAX_STEPS),
                                                    sizeof(wfa_backtrace_t)
                                                    );

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0, // Band
        false // check correctness
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 0)

    penalties = {.x = 1, .o = 0, .e = 1};

    launch_alignments_batched(
        sequence_unpacked,
        seq_buf_size,
        sequence_metadata,
        num_alignments,
        penalties,
        &results,
        backtraces,
        MAX_STEPS,
        THREADS_PER_BLOCK,
        3, // Num blocks, make it bigger than 1 to test
        num_alignments, // Batch size
        0, // Band
        false // check correctness
    );

    cudaDeviceSynchronize();

    TEST_ASSERT(results.distance == 0)

    free(sequence_unpacked);
    free(sequence_metadata);
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
