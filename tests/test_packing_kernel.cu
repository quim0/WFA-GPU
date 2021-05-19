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
#include "tests/test.h"

SET_TEST_NAME("PACKING KERNEL")

const char UNPACK[4] = {'A', 'C', 'T', 'G'};

bool verify_packed_sequence (char* sequence_packed,
                        char* sequence_unpacked,
                        size_t seq_len) {
    int i;
    for (i=0; i<(seq_len/16); i+=4) {
        uint32_t bases = *(uint32_t*)(sequence_packed + i);


        char base0_recovered = UNPACK[(bases >> 30) & 3];
        char base1_recovered = UNPACK[(bases >> 28) & 3];
        char base2_recovered = UNPACK[(bases >> 26) & 3];
        char base3_recovered = UNPACK[(bases >> 24) & 3];


        char base4_recovered = UNPACK[(bases >> 22) & 3];
        char base5_recovered = UNPACK[(bases >> 20) & 3];
        char base6_recovered = UNPACK[(bases >> 18) & 3];
        char base7_recovered = UNPACK[(bases >> 16) & 3];


        char base8_recovered =  UNPACK[(bases >> 14) & 3];
        char base9_recovered =  UNPACK[(bases >> 12) & 3];
        char base10_recovered = UNPACK[(bases >> 10) & 3];
        char base11_recovered = UNPACK[(bases >> 8)  & 3];

        char base12_recovered = UNPACK[(bases >> 6) & 3];
        char base13_recovered = UNPACK[(bases >> 4) & 3];
        char base14_recovered = UNPACK[(bases >> 2) & 3];
        char base15_recovered = UNPACK[bases        & 3];

        char base0 = sequence_unpacked[i*4];
        char base1 = sequence_unpacked[i*4 + 1];
        char base2 = sequence_unpacked[i*4 + 2];
        char base3 = sequence_unpacked[i*4 + 3];
        char base4 = sequence_unpacked[i*4 + 4];
        char base5 = sequence_unpacked[i*4 + 5];
        char base6 = sequence_unpacked[i*4 + 6];
        char base7 = sequence_unpacked[i*4 + 7];
        char base8 = sequence_unpacked[i*4 + 8];
        char base9 = sequence_unpacked[i*4 + 9];
        char base10 = sequence_unpacked[i*4 + 10];
        char base11 = sequence_unpacked[i*4 + 11];
        char base12 = sequence_unpacked[i*4 + 12];
        char base13 = sequence_unpacked[i*4 + 13];
        char base14 = sequence_unpacked[i*4 + 14];
        char base15 = sequence_unpacked[i*4 + 15];

        TEST_ASSERT(base0_recovered == base0)
        TEST_ASSERT(base1_recovered == base1)
        TEST_ASSERT(base2_recovered == base2)
        TEST_ASSERT(base3_recovered == base3)
        TEST_ASSERT(base4_recovered == base4)
        TEST_ASSERT(base5_recovered == base5)
        TEST_ASSERT(base6_recovered == base6)
        TEST_ASSERT(base7_recovered == base7)
        TEST_ASSERT(base8_recovered == base8)
        TEST_ASSERT(base9_recovered == base9)
        TEST_ASSERT(base10_recovered == base10)
        TEST_ASSERT(base11_recovered == base11)
        TEST_ASSERT(base12_recovered == base12)
        TEST_ASSERT(base13_recovered == base13)
        TEST_ASSERT(base14_recovered == base14)
        TEST_ASSERT(base15_recovered == base15)
    }

    i += 4;
    int remaining_bases = seq_len % 16;
    if (remaining_bases) {
        uint32_t bases = *(uint32_t*)(sequence_packed + i);

        char base0_recovered = UNPACK[(bases >> 30) & 3];
        char base1_recovered = UNPACK[(bases >> 28) & 3];
        char base2_recovered = UNPACK[(bases >> 26) & 3];
        char base3_recovered = UNPACK[(bases >> 24) & 3];

        char base4_recovered = UNPACK[(bases >> 22) & 3];
        char base5_recovered = UNPACK[(bases >> 20) & 3];
        char base6_recovered = UNPACK[(bases >> 18) & 3];
        char base7_recovered = UNPACK[(bases >> 16) & 3];

        char base8_recovered =  UNPACK[(bases >> 14) & 3];
        char base9_recovered =  UNPACK[(bases >> 12) & 3];
        char base10_recovered = UNPACK[(bases >> 10) & 3];
        char base11_recovered = UNPACK[(bases >> 8)  & 3];

        char base12_recovered = UNPACK[(bases >> 6) & 3];
        char base13_recovered = UNPACK[(bases >> 4) & 3];
        char base14_recovered = UNPACK[(bases >> 2) & 3];
        char base15_recovered = UNPACK[bases        & 3];

        char base0 = sequence_unpacked[i*4];
        char base1 = sequence_unpacked[i*4 + 1];
        char base2 = sequence_unpacked[i*4 + 2];
        char base3 = sequence_unpacked[i*4 + 3];
        char base4 = sequence_unpacked[i*4 + 4];
        char base5 = sequence_unpacked[i*4 + 5];
        char base6 = sequence_unpacked[i*4 + 6];
        char base7 = sequence_unpacked[i*4 + 7];
        char base8 = sequence_unpacked[i*4 + 8];
        char base9 = sequence_unpacked[i*4 + 9];
        char base10 = sequence_unpacked[i*4 + 10];
        char base11 = sequence_unpacked[i*4 + 11];
        char base12 = sequence_unpacked[i*4 + 12];
        char base13 = sequence_unpacked[i*4 + 13];
        char base14 = sequence_unpacked[i*4 + 14];
        char base15 = sequence_unpacked[i*4 + 15];

        TEST_ASSERT(base0_recovered == base0)
        TEST_ASSERT(base1_recovered == base1)
        TEST_ASSERT(base2_recovered == base2)
        TEST_ASSERT(base3_recovered == base3)
        TEST_ASSERT(base4_recovered == base4)
        TEST_ASSERT(base5_recovered == base5)
        TEST_ASSERT(base6_recovered == base6)
        TEST_ASSERT(base7_recovered == base7)
        TEST_ASSERT(base8_recovered == base8)
        TEST_ASSERT(base9_recovered == base9)
        TEST_ASSERT(base10_recovered == base10)
        TEST_ASSERT(base11_recovered == base11)
        TEST_ASSERT(base12_recovered == base12)
        TEST_ASSERT(base13_recovered == base13)
        TEST_ASSERT(base14_recovered == base14)
        TEST_ASSERT(base15_recovered == base15)

        switch (remaining_bases) {
            case 15:
                TEST_ASSERT(base14_recovered == base14)
            case 14:
                TEST_ASSERT(base13_recovered == base13)
            case 13:
                TEST_ASSERT(base12_recovered == base12)
            case 12:
                TEST_ASSERT(base11_recovered == base11)
            case 11:
                TEST_ASSERT(base10_recovered == base10)
            case 10:
                TEST_ASSERT(base9_recovered == base9)
            case 9:
                TEST_ASSERT(base8_recovered == base8)
            case 8:
                TEST_ASSERT(base7_recovered == base7)
            case 7:
                TEST_ASSERT(base6_recovered == base6)
            case 6:
                TEST_ASSERT(base5_recovered == base5)
            case 5:
                TEST_ASSERT(base4_recovered == base4)
            case 4:
                TEST_ASSERT(base3_recovered == base3)
            case 3:
                TEST_ASSERT(base2_recovered == base2)
            case 2:
                TEST_ASSERT(base1_recovered == base1)
            case 1:
                TEST_ASSERT(base0_recovered == base0)
        }
    }
    return true;
}

bool test_packed_sequences (char* seqs_unpacked,
                         char* seqs_packed,
                         size_t seqs_unpacked_size,
                         size_t seqs_packed_size,
                         sequence_pair_t* seqs_metadata,
                         size_t num_alignments) {
   if (seqs_unpacked_size % 4) {
       TEST_FAIL("Unpacked buffer size is not 32 bits aligned.")
   }

   if (seqs_packed_size % 4) {
       TEST_FAIL("Packed buffer size is not 32 bits aligned.")
   }


   for (int al_idx=0; al_idx < num_alignments; al_idx++) {
       sequence_pair_t curr_alignment = seqs_metadata[al_idx];

       size_t text_len = curr_alignment.text_len;
       size_t pattern_len = curr_alignment.pattern_len;

       verify_packed_sequence(seqs_packed + curr_alignment.text_offset_packed,
                         seqs_unpacked + curr_alignment.text_offset,
                         text_len);

       verify_packed_sequence(seqs_packed + curr_alignment.pattern_offset_packed,
                         seqs_unpacked + curr_alignment.pattern_offset,
                         pattern_len);
   }
   return true;
}

void test_one_sequence () {
    // One sequnce test
    size_t seq_buf_size = 1024;
    char* sequence_unpacked = (char*)calloc(seq_buf_size, 1);
    sequence_pair_t* sequence_metadata = (sequence_pair_t*)calloc(1, sizeof(sequence_pair_t));
    if (!sequence_unpacked || !sequence_metadata) {
        LOG_ERROR("Can not allocate memory");
        exit(-1);
    }

    sequence_metadata[0].pattern_offset = 0;
    sequence_metadata[0].pattern_len = 300;
    strcpy(sequence_unpacked, "AGGATTGGGTGTAACAGCAACTGCTAAGGAATGGACGTAAGGAGGCTCGACATAGTCTCATGTGCTTAGACAGCTATGCGTTGGAAGCAACTGCCAACATCCAATTCGGGACGTATTGCATGTCACGATGAAAACTAGCGCGATCCTCAACTTCCTGTACGAGGCTGCCAAGGCGAGCCGGTGCCTATGACACATGACTGCACACAAAGCTACCCGACGTGAATATGCCGTGGCGACTATATTGAAACGTCATCAACGCGACCTGATATTTATTGTATCGTGTGATTCCAAGGGCCGACG");

    sequence_metadata[0].text_offset = 304;
    sequence_metadata[0].text_len = 299;
    strcpy(sequence_unpacked + sequence_metadata[0].text_offset, "AGGATTGGGTGTAACAGCAACTGCTAAGGAATGGACGTAAGGAGGCTGACATAGTCGCATGTGCTTAGACCAGCTATGCGCTGGACTGAAACTGCCAACATCCAATTCGGGACGTATTGCATGTCACGATGAAACACTAGCGCGATCCTAACTTCCTGTAGGAGGCTGCCAAGCGAGCGGTGCCTATGACACATGACTGCATCACAAAGCTACCCGACGTGAATATGCCGTGGCGACTATATTGAAACGTCATTCAACGCGACCTGATATTACTGTATGGTGTGATTCCAAGGGCCGAC");
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

    TEST_ASSERT(d_seq_buf_packed != NULL)
    TEST_ASSERT(d_seq_buf_unpacked != NULL)
    TEST_ASSERT(d_seq_buf_packed_size > 0)
    TEST_ASSERT(d_seq_metadata != NULL)

    pack_sequences_gpu_async(
        d_seq_buf_unpacked,
        d_seq_buf_packed,
        seq_buf_size,
        d_seq_buf_packed_size,
        d_seq_metadata,
        num_alignments,
        0
    );

    char* host_packed_buf = (char*)calloc(d_seq_buf_packed_size, 1);
    if (!host_packed_buf) {
        LOG_ERROR("Can not allocate memory.")
        exit(-1);
    }

    cudaDeviceSynchronize();
    CUDA_TEST_CHECK_ERR

    cudaMemcpy(
        host_packed_buf,
        d_seq_buf_packed,
        d_seq_buf_packed_size,
        cudaMemcpyDeviceToHost
    );

    CUDA_TEST_CHECK_ERR

    cudaDeviceSynchronize();
    CUDA_TEST_CHECK_ERR

    test_packed_sequences(
        sequence_unpacked,
        host_packed_buf,
        seq_buf_size,
        d_seq_buf_packed_size,
        sequence_metadata,
        num_alignments
    );

    free(sequence_unpacked);
    free(sequence_metadata);
    free(host_packed_buf);
}

void test_multiple_sequences () {
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

    TEST_ASSERT(d_seq_buf_packed != NULL)
    TEST_ASSERT(d_seq_buf_unpacked != NULL)
    TEST_ASSERT(d_seq_buf_packed_size > 0)
    TEST_ASSERT(d_seq_metadata != NULL)

    pack_sequences_gpu_async(
        d_seq_buf_unpacked,
        d_seq_buf_packed,
        seq_buf_size,
        d_seq_buf_packed_size,
        d_seq_metadata,
        num_alignments,
        0
    );

    char* host_packed_buf = (char*)calloc(d_seq_buf_packed_size, 1);
    if (!host_packed_buf) {
        LOG_ERROR("Can not allocate memory.")
        exit(-1);
    }

    cudaDeviceSynchronize();
    CUDA_TEST_CHECK_ERR

    cudaMemcpy(
        host_packed_buf,
        d_seq_buf_packed,
        d_seq_buf_packed_size,
        cudaMemcpyDeviceToHost
    );

    CUDA_TEST_CHECK_ERR

    cudaDeviceSynchronize();
    CUDA_TEST_CHECK_ERR

    test_packed_sequences(
        sequence_unpacked,
        host_packed_buf,
        seq_buf_size,
        d_seq_buf_packed_size,
        sequence_metadata,
        num_alignments
    );

    free(sequence_unpacked);
    free(sequence_metadata);
    free(host_packed_buf);
}

int main () {

    test_one_sequence();
    test_multiple_sequences();

    TEST_OK
    return 0;
}
