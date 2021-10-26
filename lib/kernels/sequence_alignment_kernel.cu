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

#include <stdbool.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "sequence_alignment_kernel.cuh"

#define MAX_PB(A, B) llmax((A), (B))
#define MAX(A, B) max((A), (B))
#define MIN(A, B) min((A), (B))

// At least one of the highest two bits is set
#define BT_WORD_FULL_CMP 0x4000000000000000ULL
#define BT_IS_FULL(bt_word) ((bt_word) >= BT_WORD_FULL_CMP)

__device__ wfa_offset_t WF_extend_kernel (const char* text,
                                  const char* pattern,
                                  const int tlen,
                                  const int plen,
                                  const int k,
                                  const wfa_offset_t offset_k) {
    int v  = EWAVEFRONT_V(k, offset_k);
    int h  = EWAVEFRONT_H(k, offset_k);

    const int bases_to_cmp = 16;
    int eq_elements = 0;
    int acc = 0;
    // Compare 16 bases at once
    while (v < plen && h < tlen) {
        // Which byte to pick
        int real_v = v / 4;
        int real_h = h / 4;

        // Get the displacement inside the aligned word
        int pattern_displacement = v % bases_to_cmp;
        int text_displacement = h % bases_to_cmp;

        // 0xffffffffffffff00
        uintptr_t alignment_mask = (uintptr_t)-1 << 2;
        uint32_t* word_p_ptr = (uint32_t*)((uintptr_t)(pattern + real_v) & alignment_mask);
        uint32_t* next_word_p_ptr = word_p_ptr + 1;
        uint32_t* word_t_ptr = (uint32_t*)((uintptr_t)(text + real_h) & alignment_mask);
        uint32_t* next_word_t_ptr = word_t_ptr + 1;


        // * 2 because each element is 2 bits
        uint32_t sub_word_p_1 = *word_p_ptr;
        uint32_t sub_word_p_2 = *next_word_p_ptr;
        sub_word_p_1 = sub_word_p_1 << (pattern_displacement * 2);
        // Convert the u32 to big-endian, as little endian inverts the order
        // for the sequences.
        sub_word_p_2 = *next_word_p_ptr;
        // Cast to uint64_t is done to avoid undefined behaviour in case
        // it's shifted by 32 elements.
        // ----
        // The type of the result is that of the promoted left operand. The
        // behavior is undefined if the right operand is negative, or
        // greater than or equal to the length in bits of the promoted left
        // operand.
        // ----
        sub_word_p_2 = ((uint64_t)sub_word_p_2) >>
            ((bases_to_cmp - pattern_displacement) * 2);

        uint32_t sub_word_t_1 = *word_t_ptr;
        sub_word_t_1 = sub_word_t_1 << (text_displacement * 2);
        uint32_t sub_word_t_2 = *next_word_t_ptr;
        sub_word_t_2 = ((uint64_t)sub_word_t_2) >>
            ((bases_to_cmp - text_displacement) * 2);

        uint32_t word_p = sub_word_p_1 | sub_word_p_2;
        uint32_t word_t = sub_word_t_1 | sub_word_t_2;

        uint32_t diff = word_p ^ word_t;
        // Branchless method to remove the equal bits if we read "too far away"
        uint32_t full_mask = (uint32_t)-1;
        int next_v = v + bases_to_cmp;
        int next_h = h + bases_to_cmp;
        uint32_t mask_p = full_mask << ((next_v - plen) * 2 * (next_v > plen));
        uint32_t mask_t = full_mask << ((next_h - tlen) * 2 * (next_h > tlen));
        diff = diff | ~mask_p | ~mask_t;

        int lz = __clz(diff);

        // each element has 2 bits
        eq_elements = lz / 2;
        acc += eq_elements;

        if (eq_elements < bases_to_cmp) {
            break;
        }


        v += bases_to_cmp;
        h += bases_to_cmp;
    }

    if (v > plen || h > tlen) {
        return -10000;
    }

    return offset_k + acc;
}

__device__ uint32_t offload_backtrace (unsigned int* const last_free_bt_position,
                                   const wfa_bt_vector_t backtrace_vector,
                                   const wfa_bt_prev_t backtrace_prev,
                                   wfa_backtrace_t* const global_backtraces_array) {
    uint32_t old_val = atomicAdd(last_free_bt_position, 1);

    global_backtraces_array[old_val].backtrace = backtrace_vector;
    global_backtraces_array[old_val].prev = backtrace_prev;

    // TODO: Check if new_val is more than 32 bits
    return old_val;
}

__device__ void next_M (wfa_wavefront_t* M_wavefronts,
                        wfa_wavefront_t* I_wavefronts,
                        wfa_wavefront_t* D_wavefronts,
                        const int curr_wf,
                        const int active_working_set_size,
                        const int x,
                        const char* text,
                        const char* pattern,
                        const int tlen,
                        const int plen,
                        unsigned int* const last_free_bt_position,
                        wfa_backtrace_t* const offloaded_backtraces) {
    // The wavefront do not grow in case of mismatch
    const wfa_wavefront_t* prev_wf = &M_wavefronts[(curr_wf + x) % active_working_set_size];
    const int hi = prev_wf->hi;
    const int lo = prev_wf->lo;

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        uint4 cell = LOAD_CELL(prev_wf->cells[k]);
        wfa_offset_t curr_offset = UINT4_TO_OFFSET(cell) + 1;
        wfa_bt_vector_t backtrace_val = UINT4_TO_BT_VECTOR(cell);
        wfa_bt_prev_t prev = UINT4_TO_BT_PREV(cell);

        if (curr_offset >= 0) {
            curr_offset = WF_extend_kernel(text, pattern, tlen, plen, k, curr_offset);
            backtrace_val = (backtrace_val << 2) | OP_SUB;

            if (BT_IS_FULL(backtrace_val)) {
                prev = offload_backtrace(last_free_bt_position,
                                         backtrace_val,
                                         prev,
                                         offloaded_backtraces);
                backtrace_val = 0L;
            }
        }


        STORE_CELL(
            M_wavefronts[curr_wf].cells[k],
            curr_offset,
            backtrace_val,
            prev
        );
    }

    if (threadIdx.x == 0) {

#if 0
        wfa_wavefront_t curr_wf_obj_M = M_wavefronts[curr_wf];
        wfa_wavefront_t curr_wf_obj_I = I_wavefronts[curr_wf];
        wfa_wavefront_t curr_wf_obj_D = D_wavefronts[curr_wf];
        printf("____________OFFSETS_______________\n");
        printf("       |     ~M |     ~I |     ~D \n");
        for (int k=hi; k>=lo; k--) {
            const uint4 Mcell = LOAD_CELL(curr_wf_obj_M.cells[k]);
            wfa_offset_t Moffset = UINT4_TO_OFFSET(Mcell);

            const uint4 Icell = LOAD_CELL(curr_wf_obj_I.cells[k]);
            wfa_offset_t Ioffset = UINT4_TO_OFFSET(Icell);

            const uint4 Dcell = LOAD_CELL(curr_wf_obj_D.cells[k]);
            wfa_offset_t Doffset = UINT4_TO_OFFSET(Dcell);

            printf("k=%4d |", k);

            if (Moffset >= 0)
               printf(" %6d ", Moffset);
            else printf("      x ", Moffset);

            printf("|");

            if (Ioffset >= 0)
               printf(" %6d ", Ioffset);
            else printf("      x ", Ioffset);

            printf("|");

            if (Doffset >= 0)
               printf(" %6d ", Doffset);
            else printf("      x ", Doffset);

            printf("\n");

        }
        printf("__________________________________\n");
        printf("\n");

        printf("___________BACKTRACES_____________\n");
        printf("k=     |     ~M |     ~I |     ~D \n");
        for (int k=hi; k>=lo; k--) {
            const uint4 Mcell = LOAD_CELL(curr_wf_obj_M.cells[k]);
            wfa_bt_vector_t Mbt = UINT4_TO_BT_VECTOR(Mcell);

            const uint4 Icell = LOAD_CELL(curr_wf_obj_I.cells[k]);
            wfa_bt_vector_t Ibt = UINT4_TO_BT_VECTOR(Icell);

            const uint4 Dcell = LOAD_CELL(curr_wf_obj_D.cells[k]);
            wfa_bt_vector_t Dbt = UINT4_TO_BT_VECTOR(Dcell);

            printf("k=%4d |", k);

           printf(" %016x ", Mbt);

            printf("|");

           printf(" 0x%016x ", Ibt);

            printf("|");

           printf(" 0x%016x ", Dbt);

            printf("\n");

        }
        printf("__________________________________\n");
        printf("\n");
#endif

        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist= true;
    }
}

__device__ void next_MDI (wfa_wavefront_t* M_wavefronts,
                          wfa_wavefront_t* I_wavefronts,
                          wfa_wavefront_t* D_wavefronts,
                          const int curr_wf,
                          const int active_working_set_size,
                          const int x,
                          const int o,
                          const int e,
                          const char* text,
                          const char* pattern,
                          const int tlen,
                          const int plen,
                          unsigned int* const last_free_bt_position,
                          wfa_backtrace_t* const offloaded_backtraces) {
    const wfa_wavefront_t* prev_wf_x   = &M_wavefronts[(curr_wf + x) % active_working_set_size];
    const wfa_wavefront_t* prev_wf_o   = &M_wavefronts[(curr_wf + o + e) % active_working_set_size];
    const wfa_wavefront_t* prev_I_wf_e = &I_wavefronts[(curr_wf + e) % active_working_set_size];
    const wfa_wavefront_t* prev_D_wf_e = &D_wavefronts[(curr_wf + e) % active_working_set_size];

    const int hi_ID = MAX(prev_wf_o->hi, MAX(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1;
    const int hi    = MAX(prev_wf_x->hi, hi_ID);
    const int lo_ID = MIN(prev_wf_o->lo, MIN(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1;
    const int lo    = MIN(prev_wf_x->lo, lo_ID);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        // ~I gap open offset load
        // TODO: Move the +1 of the offsets at the end when storing
        const uint4 cell_gap_open_I = LOAD_CELL(prev_wf_o->cells[k - 1]);
        const wfa_offset_t I_gap_open_offset = UINT4_TO_OFFSET(cell_gap_open_I) + 1;
        const int64_t I_gap_open_offset_pb = (int64_t)
                                  ((uint64_t)I_gap_open_offset << 32)
                                  | GAP_OPEN;

        // ~I gap open backtrace load
        const wfa_bt_vector_t I_gap_open_bt_val = UINT4_TO_BT_VECTOR(cell_gap_open_I);
        const wfa_bt_prev_t I_gap_open_bt_prev = UINT4_TO_BT_PREV(cell_gap_open_I);

        // ~I gap extend offset load
        const uint4 cell_gap_extend_I = LOAD_CELL(prev_I_wf_e->cells[k - 1]);
        const wfa_offset_t I_gap_extend_offset = \
                                         UINT4_TO_OFFSET(cell_gap_extend_I) + 1;
        const int64_t I_gap_extend_offset_pb = (int64_t)
                                ((uint64_t)I_gap_extend_offset << 32)
                                | GAP_EXTEND;

        // ~I gap extend backtrace load
        const wfa_bt_vector_t I_gap_extend_bt_val = \
                                          UINT4_TO_BT_VECTOR(cell_gap_extend_I);
        const wfa_bt_prev_t I_gap_extend_bt_prev = \
                                          UINT4_TO_BT_PREV(cell_gap_extend_I);

        int64_t I_offset_pb = MAX_PB(I_gap_open_offset_pb,
                                     I_gap_extend_offset_pb);

        const wfa_offset_t I_offset = (wfa_offset_t)(I_offset_pb >> 32);

        // ~I backtraces
        wfa_bt_vector_t I_backtrace_vector = 0L;
        wfa_bt_prev_t   I_backtrace_prev = 0;
        if (I_offset >= 0) {
            const gap_op_t I_op = (gap_op_t)(I_offset_pb & 0xffffffff);

            if (I_op == GAP_OPEN) {
                I_backtrace_vector = I_gap_open_bt_val;
                I_backtrace_prev = I_gap_open_bt_prev;
            } else {
                I_backtrace_vector = I_gap_extend_bt_val;
                I_backtrace_prev = I_gap_extend_bt_prev;
            }

            I_backtrace_vector = (I_backtrace_vector << 2) | OP_INS;

            // TODO: Needed to offload ~I and ~D backtraces?
            // Offload ~I backtraces if the bitvector is full
            if (BT_IS_FULL(I_backtrace_vector)) {
                I_backtrace_prev = offload_backtrace(last_free_bt_position,
                                                  I_backtrace_vector,
                                                  I_backtrace_prev,
                                                  offloaded_backtraces);
                I_backtrace_vector = 0L;
            }

        }

        STORE_CELL(
            I_wavefronts[curr_wf].cells[k],
            I_offset,
            I_backtrace_vector,
            I_backtrace_prev
        );

        I_offset_pb = (uint64_t)(((uint64_t)I_offset << 32) | OP_INS);

        // ~D offsets
        const uint4 D_gap_open_cell = LOAD_CELL(prev_wf_o->cells[k + 1]);
        const wfa_offset_t D_gap_open_offset = UINT4_TO_OFFSET(D_gap_open_cell);
        const wfa_bt_vector_t D_gap_open_bt_val = UINT4_TO_BT_VECTOR(D_gap_open_cell);
        const wfa_bt_prev_t D_gap_open_bt_prev = UINT4_TO_BT_PREV(D_gap_open_cell);
        const int64_t D_gap_open_offset_pb = (int64_t)
                                  ((uint64_t)D_gap_open_offset << 32)
                                  | GAP_OPEN;

        const uint4 D_gap_extend_cell = LOAD_CELL(prev_D_wf_e->cells[k + 1]);
        const wfa_offset_t D_gap_extend_offset = UINT4_TO_OFFSET(D_gap_extend_cell);
        const wfa_bt_vector_t D_gap_extend_bt_val = \
                                          UINT4_TO_BT_VECTOR(D_gap_extend_cell);
        const wfa_bt_prev_t D_gap_extend_bt_prev = \
                                          UINT4_TO_BT_PREV(D_gap_extend_cell);
        const int64_t D_gap_extend_offset_pb = (int64_t)
                                    ((uint64_t)D_gap_extend_offset << 32)
                                    | GAP_EXTEND;

        int64_t D_offset_pb = MAX_PB(D_gap_open_offset_pb,
                                     D_gap_extend_offset_pb);

        const wfa_offset_t D_offset = (wfa_offset_t)(D_offset_pb >> 32);

        // ~D backtraces
        wfa_bt_vector_t D_backtrace_vector = 0L;
        wfa_bt_prev_t   D_backtrace_prev = 0;
        if (D_offset >= 0) {
            const gap_op_t D_op = (gap_op_t)(D_offset_pb & 0xffffffff);

            if (D_op == GAP_OPEN) {
                D_backtrace_vector = D_gap_open_bt_val;
                D_backtrace_prev = D_gap_open_bt_prev;
            } else {
                D_backtrace_vector = D_gap_extend_bt_val;
                D_backtrace_prev = D_gap_extend_bt_prev;
            }

            D_backtrace_vector = (D_backtrace_vector << 2) | OP_DEL;

            // Offload ~D backtraces if the bitvector is full
            if (BT_IS_FULL(D_backtrace_vector)) {
                D_backtrace_prev = offload_backtrace(last_free_bt_position,
                                                  D_backtrace_vector,
                                                  D_backtrace_prev,
                                                  offloaded_backtraces);
                D_backtrace_vector = 0L;
            }
        }

        STORE_CELL(
            D_wavefronts[curr_wf].cells[k],
            D_offset,
            D_backtrace_vector,
            D_backtrace_prev
        );

        D_offset_pb = (uint64_t)(((uint64_t)D_offset << 32) | OP_DEL);

        // ~M update
        const uint4 X_cell = LOAD_CELL(prev_wf_x->cells[k]);
        const wfa_offset_t X_offset = UINT4_TO_OFFSET(X_cell) + 1;
        const wfa_bt_vector_t X_backtrace_val = UINT4_TO_BT_VECTOR(X_cell);
        const wfa_bt_prev_t X_backtrace_prev = UINT4_TO_BT_PREV(X_cell);
        const int64_t X_offset_pb = (int64_t)
                                     (((uint64_t)X_offset << 32)
                                     | OP_SUB);

        const int64_t M_offset_pb = MAX_PB(
                                        MAX_PB(X_offset_pb, D_offset_pb),
                                        I_offset_pb
                                        );

        // Extend
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);
        wfa_bt_vector_t M_backtrace_vector = 0L;
        wfa_bt_prev_t   M_backtrace_prev = 0;
        if (M_offset >= 0) {
            M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);

            affine_op_t M_op = (affine_op_t)(M_offset_pb & 0xffffffff);
            if (M_op == OP_INS) {
                M_backtrace_vector = I_backtrace_vector;
                M_backtrace_prev = I_backtrace_prev;
            } else if (M_op == OP_SUB) {
                //M_backtrace_vector = X_backtrace_val;
                M_backtrace_vector = X_backtrace_val;
                M_backtrace_prev = X_backtrace_prev;
            } else {
                M_backtrace_vector = D_backtrace_vector;
                M_backtrace_prev = D_backtrace_prev;
            }

            // Always add SUB as it is also de delimiter for extensions
            M_backtrace_vector = (M_backtrace_vector << 2) | OP_SUB;

            // Offload backtraces if the bitvector is full
            if (BT_IS_FULL(M_backtrace_vector)) {
                M_backtrace_prev = offload_backtrace(last_free_bt_position,
                                                  M_backtrace_vector,
                                                  M_backtrace_prev,
                                                  offloaded_backtraces);
                M_backtrace_vector = 0L;
            }
        }

        STORE_CELL(
            M_wavefronts[curr_wf].cells[k],
            M_offset,
            M_backtrace_vector,
            M_backtrace_prev
        );
    }

    if (threadIdx.x == 0) {
#if 0
        wfa_wavefront_t curr_wf_obj_M = M_wavefronts[curr_wf];
        wfa_wavefront_t curr_wf_obj_I = I_wavefronts[curr_wf];
        wfa_wavefront_t curr_wf_obj_D = D_wavefronts[curr_wf];
        printf("____________OFFSETS_______________\n");
        printf("       |     ~M |     ~I |     ~D \n");
        for (int k=hi; k>=lo; k--) {
            const uint4 Mcell = LOAD_CELL(curr_wf_obj_M.cells[k]);
            wfa_offset_t Moffset = UINT4_TO_OFFSET(Mcell);

            const uint4 Icell = LOAD_CELL(curr_wf_obj_I.cells[k]);
            wfa_offset_t Ioffset = UINT4_TO_OFFSET(Icell);

            const uint4 Dcell = LOAD_CELL(curr_wf_obj_D.cells[k]);
            wfa_offset_t Doffset = UINT4_TO_OFFSET(Dcell);

            printf("k=%4d |", k);

            if (Moffset >= 0)
               printf(" %6d ", Moffset);
            else printf("      x ", Moffset);

            printf("|");

            if (Ioffset >= 0)
               printf(" %6d ", Ioffset);
            else printf("      x ", Ioffset);

            printf("|");

            if (Doffset >= 0)
               printf(" %6d ", Doffset);
            else printf("      x ", Doffset);

            printf("\n");

        }
        printf("__________________________________\n");
        printf("\n");

        printf("___________BACKTRACES_____________\n");
        printf("k=     |     ~M |     ~I |     ~D \n");
        for (int k=hi; k>=lo; k--) {
            const uint4 Mcell = LOAD_CELL(curr_wf_obj_M.cells[k]);
            wfa_bt_vector_t Mbt = UINT4_TO_BT_VECTOR(Mcell);

            const uint4 Icell = LOAD_CELL(curr_wf_obj_I.cells[k]);
            wfa_bt_vector_t Ibt = UINT4_TO_BT_VECTOR(Icell);

            const uint4 Dcell = LOAD_CELL(curr_wf_obj_D.cells[k]);
            wfa_bt_vector_t Dbt = UINT4_TO_BT_VECTOR(Dcell);

            printf("k=%4d |", k);

           printf(" %016llx ", Mbt);

            printf("|");

           printf(" 0x%016llx ", Ibt);

            printf("|");

           printf(" 0x%016llx ", Dbt);

            printf("\n");

        }
        printf("__________________________________\n");
        printf("\n");
#endif

        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist = true;

        I_wavefronts[curr_wf].hi = hi_ID;
        I_wavefronts[curr_wf].lo = lo_ID;
        I_wavefronts[curr_wf].exist = true;

        D_wavefronts[curr_wf].hi = hi_ID;
        D_wavefronts[curr_wf].lo = lo_ID;
        D_wavefronts[curr_wf].exist = true;
    }
}

__device__ void update_curr_wf (wfa_wavefront_t* M_wavefronts,
                                wfa_wavefront_t* I_wavefronts,
                                wfa_wavefront_t* D_wavefronts,
                                const int active_working_set_size,
                                const int max_wf_size,
                                int* curr_wf) {
    // As we read wavefronts "forward" in the waveronts arrays, so the wavefront
    // index is moved backwards.
    const int wf_idx = (*curr_wf - 1 + active_working_set_size) % active_working_set_size;

    // TODO: Check if this is necessary in some penalties combination
    // Set new wf to NULL, as new wavefront may be smaller than the
    // previous one
    //wfa_offset_t* to_clean_M = M_wavefronts[wf_idx].offsets - (max_wf_size/2);
    //M_wavefronts[wf_idx].exist = false;

    //wfa_offset_t* to_clean_I = I_wavefronts[wf_idx].offsets - (max_wf_size/2);
    //I_wavefronts[wf_idx].exist = false;

    //wfa_offset_t* to_clean_D = D_wavefronts[wf_idx].offsets - (max_wf_size/2);
    //D_wavefronts[wf_idx].exist = false;

    //for (int i=threadIdx.x; i<max_wf_size; i+=blockDim.x) {
    //    to_clean_M[i] = -1;
    //    to_clean_D[i] = -1;
    //    to_clean_I[i] = -1;
    //}

    *curr_wf = wf_idx;

}

__global__ void alignment_kernel (
                            const char* packed_sequences_buffer,
                            const sequence_pair_t* sequences_metadata,
                            const size_t num_alignments,
                            const int max_steps,
                            uint8_t* const wf_data_buffer,
                            const affine_penalties_t penalties,
                            wfa_backtrace_t* offloaded_backtraces_global,
                            wfa_backtrace_t* offloaded_backtraces_results,
                            alignment_result_t* results) {
    const int tid = threadIdx.x;
    // m = 0 for WFA
    const int x = penalties.x;
    const int o = penalties.o;
    const int e = penalties.e;

    const sequence_pair_t curr_batch_alignment_base = sequences_metadata[0];
    const size_t base_offset_packed = curr_batch_alignment_base.pattern_offset_packed;

    const sequence_pair_t metadata = sequences_metadata[blockIdx.x];
    const char* text = packed_sequences_buffer + metadata.text_offset_packed - base_offset_packed;
    const char* pattern = packed_sequences_buffer + metadata.pattern_offset_packed - base_offset_packed ;
    const int tlen = metadata.text_len;
    const int plen = metadata.pattern_len;

    // TODO: Move to function/macro + use in lib/sequence_alignment.cu
    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps);
    wfa_backtrace_t* const offloaded_backtraces = \
             &offloaded_backtraces_global[blockIdx.x * bt_offloaded_size];

    size_t bt_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps);
    wfa_backtrace_t* const offloaded_backtrace_results_base = \
             &offloaded_backtraces_results[blockIdx.x * bt_results_size];

    // In shared memory:
    // - Wavefronts needed to calculate current WF_s, there are 3 "pyramids" so
    //   this number of wavefront is 3 times (WF_{max(o+e, x)} --> WF_s)
    extern __shared__ char sh_mem[];

    // TODO: +1 because of the current wf?
    const int active_working_set_size = MAX(o+e, x) + 1;
    const int max_wf_size = 2 * max_steps + 1;

    // Offsets and backtraces must be 32 bits aligned to avoid unaligned access
    // errors on the structs
    uint32_t cells_size = active_working_set_size * max_wf_size;
    cells_size = cells_size + (4 - (cells_size % 4));

    const size_t wf_data_buffer_size = cells_size * 3 * sizeof(wfa_cell_t);
    uint8_t* curr_alignment_wf_data_buffer = wf_data_buffer
                                             + (wf_data_buffer_size * blockIdx.x);

    wfa_cell_t* M_base = (wfa_cell_t*)curr_alignment_wf_data_buffer;
    wfa_cell_t* I_base = M_base + cells_size;
    wfa_cell_t* D_base = I_base + cells_size;

    // Wavefronts structres reside in shared
    wfa_wavefront_t* M_wavefronts = (wfa_wavefront_t*)sh_mem;
    wfa_wavefront_t* I_wavefronts = (M_wavefronts + active_working_set_size);
    wfa_wavefront_t* D_wavefronts = (I_wavefronts + active_working_set_size);

    // Pointer to the current free slot to offload a backtrace block
    uint32_t* last_free_bt_position = (uint32_t*)
                                          (D_wavefronts + active_working_set_size);

    // Start at 1 because 0 is used as NULL (no more backtraces blocks to
    // recover)
    *last_free_bt_position = 1;

    // Initialize all wavefronts to -1
    for (int i=tid; i<(cells_size * 3); i+=blockDim.x) {
        STORE_CELL(M_base[i], (uint32_t)-10000, 0L, 0);
    }

    // Initialize wavefronts memory
    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts[i].cells = M_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].hi = 0;
        M_wavefronts[i].lo = 0;
        M_wavefronts[i].exist = false;

        I_wavefronts[i].cells = I_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].hi = 0;
        I_wavefronts[i].lo = 0;
        I_wavefronts[i].exist = false;

        D_wavefronts[i].cells = D_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].hi = 0;
        D_wavefronts[i].lo = 0;
        D_wavefronts[i].exist = false;
    }

    // TODO: Is this necessary? As thread 0 sets curr_wf[0] AND does the first
    // extend.
    //__syncthreads();

    int curr_wf = 0;

    if (tid == 0) {
        wfa_offset_t initial_ext = WF_extend_kernel(
            text,
            pattern,
            tlen, plen,
            0, 0);

        STORE_CELL(
            M_wavefronts[curr_wf].cells[0],
            initial_ext, // Offset
            0L,          // Backtrace vector
            0);          // Previous backtrace block

        M_wavefronts[curr_wf].exist = true;
    }

    __syncthreads();

    // TODO: Change tarket K if we don't start form WF 0 (cooperative strategy)
    const int target_k = EWAVEFRONT_DIAGONAL(tlen, plen);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const wfa_offset_t target_offset = EWAVEFRONT_OFFSET(tlen, plen);

    bool finished = false;

    int distance = 0;
    // steps = number of editions
    int steps = 0;
    // TODO: target_k_abs <= distance or <= steps (?)
    uint4 target_cell = LOAD_CELL(M_wavefronts[curr_wf].cells[target_k]);
    wfa_offset_t curr_target_offset = UINT4_TO_OFFSET(target_cell);
    if (!(target_k_abs <= distance
            && M_wavefronts[curr_wf].exist
            && curr_target_offset == target_offset)) {

        update_curr_wf(
            M_wavefronts,
            I_wavefronts,
            D_wavefronts,
            active_working_set_size,
            max_wf_size,
            &curr_wf);

        distance++;
        steps++;
        //__syncthreads();

        while (steps < (max_steps - 1)) {
            bool M_exist = false;
            bool GAP_exist = false;
            const int o_delta = (curr_wf + o + e) % active_working_set_size;
            const int e_delta = (curr_wf + e) % active_working_set_size;
            const int x_delta = (curr_wf + x) % active_working_set_size;
            if ((distance - o - e) >= 0) {
                // Just test with I because I and D exist in the same distances
                GAP_exist = M_wavefronts[o_delta].exist
                          || I_wavefronts[e_delta].exist;
            }

            if (GAP_exist) {
                M_exist = true;
            } else {
                if ((distance - x) >= 0) {
                    M_exist = M_wavefronts[x_delta].exist;
                } 
            }

            if (!GAP_exist && !M_exist) {
                distance++;
            } else {
                if (M_exist && !GAP_exist) {
                    next_M(M_wavefronts, I_wavefronts, D_wavefronts, curr_wf, active_working_set_size, x,
                           text, pattern, tlen, plen,
                           last_free_bt_position, offloaded_backtraces);
                } else {
                    next_MDI(
                        M_wavefronts, I_wavefronts, D_wavefronts,
                        curr_wf, active_working_set_size,
                        x, o, e,
                        text, pattern, tlen, plen,
                        last_free_bt_position, offloaded_backtraces);

                    // Wavefront only grows if there's an operation in the ~I or
                    // ~D matrices
                    steps++;
                }

                // TODO: This is necessary for now, try to find a less sync
                // version
                __syncthreads();

                target_cell = LOAD_CELL(M_wavefronts[curr_wf].cells[target_k]);
                curr_target_offset = UINT4_TO_OFFSET(target_cell);

                if (target_k_abs <= distance
                        && M_exist
                        && curr_target_offset == target_offset) {
                    finished = true;
                    break;
                }

                distance++;
            }

        update_curr_wf(
            M_wavefronts,
            I_wavefronts,
            D_wavefronts,
            active_working_set_size,
            max_wf_size,
            &curr_wf);

        //__syncthreads();
        }
    } else {
        finished = true;
    }

    if  (tid == 0) {
        results[blockIdx.x].distance = distance;
        results[blockIdx.x].finished = finished;
        target_cell = LOAD_CELL(M_wavefronts[curr_wf].cells[target_k]);
        wfa_bt_vector_t backtrace_val = UINT4_TO_BT_VECTOR(target_cell);
        wfa_bt_prev_t   prev          = UINT4_TO_BT_PREV(target_cell);
        wfa_backtrace_t backtrace = {
            .backtrace = backtrace_val,
            .prev      = prev
        };

        results[blockIdx.x].backtrace = backtrace;

        wfa_backtrace_t* curr_result = &backtrace;

        // Save the list in reversed order
        int i = 0;
        while (curr_result->prev != 0) {
            offloaded_backtrace_results_base[i] = \
                                        offloaded_backtraces[curr_result->prev];
            curr_result = &offloaded_backtraces[curr_result->prev];
            i++;
        }

        results[blockIdx.x].num_bt_blocks = i;
    }
}
