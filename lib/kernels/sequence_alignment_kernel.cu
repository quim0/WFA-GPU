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
#define BT_WORD_FULL_CMP 0x40000000
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

    return offset_k + acc;
}

__device__ bt_prev_t offload_backtrace (unsigned int* const last_free_bt_position,
                                       const bt_vector_t backtrace_vector,
                                       const bt_prev_t backtrace_pointer,
                                       wfa_backtrace_t* const global_backtraces_array) {
    const bt_prev_t old_val = atomicAdd(last_free_bt_position, 1);

    // TODO: Split global backtraces array between backtraces vectors and
    // pointers
    global_backtraces_array[old_val] = {.backtrace = backtrace_vector,
                                        .prev = backtrace_pointer};

    // TODO: Check if new_val is more than 32 bits
    return old_val;
}

__device__ void next_M (wfa_wavefront_t* M_wavefronts,
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
        wfa_offset_t curr_offset = prev_wf->offsets[k] + 1;

        curr_offset = WF_extend_kernel(text, pattern,
                                       tlen, plen, k, curr_offset);

        M_wavefronts[curr_wf].offsets[k] = curr_offset;

        bt_vector_t prev_bt_vector = prev_wf->backtraces_vectors[k];
        bt_prev_t prev_bt_pointer = prev_wf->backtraces_pointers[k];

        prev_bt_vector = (prev_bt_vector << 2) | OP_SUB;

        if (BT_IS_FULL(prev_bt_vector)) {
            bt_prev_t prev = offload_backtrace(last_free_bt_position,
                                     prev_bt_vector,
                                     prev_bt_pointer,
                                     offloaded_backtraces);
            prev_bt_vector = 0;
            prev_bt_pointer = prev;
        }

        M_wavefronts[curr_wf].backtraces_vectors[k] = prev_bt_vector;
        M_wavefronts[curr_wf].backtraces_pointers[k] = prev_bt_pointer;
    }

    if (threadIdx.x == 0) {
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
    const wfa_wavefront_t* prev_wf_x =   &M_wavefronts[(curr_wf + x) % active_working_set_size];
    const wfa_wavefront_t* prev_wf_o =   &M_wavefronts[(curr_wf + o + e) % active_working_set_size];
    const wfa_wavefront_t* prev_I_wf_e = &I_wavefronts[(curr_wf + e) % active_working_set_size];
    const wfa_wavefront_t* prev_D_wf_e = &D_wavefronts[(curr_wf + e) % active_working_set_size];

    const int hi_ID = MAX(prev_wf_o->hi, MAX(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1;
    const int hi    = MAX(prev_wf_x->hi, hi_ID);
    const int lo_ID = MIN(prev_wf_o->lo, MIN(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1;
    const int lo    = MIN(prev_wf_x->lo, lo_ID);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        // ~I offsets
        const wfa_offset_t I_gap_open_offset = prev_wf_o->offsets[k - 1] + 1;
        const bt_vector_t I_gap_open_bt_vector = prev_wf_o->backtraces_vectors[k - 1];
        const bt_prev_t I_gap_open_bt_pointer = prev_wf_o->backtraces_pointers[k - 1];

        const int64_t I_gap_open_offset_pb = (int64_t)
                                  ((uint64_t)I_gap_open_offset << 32)
                                  | GAP_OPEN;

        const wfa_offset_t I_gap_extend_offset = prev_I_wf_e->offsets[k - 1] + 1;
        const bt_vector_t I_gap_extend_bt_vector = prev_I_wf_e->backtraces_vectors[k - 1];
        const bt_prev_t I_gap_extend_bt_pointer = prev_I_wf_e->backtraces_pointers[k - 1];

        const int64_t I_gap_extend_offset_pb = (int64_t)
                                ((uint64_t)I_gap_extend_offset << 32)
                                | GAP_EXTEND;

        int64_t I_offset_pb = MAX_PB(I_gap_open_offset_pb,
                                     I_gap_extend_offset_pb);

        const wfa_offset_t I_offset = (wfa_offset_t)(I_offset_pb >> 32);
        I_wavefronts[curr_wf].offsets[k] = I_offset;

        // ~I backtraces
        // Include backtrace and previous backtrace offset
        const gap_op_t I_op = (gap_op_t)(I_offset_pb & 0xffffffff);
        bt_vector_t I_backtrace_vector;
        bt_prev_t I_backtrace_pointer;

        if (I_op == GAP_OPEN) {
            I_backtrace_vector = I_gap_open_bt_vector;
            I_backtrace_pointer = I_gap_open_bt_pointer;
        } else {
            I_backtrace_vector = I_gap_extend_bt_vector;
            I_backtrace_pointer = I_gap_extend_bt_pointer;
        }

        I_backtrace_vector = (I_backtrace_vector << 2) | OP_INS;

        // TODO: Needed to offload ~I and ~D backtraces?
        // Offload ~I backtraces if the bitvector is full, and reset current
        // backtrace
        if (BT_IS_FULL(I_backtrace_vector)) {
            bt_prev_t prev = offload_backtrace(last_free_bt_position,
                                              I_backtrace_vector,
                                              I_backtrace_pointer,
                                              offloaded_backtraces);
            I_backtrace_vector = 0;
            I_backtrace_pointer = prev;
        }

        I_wavefronts[curr_wf].backtraces_vectors[k] = I_backtrace_vector;
        I_wavefronts[curr_wf].backtraces_pointers[k] = I_backtrace_pointer;
        I_offset_pb = (uint64_t)(((uint64_t)I_offset << 32) | OP_INS);

        // ------- End of ~I processing -------

        // ~D offsets
        const wfa_offset_t D_gap_open_offset = prev_wf_o->offsets[k + 1];
        const bt_vector_t D_gap_open_bt_vector = prev_wf_o->backtraces_vectors[k + 1];
        const bt_prev_t D_gap_open_bt_pointer = prev_wf_o->backtraces_pointers[k + 1];

        const int64_t D_gap_open_offset_pb = (int64_t)
                                  ((uint64_t)D_gap_open_offset << 32)
                                  | GAP_OPEN;

        const wfa_offset_t D_gap_extend_offset = prev_D_wf_e->offsets[k + 1];
        const bt_vector_t D_gap_extend_bt_vector = prev_D_wf_e->backtraces_vectors[k + 1];
        const bt_prev_t D_gap_extend_bt_pointer = prev_D_wf_e->backtraces_pointers[k + 1];

        const int64_t D_gap_extend_offset_pb = (int64_t)
                                    ((uint64_t)D_gap_extend_offset << 32)
                                    | GAP_EXTEND;

        int64_t D_offset_pb = MAX_PB(D_gap_open_offset_pb,
                                     D_gap_extend_offset_pb);

        const wfa_offset_t D_offset = (wfa_offset_t)(D_offset_pb >> 32);
        D_wavefronts[curr_wf].offsets[k] = D_offset;

        // ~D backtraces
        const gap_op_t D_op = (gap_op_t)(D_offset_pb & 0xffffffff);
        bt_vector_t D_backtrace_vector;
        bt_prev_t D_backtrace_pointer;

        if (D_op == GAP_OPEN) {
            D_backtrace_vector = D_gap_open_bt_vector;
            D_backtrace_pointer = D_gap_open_bt_pointer;
        } else {
            D_backtrace_vector = D_gap_extend_bt_vector;
            D_backtrace_pointer = D_gap_extend_bt_pointer;
        }

        D_backtrace_vector = (D_backtrace_vector << 2) | OP_DEL;

        // Offload ~D backtraces if the bitvector is full
        if (BT_IS_FULL(D_backtrace_vector)) {
            bt_prev_t prev = offload_backtrace(last_free_bt_position,
                                              D_backtrace_vector,
                                              D_backtrace_pointer,
                                              offloaded_backtraces);
            D_backtrace_vector = 0;
            D_backtrace_pointer = prev;
        }

        D_wavefronts[curr_wf].backtraces_vectors[k] = D_backtrace_vector;
        D_wavefronts[curr_wf].backtraces_pointers[k] = D_backtrace_pointer;

        D_offset_pb = (uint64_t)(((uint64_t)D_offset << 32) | OP_DEL);

        // ------- End of ~D processing -------

        // ~M update
        const wfa_offset_t X_offset = prev_wf_x->offsets[k] + 1;
        const bt_vector_t X_backtrace_vector = prev_wf_x->backtraces_vectors[k];
        const bt_prev_t X_backtrace_pointer = prev_wf_x->backtraces_pointers[k];

        const int64_t X_offset_pb = (int64_t)
                                     (((uint64_t)X_offset << 32)
                                     | OP_SUB);

        const int64_t M_offset_pb = MAX_PB(
                                        MAX_PB(X_offset_pb, D_offset_pb),
                                        I_offset_pb
                                        );
        // Extend
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);
        if (M_offset >= 0)
            M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);

        M_wavefronts[curr_wf].offsets[k] = M_offset;

        affine_op_t M_op = (affine_op_t)(M_offset_pb & 0xffffffff);
        bt_vector_t M_backtrace_vector;
        bt_prev_t M_backtrace_pointer;
        if (M_op == OP_SUB) {
            M_backtrace_vector = X_backtrace_vector;
            M_backtrace_pointer = X_backtrace_pointer;
        } else if (M_op == OP_INS) {
            M_backtrace_vector = I_backtrace_vector;
            M_backtrace_pointer = I_backtrace_pointer;
        } else {
            M_backtrace_vector = D_backtrace_vector;
            M_backtrace_pointer = D_backtrace_pointer;
        }

        M_backtrace_vector = (M_backtrace_vector << 2) | OP_SUB;

        // Offload backtraces if the bitvector is full
        if (BT_IS_FULL(M_backtrace_vector)) {
            bt_prev_t prev = offload_backtrace(last_free_bt_position,
                                              M_backtrace_vector,
                                              M_backtrace_pointer,
                                              offloaded_backtraces);
            M_backtrace_vector = 0;
            M_backtrace_pointer = prev;
        }

        M_wavefronts[curr_wf].backtraces_vectors[k] = M_backtrace_vector;
        M_wavefronts[curr_wf].backtraces_pointers[k] = M_backtrace_pointer;
    }

    if (threadIdx.x == 0) {
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
    int offsets_size = active_working_set_size * max_wf_size;
    offsets_size = offsets_size + (4 - (offsets_size % 4));

    int bt_size = active_working_set_size * max_wf_size;
    bt_size = bt_size + (4 - (bt_size % 4));

    const size_t wf_data_buffer_size =
                    // Offsets space
                    (offsets_size * 3 * sizeof(wfa_offset_t))
                    // Backtraces space
                    + (bt_size * 3 * sizeof(wfa_backtrace_t));
    uint8_t* curr_alignment_wf_data_buffer = wf_data_buffer
                                             + (wf_data_buffer_size * blockIdx.x);

    wfa_offset_t* M_base = (wfa_offset_t*)curr_alignment_wf_data_buffer;
    wfa_offset_t* I_base = M_base + offsets_size;
    wfa_offset_t* D_base = I_base + offsets_size;

    // Backtrace vectors
    bt_vector_t* M_bt_vector_base = (bt_vector_t*)(D_base + offsets_size);
    bt_vector_t* I_bt_vector_base = M_bt_vector_base + bt_size;
    bt_vector_t* D_bt_vector_base = I_bt_vector_base + bt_size;

    // Baktrace vector pointers
    bt_prev_t* M_bt_prev_base = (bt_prev_t*)(D_bt_vector_base + bt_size);
    bt_prev_t* I_bt_prev_base = M_bt_prev_base + bt_size;
    bt_prev_t* D_bt_prev_base = I_bt_prev_base + bt_size;

    /*
    wfa_backtrace_t* M_bt_base = (wfa_backtrace_t*)(D_base + offsets_size);
    wfa_backtrace_t* I_bt_base = M_bt_base + bt_size;
    wfa_backtrace_t* D_bt_base = I_bt_base + bt_size;
    */

    // Wavefronts structres reside in shared
    wfa_wavefront_t* M_wavefronts = (wfa_wavefront_t*)sh_mem;
    wfa_wavefront_t* I_wavefronts = (M_wavefronts + active_working_set_size);
    wfa_wavefront_t* D_wavefronts = (I_wavefronts + active_working_set_size);

    uint32_t* last_free_bt_position = (uint32_t*)
                                          (D_wavefronts + active_working_set_size);

    // Start at 1 because 0 is used as NULL (no more backtraces blocks to
    // recover)
    *last_free_bt_position = 1;

    // Initialize all wavefronts to -1
    for (int i=tid; i<(offsets_size * 3); i+=blockDim.x) {
        M_base[i] = -1000;
    }

    // Initialise all backtrace vectors to 0
    for (int i=tid; i<(bt_size * 3); i+=blockDim.x) {
        M_bt_vector_base[i] = 0;
    }

    // Initialise all backtrace pointers to 0
    for (int i=tid; i<(bt_size * 3); i+=blockDim.x) {
        M_bt_prev_base[i] = 0;
    }

    // Initialize wavefronts memory
    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts[i].offsets = M_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].backtraces_vectors = M_bt_vector_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].backtraces_pointers = M_bt_prev_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].hi = 0;
        M_wavefronts[i].lo = 0;
        M_wavefronts[i].exist = false;

        I_wavefronts[i].offsets = I_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].backtraces_vectors = I_bt_vector_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].backtraces_pointers = I_bt_prev_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].hi = 0;
        I_wavefronts[i].lo = 0;
        I_wavefronts[i].exist = false;

        D_wavefronts[i].offsets = D_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].backtraces_vectors = D_bt_vector_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].backtraces_pointers = D_bt_prev_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].hi = 0;
        D_wavefronts[i].lo = 0;
        D_wavefronts[i].exist = false;
    }

    __syncthreads();

    int curr_wf = 0;

    if (tid == 0) {
        wfa_offset_t initial_ext = WF_extend_kernel(
            text,
            pattern,
            tlen, plen,
            0, 0);
        M_wavefronts[curr_wf].offsets[0] = initial_ext;
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
    if (!(target_k_abs <= distance && M_wavefronts[curr_wf].exist && M_wavefronts[curr_wf].offsets[target_k] == target_offset)) {

        update_curr_wf(
            M_wavefronts,
            I_wavefronts,
            D_wavefronts,
            active_working_set_size,
            max_wf_size,
            &curr_wf);

        distance++;
        steps++;
        __syncthreads();

        while (steps < (max_steps - 1)) {

            bool M_exist = false;
            bool GAP_exist = false;
            const int o_delta = (curr_wf + o + e) % active_working_set_size;
            const int e_delta = (curr_wf + e) % active_working_set_size;
            const int x_delta = (curr_wf + x) % active_working_set_size;
            if ((distance - o - e) >= 0) {
                // Just test ~I matrix as it will exist at the same wavefronts
                // as ~D
                GAP_exist = M_wavefronts[o_delta].exist ||
                            I_wavefronts[e_delta].exist;
            }

            if (GAP_exist) {
                M_exist = true;
            } else {
                if ((distance - x) >= 0) {
                    M_exist = M_wavefronts[x_delta].exist;
                } 
            }

            if (!GAP_exist && !M_exist) {
                M_wavefronts[curr_wf].exist = false;
                D_wavefronts[curr_wf].exist = false;
                I_wavefronts[curr_wf].exist = false;
                distance++;
            } else {
                if (M_exist && !GAP_exist) {
                    next_M(M_wavefronts, curr_wf, active_working_set_size, x,
                           text, pattern, tlen, plen,
                           last_free_bt_position, offloaded_backtraces);
                    D_wavefronts[curr_wf].exist = false;
                    I_wavefronts[curr_wf].exist = false;
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

                if (target_k_abs <= distance && M_exist && M_wavefronts[curr_wf].offsets[target_k] == target_offset) {
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

        __syncthreads();
        }
    } else {
        finished = true;
    }

    if  (tid == 0) {
        results[blockIdx.x].distance = distance;
        results[blockIdx.x].finished = finished;
        bt_vector_t bt_vector = M_wavefronts[curr_wf].backtraces_vectors[target_k];
        bt_prev_t bt_pointer = M_wavefronts[curr_wf].backtraces_pointers[target_k];

        results[blockIdx.x].backtrace = {.backtrace = bt_vector,
                                         .prev = bt_pointer};

        // TODO: Change this hack
        wfa_backtrace_t curr_res_tmp = {.backtrace = bt_vector,
                                         .prev = bt_pointer};
        wfa_backtrace_t* curr_result = &curr_res_tmp;

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
