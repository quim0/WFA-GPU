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
#include "sequence_alignment_kernel.cuh"

#ifdef DEBUG

#define PPRINT_WFS(aws, wfs, d, title) pprint_wavefronts(aws, wfs, d, title)
__device__ void pprint_wavefronts (const int active_working_set_size,
                                   wfa_wavefront_t** wavefronts,
                                   const int distance,
                                   char* title) {
    if (threadIdx.x == 0) {
        if (!title) title = (char*)("");

        printf("%s (distance: %d)\n", title, distance);

        //const int wf_padding_len = 14;

        // Header (WF number)
        for (int i=0; i<active_working_set_size; i++) {
            const int curr_n = i - (active_working_set_size - 1);
            printf("| %.2d          ", curr_n);
        }
        printf(" |\n");

        // exist
        for (int i=0; i<active_working_set_size; i++) {
            printf("| exist: %d     ", wavefronts[i]->exist);
        }
        printf("|\n");

        for (int i=0; i<active_working_set_size; i++) {
            printf("| hi=%.2d, lo=%.2d", wavefronts[i]->hi, wavefronts[i]->lo);
        }
        printf("|\n");
    }
    __syncthreads();
}

#else // DEBUG

#define PPRINT_WFS(aws, wfs, d, title) pprint_wavefronts(aws, wfs, d, title)

#endif // DEBUG


__device__ wfa_offset_t WF_extend_kernel (const char* text,
                                  const char* pattern,
                                  const int tlen,
                                  const int plen,
                                  const int k,
                                  const wfa_offset_t offset_k) {
    int v  = EWAVEFRONT_V(k, offset_k);
    int h  = EWAVEFRONT_H(k, offset_k);

    if (blockIdx.x == 0 && k == -2) {
        //printf("\nextending v=%d, h=%d\n", v, h);
    }

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

__device__ void next_M (wfa_wavefront_t** M_wavefronts,
                        const int curr_wf,
                        const int x,
                        const char* text,
                        const char* pattern,
                        const int tlen,
                        const int plen) {
    // The wavefront do not grow in case of mismatch
    const wfa_wavefront_t* prev_wf = M_wavefronts[curr_wf - x];
    const int hi = prev_wf->hi;
    const int lo = prev_wf->lo;

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        wfa_offset_t curr_offset = prev_wf->offsets[k] + 1;

        curr_offset = WF_extend_kernel(text, pattern,
                                       tlen, plen, k, curr_offset);

        M_wavefronts[curr_wf]->offsets[k] = curr_offset;
        M_wavefronts[curr_wf]->backtraces[k] = (prev_wf->backtraces[k] << 2) | OP_SUB;
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf]->hi = hi;
        M_wavefronts[curr_wf]->lo = lo;
        M_wavefronts[curr_wf]->exist= true;
    }
}

__device__ void next_MI (wfa_wavefront_t** M_wavefronts,
                         wfa_wavefront_t** I_wavefronts,
                         const int curr_wf,
                         const int x,
                         const int o,
                         const int e,
                         const char* text,
                         const char* pattern,
                         const int tlen,
                         const int plen) {
    const wfa_wavefront_t* prev_wf_x = M_wavefronts[curr_wf - x];
    const wfa_wavefront_t* prev_wf_o = M_wavefronts[curr_wf - o - e];
    const wfa_wavefront_t* prev_wf_e = I_wavefronts[curr_wf - e];

    const int hi = max(prev_wf_x->hi, max(prev_wf_o->hi, prev_wf_e->hi)) + 1;
    const int lo = min(prev_wf_x->lo, max(prev_wf_o->lo, prev_wf_e->lo)) - 1;

    for (int k=lo + threadIdx.x; k <= hi; k+=blockIdx.x) {
        wfa_offset_t gap_open_offset = prev_wf_o->offsets[k - 1] + 1;
        wfa_backtrace_t gap_open_bt = prev_wf_o->backtraces[k - 1];
        // {16 bits unused} | {16 bits offset} | {32 bits backtrace}
        uint64_t gap_open_offset_pb = (uint64_t)((uint64_t)gap_open_offset << 32)
                                      | gap_open_bt;

        wfa_offset_t gap_extend_offset = prev_wf_e->offsets[k - 1] + 1;
        wfa_backtrace_t gap_extend_bt = prev_wf_e->backtraces[k - 1];
        uint64_t gap_extend_offset_pb = (uint64_t)((uint64_t)gap_extend_offset << 32)
                                        | gap_extend_bt;

        const uint64_t I_offset_pb = ullmax(gap_open_offset_pb,
                                            gap_extend_offset_pb);

        // Update offsets in the ~I pyramid
        wfa_offset_t I_offset = (wfa_offset_t)(I_offset_pb >> 32);
        I_wavefronts[curr_wf]->offsets[k] = I_offset;

        // Update backtraces in ~I pyramid, adding an INS transdormation
        wfa_backtrace_t I_backtrace = (wfa_backtrace_t)(I_offset_pb & 0xffffffff);
        I_backtrace = (I_backtrace << 2) | OP_INS;
        I_wavefronts[curr_wf]->backtraces[k] = I_backtrace;

        // Update backtraces in the ~M matrix
        const wfa_offset_t X_offset = prev_wf_x->offsets[k] + 1;
        const wfa_backtrace_t X_backtrace = (prev_wf_x->backtraces[k] << 2) | OP_SUB;
        const uint64_t X_offset_pb = ((uint64_t)X_offset << 32) | X_backtrace;

        const uint64_t M_offset_pb = ullmax(X_offset_pb, I_offset_pb);

        const wfa_backtrace_t M_backtrace = (wfa_backtrace_t) \
                                            (M_offset_pb & 0xffffffff);
        M_wavefronts[curr_wf]->backtraces[k] = M_backtrace;

        // Update offsets in the ~M matrix
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);
        M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
        M_wavefronts[curr_wf]->offsets[k] = M_offset;
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf]->hi = hi;
        M_wavefronts[curr_wf]->lo = lo;
        M_wavefronts[curr_wf]->exist = true;

        I_wavefronts[curr_wf]->hi = hi;
        I_wavefronts[curr_wf]->lo = lo;
        I_wavefronts[curr_wf]->exist = true;
    }
}

__device__ void next_MD (wfa_wavefront_t** M_wavefronts,
                         wfa_wavefront_t** D_wavefronts,
                         const int curr_wf,
                         const int x,
                         const int o,
                         const int e,
                         const char* text,
                         const char* pattern,
                         const int tlen,
                         const int plen) {
    const wfa_wavefront_t* prev_wf_x = M_wavefronts[curr_wf - x];
    const wfa_wavefront_t* prev_wf_o = M_wavefronts[curr_wf - o - e];
    const wfa_wavefront_t* prev_wf_e = D_wavefronts[curr_wf - e];

    const int hi = max(prev_wf_x->hi, max(prev_wf_o->hi, prev_wf_e->hi) + 1);
    const int lo = min(prev_wf_x->lo, max(prev_wf_o->lo, prev_wf_e->lo) - 1);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockIdx.x) {
        wfa_offset_t gap_open_offset = prev_wf_o->offsets[k + 1];
        wfa_backtrace_t gap_open_bt = prev_wf_o->backtraces[k + 1];
        // {16 bits unused} | {16 bits offset} | {32 bits backtrace}
        uint64_t gap_open_offset_pb = (uint64_t)
                                ((uint64_t)gap_open_offset << 32) | gap_open_bt;

        wfa_offset_t gap_extend_offset = prev_wf_e->offsets[k + 1];
        wfa_backtrace_t gap_extend_bt = prev_wf_e->backtraces[k + 1];
        uint64_t gap_extend_offset_pb = (uint64_t)
                            ((uint64_t)gap_extend_offset << 32) | gap_extend_bt;

        const uint64_t D_offset_pb = ullmax(gap_open_offset_pb,
                                            gap_extend_offset_pb);

        // Update offsets in the ~D pyramid
        wfa_offset_t D_offset = (wfa_offset_t)(D_offset_pb >> 32);
        D_wavefronts[curr_wf]->offsets[k] = D_offset;

        // Update backtraces in the ~D pyramid
        wfa_backtrace_t D_backtrace = (wfa_backtrace_t)(D_offset_pb & 0xffffffff);
        D_backtrace = (D_backtrace << 2) | OP_DEL;
        D_wavefronts[curr_wf]->backtraces[k] = D_backtrace;

        // Update backtraces in the ~M matrix
        const wfa_offset_t X_offset = prev_wf_x->offsets[k] + 1;
        const wfa_backtrace_t X_backtrace = (prev_wf_x->backtraces[k] << 2) | OP_SUB;
        const uint64_t X_offset_pb = ((uint64_t)X_offset << 32) | X_backtrace;

        const uint64_t M_offset_pb = ullmax(X_offset_pb, D_offset_pb);
        const wfa_backtrace_t M_backtrace = (wfa_backtrace_t) \
                                            (M_offset_pb & 0xffffffff);

        M_wavefronts[curr_wf]->backtraces[k] = M_backtrace;

        // Update offsets in the ~M matrix
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);
        M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
        M_wavefronts[curr_wf]->offsets[k] = M_offset;
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf]->hi = hi;
        M_wavefronts[curr_wf]->lo = lo;
        M_wavefronts[curr_wf]->exist = true;

        D_wavefronts[curr_wf]->hi = hi;
        D_wavefronts[curr_wf]->lo = lo;
        D_wavefronts[curr_wf]->exist = true;
    }
}

__device__ void next_MDI (wfa_wavefront_t** M_wavefronts,
                          wfa_wavefront_t** I_wavefronts,
                          wfa_wavefront_t** D_wavefronts,
                          const int curr_wf,
                          const int x,
                          const int o,
                          const int e,
                          const char* text,
                          const char* pattern,
                          const int tlen,
                          const int plen) {
    const wfa_wavefront_t* prev_wf_x =   M_wavefronts[curr_wf - x];
    const wfa_wavefront_t* prev_wf_o =   M_wavefronts[curr_wf - o - e];
    const wfa_wavefront_t* prev_I_wf_e = I_wavefronts[curr_wf - e];
    const wfa_wavefront_t* prev_D_wf_e = D_wavefronts[curr_wf - e];

    const int hi_ID = max(prev_wf_o->hi, max(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1;
    const int hi    = max(prev_wf_x->hi, hi_ID);
    const int lo_ID = min(prev_wf_o->lo, min(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1;
    const int lo    = min(prev_wf_x->lo, lo_ID);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        // TODO: remove
        const int target_k = EWAVEFRONT_DIAGONAL(tlen, plen);

        // ~I offsets
        const wfa_offset_t I_gap_open_offset = prev_wf_o->offsets[k - 1] + 1;
        const wfa_backtrace_t I_gap_open_bt = prev_wf_o->backtraces[k - 1];
        const int64_t I_gap_open_offset_pb = (int64_t)
                                              ((uint64_t)I_gap_open_offset << 32)
                                              | I_gap_open_bt;

        const wfa_offset_t I_gap_extend_offset = prev_I_wf_e->offsets[k - 1] + 1;
        const wfa_backtrace_t I_gap_extend_bt = prev_I_wf_e->backtraces[k - 1];
        const int64_t I_gap_extend_offset_pb = (int64_t)
                                                ((uint64_t)I_gap_extend_offset << 32)
                                                | I_gap_extend_bt;

        const int64_t I_offset_pb = max(I_gap_open_offset_pb,
                                        I_gap_extend_offset_pb);

        const wfa_offset_t I_offset = (wfa_offset_t)(I_offset_pb >> 32);
        I_wavefronts[curr_wf]->offsets[k] = I_offset;

        // ~I backtraces
        wfa_backtrace_t I_backtrace = (wfa_backtrace_t)(I_offset_pb & 0xffffffff);
        I_backtrace = (I_backtrace << 2) | OP_INS;
        I_wavefronts[curr_wf]->backtraces[k] = I_backtrace;


        // ~D offsets
        const wfa_offset_t D_gap_open_offset = prev_wf_o->offsets[k + 1];
        const wfa_backtrace_t D_gap_open_bt = prev_wf_o->backtraces[k + 1];
        const int64_t D_gap_open_offset_pb = (int64_t)
                                              ((uint64_t)D_gap_open_offset << 32)
                                              | D_gap_open_bt;

        const wfa_offset_t D_gap_extend_offset = prev_D_wf_e->offsets[k + 1];
        const wfa_backtrace_t D_gap_extend_bt = prev_D_wf_e->backtraces[k + 1];
        const int64_t D_gap_extend_offset_pb = (int64_t)
                                                ((uint64_t)D_gap_extend_offset << 32)
                                                | D_gap_extend_bt;

        const int64_t D_offset_pb = max(D_gap_open_offset_pb,
                                        D_gap_extend_offset_pb);

        const wfa_offset_t D_offset = (wfa_offset_t)(D_offset_pb >> 32);
        D_wavefronts[curr_wf]->offsets[k] = D_offset;

        //if (blockIdx.x == 0 && k == 0 ) printf("I_OFFSET: %d\n", I_offset);
        //if (blockIdx.x == 0 && k == 0) printf("D_OFFSET: %d\n", D_offset);

        // ~D backtraces
        wfa_backtrace_t D_backtrace = (wfa_backtrace_t)(D_offset_pb & 0xffffffff);
        D_backtrace = (D_backtrace << 2) | OP_DEL;
        D_wavefronts[curr_wf]->backtraces[k] = D_backtrace;

        // ~M update
        const wfa_offset_t X_offset = prev_wf_x->offsets[k] + 1;
        //if (blockIdx.x == 0 && k == 0 ) printf("X_OFFSET: %d\n", X_offset);
        const wfa_backtrace_t X_backtrace = (prev_wf_x->backtraces[k] << 2) | OP_SUB;
        const int64_t X_offset_pb = (int64_t)
                                     (((uint64_t)X_offset << 32) | X_backtrace);
        //if (blockIdx.x == 0 && k == 0) printf("X_OFFSET_pb: %lld\n", X_offset_pb);

        const int64_t M_offset_pb = max(
                                        max(X_offset_pb, D_offset_pb),
                                        I_offset_pb
                                        );

        const wfa_backtrace_t M_backtrace = (wfa_backtrace_t)
                                            (M_offset_pb & 0xffffffff);
        M_wavefronts[curr_wf]->backtraces[k] = M_backtrace;
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);

        int prev = M_offset;
        //if (blockIdx.x == 0 && k == target_k ) printf("OFFSET: %llu\n", M_offset_pb);
        //if (blockIdx.x == 0 && k == target_k ) printf("MID (k=%d) before_ext: %d, ", k, M_offset);
        //if (blockIdx.x == 0 && k == 0 ) printf("M_OFFSET: %d\n", M_offset);
        M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
        //if (blockIdx.x == 0 && k == 0 ) printf("M_OFFSET (ext): %d (+%d)\n", M_offset, M_offset - prev);
        //if (blockIdx.x == 0 && k == target_k) printf("after_ext: %d (+%d)\n", M_offset, M_offset - prev);

        M_wavefronts[curr_wf]->offsets[k] = M_offset;
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf]->hi = hi;
        M_wavefronts[curr_wf]->lo = lo;
        M_wavefronts[curr_wf]->exist = true;

        I_wavefronts[curr_wf]->hi = hi;
        I_wavefronts[curr_wf]->lo = lo;
        I_wavefronts[curr_wf]->exist = true;

        D_wavefronts[curr_wf]->hi = hi;
        D_wavefronts[curr_wf]->lo = lo;
        D_wavefronts[curr_wf]->exist = true;
    }
}

__device__ void update_wf_ptr_array (wfa_wavefront_t** M_wavefronts,
                                     wfa_wavefront_t** I_wavefronts,
                                     wfa_wavefront_t** D_wavefronts,
                                     const int active_working_set_size,
                                     const int max_wf_size,
                                     const int curr_wf) {
    const int tid = threadIdx.x;

    // TODO: Assumes blockDim.x > active_working_set_size, maybe double
    // check before launching kernel?
    // TODO: Do this with warp shuffle primitives
    wfa_wavefront_t* m_ptr;
    wfa_wavefront_t* i_ptr;
    wfa_wavefront_t* d_ptr;
    if (tid < active_working_set_size) {
        m_ptr = M_wavefronts[tid];
        i_ptr = I_wavefronts[tid];
        d_ptr = D_wavefronts[tid];
    }

    // Make sure all values are read before writing
    __syncthreads();

    if (tid == 0) {
        M_wavefronts[active_working_set_size - 1] = m_ptr;
        I_wavefronts[active_working_set_size - 1] = i_ptr;
        D_wavefronts[active_working_set_size - 1] = d_ptr;

        m_ptr->exist = false;
        i_ptr->exist = false;
        d_ptr->exist = false;

    }
    else if (tid < active_working_set_size) {
        // From 1 to active_working_set_size - 1
        M_wavefronts[tid - 1] = m_ptr;
        I_wavefronts[tid - 1] = i_ptr;
        D_wavefronts[tid - 1] = d_ptr;
    }

    // Make sure values are written before reseting the "new" wavefront
    __syncthreads();

    // Set new wf to NULL, as new wavefront may be smaller than the
    // previous one
    wfa_offset_t* to_clean = M_wavefronts[curr_wf]->offsets - (max_wf_size/2);

    for (int i=tid; i<max_wf_size; i+=blockDim.x) {
        to_clean[i] = -1;
    }

}

__global__ void alignment_kernel (
                            const char* packed_sequences_buffer,
                            const sequence_pair_t* sequences_metadata,
                            const size_t num_alignments,
                            const int max_steps,
                            const affine_penalties_t penalties,
                            alignment_result_t* results) {
    const int tid = threadIdx.x;
    // m = 0 for WFA
    const int x = penalties.x;
    const int o = penalties.o;
    const int e = penalties.e;

    const sequence_pair_t metadata = sequences_metadata[blockIdx.x];
    const char* text = packed_sequences_buffer + metadata.text_offset_packed;
    const char* pattern = packed_sequences_buffer + metadata.pattern_offset_packed;
    const int tlen = metadata.text_len;
    const int plen = metadata.pattern_len;

    // In shared memory:
    // - Offsets for all the wavefronts
    // - Wavefronts needed to calculate current WF_s, there are 3 "pyramids" so
    //   this number of wavefront is 3 times (WF_{max(o+e, x)} --> WF_s)
    // - 3 arrays of pointers to wavefronts of ~M, ~I and ~D.
    extern __shared__ char sh_mem[];

    // TODO: +1 because of the current wf?
    const int active_working_set_size = max(o+e, x) + 1;
    const int max_wf_size = 2 * max_steps + 1;

    // Offsets and backtraces must be 32 bits aligned to avoid unaligned access
    // errors on the structs
    int offsets_size = active_working_set_size * max_wf_size;
    offsets_size = offsets_size + (4 - (offsets_size % 4));

    int bt_size = active_working_set_size * max_wf_size;
    bt_size = bt_size + (4 - (bt_size % 4));

    wfa_offset_t* M_base = (wfa_offset_t*)sh_mem;
    wfa_offset_t* I_base = M_base + offsets_size;
    wfa_offset_t* D_base = I_base + offsets_size;

    wfa_backtrace_t* M_bt_base = (wfa_backtrace_t*)(D_base + offsets_size);
    wfa_backtrace_t* I_bt_base = M_bt_base + bt_size;
    wfa_backtrace_t* D_bt_base = I_bt_base + bt_size;

    wfa_wavefront_t* M_wavefronts_mem = (wfa_wavefront_t*)(D_bt_base + bt_size);
    wfa_wavefront_t* I_wavefronts_mem = (M_wavefronts_mem + active_working_set_size);
    wfa_wavefront_t* D_wavefronts_mem = (I_wavefronts_mem + active_working_set_size);

    wfa_wavefront_t** M_wavefronts = (wfa_wavefront_t**)
                                        (D_wavefronts_mem
                                         + active_working_set_size);

    wfa_wavefront_t** I_wavefronts = (M_wavefronts
                                      + active_working_set_size);

    wfa_wavefront_t** D_wavefronts = (I_wavefronts
                                      + active_working_set_size);

    // Initialize all wavefronts to -1
    for (int i=tid; i<(offsets_size * 3); i+=blockDim.x) {
        M_base[i] = -1;
    }

    for (int i=tid; i<(bt_size * 3); i+=blockDim.x) {
        M_bt_base[i] = 0;
    }

    // Initialize wavefronts memory
    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts_mem[i].offsets = M_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts_mem[i].backtraces = M_bt_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts_mem[i].hi = 0;
        M_wavefronts_mem[i].lo = 0;
        M_wavefronts_mem[i].exist = false;
        M_wavefronts[i] = M_wavefronts_mem + i;

        I_wavefronts_mem[i].offsets = I_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts_mem[i].backtraces = I_bt_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts_mem[i].hi = 0;
        I_wavefronts_mem[i].lo = 0;
        I_wavefronts_mem[i].exist = false;
        I_wavefronts[i] = I_wavefronts_mem + i;

        D_wavefronts_mem[i].offsets = D_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts_mem[i].backtraces = D_bt_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts_mem[i].hi = 0;
        D_wavefronts_mem[i].lo = 0;
        D_wavefronts_mem[i].exist = false;
        D_wavefronts[i] = D_wavefronts_mem + i;
    }

    // TODO: Needed?
    __syncthreads();

    const int curr_wf = active_working_set_size - 1;

    if (tid == 0) {
        wfa_offset_t initial_ext = WF_extend_kernel(
            text,
            pattern,
            tlen, plen,
            0, 0);
        M_wavefronts[curr_wf]->offsets[0] = initial_ext;
        M_wavefronts[curr_wf]->exist = true;
    }

    __syncthreads();

    // TODO: Change tarket K if we don't start form WF 0 (cooperative strategy)
    const int target_k = EWAVEFRONT_DIAGONAL(tlen, plen);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const wfa_offset_t target_offset = EWAVEFRONT_OFFSET(tlen, plen);

    int distance = 0;
    // steps = number of editions
    int steps = 0;
    // TODO: target_k_abs <= distance or <= steps (?)
    if (!(target_k_abs <= distance && M_wavefronts[curr_wf]->exist && M_wavefronts[curr_wf]->offsets[target_k] == target_offset)) {

        update_wf_ptr_array(
            M_wavefronts,
            I_wavefronts,
            D_wavefronts,
            active_working_set_size,
            max_wf_size,
            curr_wf);

        distance++;

        while (steps < (max_steps - 1)) {

            bool M_exist, I_exist, D_exist;
            if ((distance - o - e) >= 0) {
                I_exist = M_wavefronts[curr_wf - o - e]->exist
                          || I_wavefronts[curr_wf - e]->exist;
                D_exist = M_wavefronts[curr_wf - o - e]->exist
                          || D_wavefronts[curr_wf - e]->exist;
            } else {
                I_exist = false;
                D_exist = false;
            }

            if (D_exist || I_exist) {
                M_exist = true;
            } else {
                if ((distance - x) >= 0) {
                    M_exist = M_wavefronts[curr_wf - x]->exist;
                } else {
                    M_exist = false;
                }
            }

            // TODO: Optimize this if-else nightmare
            if (!I_exist && !D_exist && !M_exist) {
                //if (tid == 0) printf("pass distance!\n");
                distance++;
            } else {
                if (M_exist && !I_exist && !D_exist) {
                    next_M(M_wavefronts, curr_wf, x,
                           text, pattern, tlen, plen);
                } else {
                    if (I_exist && D_exist) {
                        next_MDI(
                            M_wavefronts, I_wavefronts, D_wavefronts, curr_wf,
                            x, o, e,
                            text, pattern, tlen, plen);
                    } else {
                        if (I_exist) {
                            next_MI(
                                M_wavefronts, I_wavefronts, curr_wf,
                                x, o, e,
                                text, pattern, tlen, plen);
                        }
                        if (D_exist) {
                            next_MD(
                                M_wavefronts, D_wavefronts, curr_wf,
                                x, o, e,
                                text, pattern, tlen, plen);
                        }
                    }
                }
                //if (blockIdx.x == 0 && tid == 0) {
                //    printf("id: %d, distance: %d, steps: %d, offset: %d (%d)\n", blockIdx.x, distance, steps, M_wavefronts[curr_wf]->offsets[target_k], target_offset);
                //}

                if (target_k_abs <= distance && M_wavefronts[curr_wf]->exist && M_wavefronts[curr_wf]->offsets[target_k] == target_offset) {
                    break;
                }

                steps++;
                distance++;
            }

#if 0
// DEVELOPING HELP ONLY
            PPRINT_WFS(active_working_set_size,
                       M_wavefronts,
                       distance,
                       (char*)("~M wavefronts"));
            PPRINT_WFS(active_working_set_size,
                       I_wavefronts,
                       distance,
                       (char*)("~I wavefronts"));
            PPRINT_WFS(active_working_set_size,
                       D_wavefronts,
                       distance,
                       (char*)("~D wavefronts"));
            if (tid == 0) printf("---------------------------------------------\n");
#endif

        update_wf_ptr_array(
            M_wavefronts,
            I_wavefronts,
            D_wavefronts,
            active_working_set_size,
            max_wf_size,
            curr_wf);

        }
    }


    if  (tid == 0) {
        //if (blockIdx.x == 0) printf("Alignment: %d, distance: %d, steps: %d\n", blockIdx.x, distance, steps);
        results[blockIdx.x].distance = distance;
    }
}
