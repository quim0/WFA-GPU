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

#include <stdbool.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "sequence_distance_kernel.cuh"
#include "common_alignment_kernels.cuh"

__forceinline__
__device__ static wfa_offset_t get_offset (const wfa_distance_wavefront_t* const wf,
                                    int k,
                                    const size_t half_num_sh_offsets_per_wf) {
    const int limit = half_num_sh_offsets_per_wf;
    const int is_global = (k > limit) || (k < -limit);
    // predicate k operations
    k += limit * (k < -limit);
    k -= limit * (k > limit);
    // offsets[0] -> ptr to shared memory
    // offsets[1] -> ptr to global memory
    return wf->offsets[is_global][k];
}

__forceinline__
__device__ static void set_offset (wfa_distance_wavefront_t* const wf,
                            int k,
                            const size_t half_num_sh_offsets_per_wf,
                            wfa_offset_t value) {
    const int limit = half_num_sh_offsets_per_wf;
    const int is_global = (k > limit) || (k < -limit);
    // predicate k operations
    k += limit * (k < -limit);
    k -= limit * (k > limit);
    // offsets[0] -> ptr to shared memory
    // offsets[1] -> ptr to global memory
    wf->offsets[is_global][k] = value;
}

__device__ static void next_M (wfa_distance_wavefront_t* M_wavefronts,
                        const int curr_wf,
                        const int active_working_set_size,
                        const int x,
                        const char* text,
                        const char* pattern,
                        const int tlen,
                        const int plen,
                        const size_t half_num_sh_offsets_per_wf,
                        int* band_hi,
                        int* band_lo,
                        const int d) {
    // The wavefront do not grow in case of mismatch
    const wfa_distance_wavefront_t* prev_wf = &M_wavefronts[(curr_wf + x) % active_working_set_size];

    const int hi = MIN(prev_wf->hi, *band_hi);
    const int lo = MAX(prev_wf->lo, *band_lo);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        wfa_offset_t curr_offset = get_offset(prev_wf, k, half_num_sh_offsets_per_wf) + 1;

        // Only extend and update backtrace if the previous offset exist
        if (curr_offset >= 0) {
            curr_offset = WF_extend_kernel(text, pattern,
                                           tlen, plen, k, curr_offset);
        }

        set_offset(&M_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf, curr_offset);
    }

    // Needed for the band, TODO: Do only one sync (now one is done here,
    // another on the main while loop)
    __syncthreads();

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist= true;

#if 0
        // Update band
        if (hi >= *band_hi && lo <= *band_lo && (d % 10 == 0)) {
            const int quarter = (hi - lo) / 4;
            const wfa_offset_t hi_candidate = get_offset(
                    &M_wavefronts[curr_wf],
                    hi - quarter,
                    half_num_sh_offsets_per_wf
                    );
            const wfa_offset_t lo_candidate = get_offset(
                    &M_wavefronts[curr_wf],
                    lo + quarter,
                    half_num_sh_offsets_per_wf
                    );
            if (lo_candidate < hi_candidate) {
                (*band_hi)++;
                (*band_lo)++;
            } else {
                (*band_hi)--;
                (*band_lo)--;
            }
        }
#endif
    }
}

__device__ static void next_MDI (wfa_distance_wavefront_t* M_wavefronts,
                          wfa_distance_wavefront_t* I_wavefronts,
                          wfa_distance_wavefront_t* D_wavefronts,
                          const int curr_wf,
                          const int active_working_set_size,
                          const int x,
                          const int o,
                          const int e,
                          const char* text,
                          const char* pattern,
                          const int tlen,
                          const int plen,
                          const size_t half_num_sh_offsets_per_wf,
                          int* band_hi,
                          int* band_lo,
                          const int d) {
    wfa_distance_wavefront_t* const prev_wf_x  = &M_wavefronts[(curr_wf + x) % active_working_set_size];
    wfa_distance_wavefront_t* const prev_wf_o  = &M_wavefronts[(curr_wf + o + e) % active_working_set_size];
    wfa_distance_wavefront_t* const prev_I_wf_e = &I_wavefronts[(curr_wf + e) % active_working_set_size];
    wfa_distance_wavefront_t* const prev_D_wf_e = &D_wavefronts[(curr_wf + e) % active_working_set_size];

    const int hi_ID = MIN(MAX(prev_wf_o->hi, MAX(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1, *band_hi);
    const int hi    = MIN(MAX(prev_wf_x->hi, hi_ID), *band_hi);
    const int lo_ID = MAX(MIN(prev_wf_o->lo, MIN(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1, *band_lo);
    const int lo    = MAX(MIN(prev_wf_x->lo, lo_ID), *band_lo);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        // ~I offsets
        const wfa_offset_t I_gap_open_offset = get_offset(prev_wf_o, k - 1, half_num_sh_offsets_per_wf) + 1;
        const wfa_offset_t I_gap_extend_offset = get_offset(prev_I_wf_e, k - 1, half_num_sh_offsets_per_wf) + 1;

        int64_t I_offset = MAX(I_gap_open_offset, I_gap_extend_offset);

        set_offset(&I_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf, I_offset);

        // ~D offsets
        const wfa_offset_t D_gap_open_offset = get_offset(prev_wf_o, k + 1, half_num_sh_offsets_per_wf);

        const wfa_offset_t D_gap_extend_offset = get_offset(prev_D_wf_e, k + 1, half_num_sh_offsets_per_wf);

        int64_t D_offset = MAX(D_gap_open_offset, D_gap_extend_offset);

        set_offset(&D_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf, D_offset);

        // ~M update
        const wfa_offset_t X_offset = get_offset(prev_wf_x, k, half_num_sh_offsets_per_wf) + 1;

        const int64_t M_offset_pb = MAX(MAX((int64_t)X_offset, D_offset), I_offset);

        // Extend
        wfa_offset_t M_offset = (wfa_offset_t)(M_offset_pb >> 32);
        if (M_offset >= 0) {
            M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
        }

        set_offset(&M_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf,  M_offset);
    }

    // Needed for the band, TODO: Do only one sync (now one is done here,
    // another on the main while loop)
    __syncthreads();

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

        // Update band
        if (hi >= *band_hi && lo <= *band_lo && (d % 10 == 0)) {
            const int quarter = (hi - lo) / 4;
            const wfa_offset_t hi_offset = get_offset(
                    &M_wavefronts[curr_wf],
                    hi - quarter,
                    half_num_sh_offsets_per_wf
                    );
            const wfa_offset_t lo_offset = get_offset(
                    &M_wavefronts[curr_wf],
                    lo + quarter,
                    half_num_sh_offsets_per_wf
                    );

            const int hi_distance = compute_distance_to_target(hi_offset,
                                                               hi - quarter,
                                                               plen,
                                                               tlen);
            const int lo_distance = compute_distance_to_target(lo_offset,
                                                               lo + quarter,
                                                               plen,
                                                               tlen);

            if (lo_distance > hi_distance) {
                (*band_hi)++;
                (*band_lo)++;
            } else {
                (*band_hi)--;
                (*band_lo)--;
            }
        }
    }
}

__device__ static void update_curr_wf (wfa_distance_wavefront_t* M_wavefronts,
                                       wfa_distance_wavefront_t* I_wavefronts,
                                       wfa_distance_wavefront_t* D_wavefronts,
                                       const int active_working_set_size,
                                       const int max_wf_size,
                                       int* curr_wf) {
    // As we read wavefronts "forward" in the waveronts arrays, so the wavefront
    // index is moved backwards.
    const int wf_idx = (*curr_wf - 1 + active_working_set_size) % active_working_set_size;
    *curr_wf = wf_idx;

}

__global__ void distance_kernel (
                            const char* packed_sequences_buffer,
                            const sequence_pair_t* sequences_metadata,
                            const size_t num_alignments,
                            const int max_steps,
                            uint8_t* const wf_data_buffer,
                            const affine_penalties_t penalties,
                            alignment_result_t* results,
                            uint32_t* const next_alignment_idx,
                            const size_t num_sh_offsets_per_wf,
                            const int band) {
    const int tid = threadIdx.x;
    // m = 0 for WFA
    const int x = penalties.x;
    const int o = penalties.o;
    const int e = penalties.e;

    // For adaptative band
    __shared__ int band_hi, band_lo;

    __shared__ uint32_t alignment_idx;

    // Get the first alignment to compute from the alignments pool
    // Initialise band
    if (tid == 0) {
        alignment_idx = get_alignment_idx(next_alignment_idx);
        band_hi = band;
        band_lo = -band;
    }

    __syncthreads();

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

    const size_t wf_data_buffer_size = (offsets_size * 3 * sizeof(wfa_offset_t));
    uint8_t* curr_alignment_wf_data_buffer = wf_data_buffer
                                             + (wf_data_buffer_size * blockIdx.x);

    wfa_offset_t* M_base = (wfa_offset_t*)curr_alignment_wf_data_buffer;
    wfa_offset_t* I_base = M_base + offsets_size;
    wfa_offset_t* D_base = I_base + offsets_size;

    // Backtrace vectors
    //bt_vector_t* M_bt_vector_base = (bt_vector_t*)(D_base + offsets_size);
    //bt_vector_t* I_bt_vector_base = M_bt_vector_base + bt_size;
    //bt_vector_t* D_bt_vector_base = I_bt_vector_base + bt_size;

    // Baktrace vector pointers
    //bt_prev_t* M_bt_prev_base = (bt_prev_t*)(D_bt_vector_base + bt_size);
    //bt_prev_t* I_bt_prev_base = M_bt_prev_base + bt_size;
    //bt_prev_t* D_bt_prev_base = I_bt_prev_base + bt_size;

    // Wavefronts structres reside in shared
    wfa_distance_wavefront_t* M_wavefronts = (wfa_distance_wavefront_t*)sh_mem;
    wfa_distance_wavefront_t* I_wavefronts = (M_wavefronts + active_working_set_size);
    wfa_distance_wavefront_t* D_wavefronts = (I_wavefronts + active_working_set_size);

    // TODO: Check this
    wfa_offset_t* M_sh_offsets_base = (wfa_offset_t*)((uint32_t*)(D_wavefronts + active_working_set_size));
    wfa_offset_t* I_sh_offsets_base = M_sh_offsets_base
                                      + (num_sh_offsets_per_wf * active_working_set_size);
    wfa_offset_t* D_sh_offsets_base = I_sh_offsets_base
                                      + (num_sh_offsets_per_wf * active_working_set_size);

    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts[i].offsets[1] = M_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].offsets[0] = M_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);

        I_wavefronts[i].offsets[1] = I_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].offsets[0] = I_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);

        D_wavefronts[i].offsets[1] = D_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].offsets[0] = D_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);
    }

    // Iterate until there are no more alignments in the pool to compute
    while (alignment_idx < num_alignments) {
        const sequence_pair_t curr_batch_alignment_base = sequences_metadata[0];
        const size_t base_offset_packed = curr_batch_alignment_base.pattern_offset_packed;

        const sequence_pair_t metadata = sequences_metadata[alignment_idx];
        // Sequences with "N" chatacters not supported yet
        if (metadata.has_N) {
            if (tid == 0) {
                results[alignment_idx].distance = 0;
                results[alignment_idx].finished = false;
                alignment_idx = get_alignment_idx(next_alignment_idx);
            }
            __syncthreads();
            continue;
        }

        const char* text = packed_sequences_buffer + metadata.text_offset_packed - base_offset_packed;
        const char* pattern = packed_sequences_buffer + metadata.pattern_offset_packed - base_offset_packed ;
        const int tlen = metadata.text_len;
        const int plen = metadata.pattern_len;

        // Initialize all wavefronts to NULL
        for (int i=tid; i<(offsets_size * 3); i+=blockDim.x) {
#if __CUDA_ARCH__ >= 800
            __stwt(&M_base[i], OFFSET_NULL);
#else
            M_base[i] = OFFSET_NULL;
#endif
        }
        // Shared mem
        for (int i=tid; i<(num_sh_offsets_per_wf * active_working_set_size * 3); i+=blockDim.x) {
            M_sh_offsets_base[i] =  OFFSET_NULL;
        }

        for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
            M_wavefronts[i].hi = 0;
            M_wavefronts[i].lo = 0;
            M_wavefronts[i].exist = false;

            I_wavefronts[i].hi = 0;
            I_wavefronts[i].lo = 0;
            I_wavefronts[i].exist = false;

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
            //M_wavefronts[curr_wf].offsets[0] = initial_ext;
            set_offset(&M_wavefronts[curr_wf], 0, num_sh_offsets_per_wf/2, initial_ext);
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
        if (!(target_k_abs <= distance && M_wavefronts[curr_wf].exist && get_offset(&M_wavefronts[curr_wf], target_k, num_sh_offsets_per_wf/2) == target_offset)) {

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
                               text, pattern, tlen, plen, num_sh_offsets_per_wf/2,
                               &band_hi, &band_lo, distance);
                        D_wavefronts[curr_wf].exist = false;
                        I_wavefronts[curr_wf].exist = false;
                    } else {
                        next_MDI(
                            M_wavefronts, I_wavefronts, D_wavefronts,
                            curr_wf, active_working_set_size,
                            x, o, e,
                            text, pattern, tlen, plen, num_sh_offsets_per_wf/2,
                            &band_hi, &band_lo, distance);

                        // Wavefront only grows if there's an operation in the ~I or
                        // ~D matrices
                        steps++;
                    }

                    // TODO: This is necessary for now, try to find a less sync
                    // version
                    __syncthreads();

                    if (target_k_abs <= distance && M_exist && get_offset(&M_wavefronts[curr_wf], target_k, num_sh_offsets_per_wf/2) == target_offset) {
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
            results[alignment_idx].distance = distance;
            results[alignment_idx].finished = finished;
            results[alignment_idx].num_bt_blocks = 0;
        }
        // Get next alignment
        if (tid == 0) alignment_idx = get_alignment_idx(next_alignment_idx);
        __syncthreads();
    } // while
}
