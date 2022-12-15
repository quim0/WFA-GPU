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
#include "sequence_distance_kernel_aband.cuh"
#include "common_alignment_kernels.cuh"

__forceinline__
__device__ static wfa_offset_t get_offset (const wfa_distance_aband_wavefront_t* const wf,
                                    int k) {
    if (k > wf->hi || k < wf->lo) return OFFSET_NULL;
    return wf->offsets[k - wf->lo];
}

__forceinline__
__device__ static void set_offset (wfa_distance_aband_wavefront_t* const wf,
                            int k,
                            wfa_offset_t value) {
    if (k > wf->hi || k < wf->lo) {
        return;
    }
    wf->offsets[k - wf->lo] = value;
}

__device__ static void next_M (wfa_distance_aband_wavefront_t* M_wavefronts,
                        const int curr_wf,
                        const int active_working_set_size,
                        const int x,
                        const char* text,
                        const char* pattern,
                        const int tlen,
                        const int plen,
                        const int d) {
    // The wavefront do not grow in case of mismatch
    const wfa_distance_aband_wavefront_t* prev_wf = &M_wavefronts[(curr_wf + x) % active_working_set_size];

    const int hi = prev_wf->hi;
    const int lo = prev_wf->lo;

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        wfa_offset_t curr_offset = get_offset(prev_wf, k) + 1;

        // Only extend and update backtrace if the previous offset exist
        if (curr_offset >= 0) {
            curr_offset = WF_extend_kernel(text, pattern,
                                           tlen, plen, k, curr_offset);
        }

        set_offset(&M_wavefronts[curr_wf], k, curr_offset);
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist= true;
    }
}

__device__ static void next_MDI (wfa_distance_aband_wavefront_t* M_wavefronts,
                          wfa_distance_aband_wavefront_t* I_wavefronts,
                          wfa_distance_aband_wavefront_t* D_wavefronts,
                          const int curr_wf,
                          const int active_working_set_size,
                          const int x,
                          const int o,
                          const int e,
                          const char* text,
                          const char* pattern,
                          const int tlen,
                          const int plen,
                          const int band,
                          const int num_sh_offsets_per_wf,
                          const int d) {
    wfa_distance_aband_wavefront_t* const prev_wf_x  = &M_wavefronts[(curr_wf + x) % active_working_set_size];
    wfa_distance_aband_wavefront_t* const prev_wf_o  = &M_wavefronts[(curr_wf + o + e) % active_working_set_size];
    wfa_distance_aband_wavefront_t* const prev_I_wf_e = &I_wavefronts[(curr_wf + e) % active_working_set_size];
    wfa_distance_aband_wavefront_t* const prev_D_wf_e = &D_wavefronts[(curr_wf + e) % active_working_set_size];

    const int hi_ID = MAX(prev_wf_o->hi, MAX(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1 ;
    int hi    = MAX(prev_wf_x->hi, hi_ID);
    const int lo_ID = MIN(prev_wf_o->lo, MIN(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1;
    int lo    = MIN(prev_wf_x->lo, lo_ID);

    while ((hi-lo) > num_sh_offsets_per_wf-1) {
        hi--;
        if ((hi-lo) <= num_sh_offsets_per_wf-1) break;
        lo++;
    }

    // TODO: make cooperative
    const int prev_lo = prev_wf_x->lo;
    const int prev_hi = prev_wf_x->hi;

    if (((prev_hi-prev_lo) >= num_sh_offsets_per_wf-1) && (d % band) == 0) {
        // start with current center
        int bmind = 2 * (tlen + plen);
        int new_center = prev_lo;
        for (int i=prev_lo; i<prev_hi; i++) {
            wfa_offset_t boffset = get_offset( prev_wf_x, i);
            int d_to_target = compute_distance_to_target(boffset, i, plen, tlen);
            if (d_to_target < bmind) {
                bmind = d_to_target;
                new_center = i;
            }

        }
        // The offset with hte minimum distance gets centered
        lo = new_center - (num_sh_offsets_per_wf/2);
        hi = lo + num_sh_offsets_per_wf - 1;
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist = true;

        I_wavefronts[curr_wf].hi = hi;
        I_wavefronts[curr_wf].lo = lo;
        I_wavefronts[curr_wf].exist = true;

        D_wavefronts[curr_wf].hi = hi;
        D_wavefronts[curr_wf].lo = lo;
        D_wavefronts[curr_wf].exist = true;

    }

    __syncthreads();

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        // ~I offsets
        const wfa_offset_t I_gap_open_offset = get_offset(prev_wf_o, k - 1) + 1;
        const wfa_offset_t I_gap_extend_offset = get_offset(prev_I_wf_e, k - 1) + 1;

        int64_t I_offset = MAX(I_gap_open_offset, I_gap_extend_offset);

        set_offset(&I_wavefronts[curr_wf], k, I_offset);

        // ~D offsets
        const wfa_offset_t D_gap_open_offset = get_offset(prev_wf_o, k + 1);

        const wfa_offset_t D_gap_extend_offset = get_offset(prev_D_wf_e, k + 1);

        int64_t D_offset = MAX(D_gap_open_offset, D_gap_extend_offset);

        set_offset(&D_wavefronts[curr_wf], k, D_offset);

        // ~M update
        const wfa_offset_t X_offset = get_offset(prev_wf_x, k) + 1;

        wfa_offset_t M_offset = MAX(MAX((int64_t)X_offset, D_offset), I_offset);

        // Extend
        if (M_offset >= 0) {
            M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
        }

        set_offset(&M_wavefronts[curr_wf], k, M_offset);
    }
}

__device__ static void update_curr_wf (wfa_distance_aband_wavefront_t* M_wavefronts,
                                       wfa_distance_aband_wavefront_t* I_wavefronts,
                                       wfa_distance_aband_wavefront_t* D_wavefronts,
                                       const int active_working_set_size,
                                       const int max_wf_size,
                                       int* curr_wf) {
    // As we read wavefronts "forward" in the waveronts arrays, so the wavefront
    // index is moved backwards.
    const int wf_idx = (*curr_wf - 1 + active_working_set_size) % active_working_set_size;
    *curr_wf = wf_idx;

}

__global__ void distance_kernel_aband (
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

    __shared__ uint32_t alignment_idx;

    // Get the first alignment to compute from the alignments pool
    // Initialise band
    if (tid == 0) {
        alignment_idx = get_alignment_idx(next_alignment_idx);
    }

    __syncthreads();

    extern __shared__ char sh_mem[];

    const int active_working_set_size = MAX(o+e, x) + 1;
    const int max_wf_size = 2 * max_steps + 1;

    // Offsets and backtraces must be 32 bits aligned to avoid unaligned access
    // errors on the structs
    int offsets_size = active_working_set_size * max_wf_size;
    offsets_size = offsets_size + (4 - (offsets_size % 4));

    const size_t wf_data_buffer_size = (offsets_size * 3 * sizeof(wfa_offset_t));
    uint8_t* curr_alignment_wf_data_buffer = wf_data_buffer
                                             + (wf_data_buffer_size * blockIdx.x);

    // Wavefronts structres reside in shared
    wfa_distance_aband_wavefront_t* M_wavefronts = (wfa_distance_aband_wavefront_t*)sh_mem;
    wfa_distance_aband_wavefront_t* I_wavefronts = (M_wavefronts + active_working_set_size);
    wfa_distance_aband_wavefront_t* D_wavefronts = (I_wavefronts + active_working_set_size);

    wfa_offset_t* M_sh_offsets_base = (wfa_offset_t*)((uint32_t*)(D_wavefronts + active_working_set_size));
    wfa_offset_t* I_sh_offsets_base = M_sh_offsets_base
                                      + (num_sh_offsets_per_wf * active_working_set_size);
    wfa_offset_t* D_sh_offsets_base = I_sh_offsets_base
                                      + (num_sh_offsets_per_wf * active_working_set_size);

    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts[i].offsets = M_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf);
        I_wavefronts[i].offsets = I_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf);
        D_wavefronts[i].offsets = D_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf);
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
            set_offset(&M_wavefronts[curr_wf], 0, initial_ext);
            M_wavefronts[curr_wf].exist = true;
        }

        __syncthreads();

        const int target_k = EWAVEFRONT_DIAGONAL(tlen, plen);
        const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
        const wfa_offset_t target_offset = EWAVEFRONT_OFFSET(tlen, plen);

        bool finished = false;

        int distance = 0;
        // steps = number of editions
        int steps = 0;
        if (!(target_k_abs <= distance && M_wavefronts[curr_wf].exist && get_offset(&M_wavefronts[curr_wf], target_k) == target_offset)) {

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
                               text, pattern, tlen, plen, distance);
                        D_wavefronts[curr_wf].exist = false;
                        I_wavefronts[curr_wf].exist = false;
                    } else {
                        next_MDI(
                            M_wavefronts, I_wavefronts, D_wavefronts,
                            curr_wf, active_working_set_size,
                            x, o, e,
                            text, pattern, tlen, plen, band, num_sh_offsets_per_wf,
                            distance);

                        // Wavefront only grows if there's an operation in the ~I or
                        // ~D matrices
                        steps++;
                    }

                    __syncthreads();

                    if (target_k_abs <= distance && M_exist) {
                        if (get_offset(&M_wavefronts[curr_wf], target_k) == target_offset) {
                            finished = true;
                            break;
                        } else if (get_offset(&M_wavefronts[curr_wf], target_k) > target_offset) {
                            finished = false;
                            break;
                        }
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
