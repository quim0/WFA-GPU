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
#include "garbage_collection.cuh"

#define FULL_MASK 0xffffffff

__device__ void mark_offsets (
                        wfa_wavefront_t* const M_wavefronts,
                        wfa_wavefront_t* const I_wavefronts,
                        wfa_wavefront_t* const D_wavefronts,
                        const int active_working_set_size,
                        wfa_backtrace_t* offloaded_buffer
                        ) {
    int tid = threadIdx.x;

    // Mark all chains from all wavefronts in the active working set
    for (int wf_idx=0; wf_idx<active_working_set_size; wf_idx++) {
        // M wavefront
        int hi = M_wavefronts[wf_idx].hi;
        int lo = M_wavefronts[wf_idx].lo;
        wfa_cell_t* cells = M_wavefronts[wf_idx].cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);
            wfa_offset_t offset = UINT4_TO_OFFSET(cell);
            if (offset < 0) continue;
            while (bt_prev_idx != 0) {
                wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

                // This race condition is not important, as all threads do the same
                // operation on the same bit (setting the highest bit to 1)
                bt_prev_idx = prev_bt->prev & 0x7fffffffU;
                MARK_BACKTRACE(prev_bt);
            }
        }

        // I wavefront
        hi = I_wavefronts[wf_idx].hi;
        lo = I_wavefronts[wf_idx].lo;
        cells = I_wavefronts[wf_idx].cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);
            wfa_offset_t offset = UINT4_TO_OFFSET(cell);
            if (offset < 0) continue;
            while (bt_prev_idx != 0) {
                wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

                // This race condition is not important, as all threads do the same
                // operation on the same bit (setting the highest bit to 1)
                bt_prev_idx = prev_bt->prev & 0x7fffffffU;
                MARK_BACKTRACE(prev_bt);
            }
        }

        // D wavefront
        hi = D_wavefronts[wf_idx].hi;
        lo = D_wavefronts[wf_idx].lo;
        cells = D_wavefronts[wf_idx].cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);
            wfa_offset_t offset = UINT4_TO_OFFSET(cell);
            if (offset < 0) continue;
            while (bt_prev_idx != 0) {
                wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

                // This race condition is not important, as all threads do the same
                // operation on the same bit (setting the highest bit to 1)
                bt_prev_idx = prev_bt->prev & 0x7fffffffU;
                MARK_BACKTRACE(prev_bt);
            }
        }
    }
}

// Assume number of threads is multiple of 32 (no half warps)
__device__ void fill_bitmap (
                            wfa_cell_t* const cells,
                            wfa_backtrace_t* const offloaded_buffer,
                            const size_t offloaded_buffer_size,
                            const size_t last_free_bt_position,
                            const size_t bitmaps_size,
                            wfa_bitmap_t* const bitmaps,
                            wfa_rank_t* const ranks) {
    const int tid = threadIdx.x;

    //if (tid == 0) printf("bitmap_elements = %llu\n", bitmaps_size); __syncthreads();

    // Set ranks to 0
    for (int i=tid; i<bitmaps_size; i+=blockDim.x) {
        //printf("thread %i accessing position %d\n", tid, i);
        ranks[i] = 0;
    }

    __syncthreads();

    // Set the marked backtraces on the offloaded buffer in the bitvectors, and
    // partially update ranks
    // Start from position 1, as position zero of the buffer is never used (used
    // as delimiter for the backtrace chains)
    for (int i=tid+1; i<last_free_bt_position; i+=blockDim.x) {
        wfa_backtrace_t* const backtrace = &offloaded_buffer[i];
        const bool is_marked = IS_MARKED(backtrace);
        UNMARK_BACKTRACE(backtrace);

        // Id inside the warp
        const int wtid = tid % 32;
        // Id of the current warp
        const int wid = tid / 32;

        // In marked there is a bitmap of all threads in a warp that have a
        // marked backtrace
        const size_t remaining_elements_of_warp = last_free_bt_position - (i - wtid);
        unsigned mask = FULL_MASK;
        if (remaining_elements_of_warp < 32) {
            // First threads of the warp is on the lower bits, so lane_id=0 is
            // represented by the lower bit instead of the higher one
            mask >>= (32 - remaining_elements_of_warp);
        }
        const unsigned marked = __ballot_sync(mask, is_marked);

        // marked is a bitmap that contains a 1 if the lane at that position has
        // marked the backtrace, and a 0 otherwise. The lower bit contains the
        // first lane result, and the highest bit contains the last lane result.

        if (wtid == 0) {
            uint32_t bitmap_idx = GET_BITMAP_IDX(i);
            wfa_bitmap_t* curr_bitmap = &bitmaps[bitmap_idx];
            wfa_rank_t* curr_rank = &ranks[bitmap_idx];

            // Warps are 32 threads, but bitmaps have 64 elements. Reinterpret
            // the bitmaps as if they had 32 elements to avoid atomic
            // operations. Warps with even id will have the higher part of the
            // bitmaps, while warps with odd id will have the lowest part of the
            // bitmaps.
            uint32_t* curr_bitmap_32 = ((uint32_t*)curr_bitmap) + (wid % 2);
            *curr_bitmap_32 = marked;

            // Warps access the rank in groups of two: warps 0 and 1 access
            // position 0, warps 2 and 3 access position 1... etc
            atomicAdd_block((unsigned long long*)curr_rank,
                            (unsigned long long)MARKED_POPC(marked));
            //printf("warp %d, bitmap_idx=%d, popc=%llu\n", wid, bitmap_idx, (unsigned long long)MARKED_POPC(marked));
        }
    }

    __syncthreads();

    // At this point, ranks contains the number of elements only in its
    // corresponding bitmap.
    // Make the reduction of ranks (prefix sum the array)
    // TODO: Implement a thread cooperative strategy, this is just to test if it
    // works
    // XXX: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu#L56
    if (tid == 0) {
        unsigned last_rank_idx = GET_BITMAP_IDX((last_free_bt_position - 1)) + 1;
        last_rank_idx++;
        for (int i=0; i<last_rank_idx; i++) {
            ranks[i] += (i == 0) ? 0 : ranks[i-1];
        }
    }

    __syncthreads();
}


__device__ void clean_offloaded_offsets (
                                    wfa_backtrace_t* const src_offloaded_buffer,
                                    wfa_backtrace_t* const dst_offloaded_buffer,
                                    const size_t offloaded_buffer_size,
                                    const size_t bitmaps_size,
                                    uint32_t* const last_free_bt_position,
                                    const wfa_bitmap_t* const bitmaps,
                                    const wfa_rank_t* const ranks,
                                    wfa_wavefront_t* const M_wavefronts,
                                    wfa_wavefront_t* const I_wavefronts,
                                    wfa_wavefront_t* const D_wavefronts,
                                    const int active_working_set_size) {
    // TODO: Change implementation so it does one iteration on each marked
    // block, instead of one iteration per bitvector ??
    unsigned last_rank_idx = GET_BITMAP_IDX((*last_free_bt_position) - 1) + 1;
    int tid = threadIdx.x;

    for (int i=0+tid; i<last_rank_idx; i+=blockDim.x) {
        wfa_rank_t rank = (i == 0) ? 0 : ranks[i-1];
        wfa_bitmap_t bitmap = bitmaps[i];
        int num_set = BITMAP_POPC(bitmap);

        unsigned curr_base_idx = i * bitmap_size_bits;

        int first_set_idx = BITMAP_FFS(bitmap);
        // TODO: Check if this should really be 0 or -1
        int backtrace_delta = 0;
        for (int j=0; j<num_set; j++) {
            // Position where this backtrace vector will be moved
            const int to = rank + j + 1;

            // Position where this backtrace vector "prev" will point
            // first_set_idx starts at index 1 instead of zero,
            // backtrace_delta is initialized at 0 anyway as the first element
            // of the offloaded backtrace is "NULL" and not counted
            backtrace_delta += first_set_idx;

            wfa_backtrace_t backtrace = src_offloaded_buffer[curr_base_idx + backtrace_delta];
            wfa_bt_prev_t prev = backtrace.prev;

            int link;
            if (prev > 0) {
                int prev_bitmap_idx = GET_BITMAP_IDX(prev);
                wfa_bitmap_t prev_bitmap = bitmaps[prev_bitmap_idx];
                wfa_rank_t prev_rank = (prev_bitmap_idx == 0) ? 0 : ranks[prev_bitmap_idx - 1];

                int prev_bitmap_delta = prev % bitmap_size_bits;
                // Count how many set bits there are until the target bit
                // (including the target bit)
                prev_bitmap <<= (bitmap_size_bits - prev_bitmap_delta);
                link = prev_rank + BITMAP_POPC(prev_bitmap);
            } else {
                // End of chain, link to 0
                link = 0;
            }


            //printf("moving from %d to %d (link from %d to %d)\n",
            //       curr_base_idx + backtrace_delta, to, backtrace.prev, link);
            backtrace.prev = link;
            dst_offloaded_buffer[to] = backtrace;
            
            // Remove the bits that we already consumed
            bitmap >>= first_set_idx;
            first_set_idx = BITMAP_FFS(bitmap);
        }
    }

    // XXX: This can be removed (?)
    __syncthreads();

    for (int wf_idx=0; wf_idx<active_working_set_size; wf_idx++) {
        // Update ~M wavefront
        wfa_wavefront_t curr_wf = M_wavefronts[wf_idx];
        int hi = curr_wf.hi;
        int lo = curr_wf.lo;
        wfa_cell_t* cells = curr_wf.cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t prev = UINT4_TO_BT_PREV(cell);

            wfa_bt_prev_t link;
            if (prev > 0) {
                int prev_bitmap_idx = GET_BITMAP_IDX(prev);
                wfa_bitmap_t prev_bitmap = bitmaps[prev_bitmap_idx];
                wfa_rank_t prev_rank = ranks[prev_bitmap_idx - 1];

                int prev_bitmap_delta = prev % bitmap_size_bits;
                // Count how many set bits there are until the target bit
                // (including the target bit)
                prev_bitmap <<= (bitmap_size_bits - prev_bitmap_delta);
                link = prev_rank + BITMAP_POPC(prev_bitmap);
                //if (link == 1359) printf("LINK wf=%d, k=%d, prev=%d, prev_bitmap_idx=%d, prev_bitmap_delta=%d\n", wf_idx, k,
                //    prev, prev_bitmap_idx, prev_bitmap_delta);
                
            } else link = 0;


            STORE_CELL(cells[k],
                       UINT4_TO_OFFSET(cell),
                       UINT4_TO_BT_VECTOR(cell),
                       link);
        }

        // Update ~I wavefront
        curr_wf = I_wavefronts[wf_idx];
        hi = curr_wf.hi;
        lo = curr_wf.lo;
        cells = curr_wf.cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t prev = UINT4_TO_BT_PREV(cell);

            wfa_bt_prev_t link;
            if (prev > 0) {
                int prev_bitmap_idx = GET_BITMAP_IDX(prev);
                wfa_bitmap_t prev_bitmap = bitmaps[prev_bitmap_idx];
                wfa_rank_t prev_rank = ranks[prev_bitmap_idx - 1];

                int prev_bitmap_delta = prev % bitmap_size_bits;
                prev_bitmap <<= (bitmap_size_bits - prev_bitmap_delta);
                link = prev_rank + BITMAP_POPC(prev_bitmap);
                
            } else link = 0;


            STORE_CELL(cells[k],
                       UINT4_TO_OFFSET(cell),
                       UINT4_TO_BT_VECTOR(cell),
                       link);
        }

        // Update ~D wavefront
        curr_wf = D_wavefronts[wf_idx];
        hi = curr_wf.hi;
        lo = curr_wf.lo;
        cells = curr_wf.cells;
        for (int k=lo+tid; k<=hi; k+=blockDim.x) {
            uint4 cell = LOAD_CELL(cells[k]);
            wfa_bt_prev_t prev = UINT4_TO_BT_PREV(cell);

            wfa_bt_prev_t link;
            if (prev > 0) {
                int prev_bitmap_idx = GET_BITMAP_IDX(prev);
                wfa_bitmap_t prev_bitmap = bitmaps[prev_bitmap_idx];
                wfa_rank_t prev_rank = ranks[prev_bitmap_idx - 1];

                int prev_bitmap_delta = prev % bitmap_size_bits;
                prev_bitmap <<= (bitmap_size_bits - prev_bitmap_delta);
                link = prev_rank + BITMAP_POPC(prev_bitmap);
                
            } else link = 0;


            STORE_CELL(cells[k],
                       UINT4_TO_OFFSET(cell),
                       UINT4_TO_BT_VECTOR(cell),
                       link);
        }
    }

    if (threadIdx.x == 0) {
        wfa_rank_t last_rank = ranks[last_rank_idx - 1];
        *last_free_bt_position = last_rank + 1;
    }
    __syncthreads();
}


#if 0
// This implementation does not take into account that multiple blocks can point
// to the same "previous" block.
__device__ void clean_offloaded_backtraces_buffer (
                                const int32_t hi,
                                const int32_t lo,
                                uint32_t* last_free_idx,
                                wfa_cell_t* cells,
                                wfa_backtrace_t* offloaded_buffer_src,
                                wfa_backtrace_t* offloaded_buffer_dst) {
    int tid = threadIdx.x;
    // TODO: Remove
    uint32_t original = *last_free_idx;
    if (tid==0) printf("Garbage collecting!!! src=%p, dst=%p\n",
        offloaded_buffer_src, offloaded_buffer_dst);
    __syncthreads();

    if (tid == 0) *last_free_idx = 1;
    __syncthreads();

    if (tid == 33) printf("last_free_idx=%d\n", *last_free_idx);

    for (int k=lo + tid; k<=hi; k+=blockDim.x) {
        int blocks = 0;

        uint4 cell = LOAD_CELL(cells[k]);

        // Get current backtrace
        wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);
        if (bt_prev_idx != 0) {
            blocks++;
            // Get previous backtrace
            wfa_backtrace_t* prev_bt = &offloaded_buffer_src[bt_prev_idx];
            
            // Store previous backtrace in the new buffer
            int old_val = atomicAdd_block(last_free_idx, 1);
            offloaded_buffer_dst[old_val] = *prev_bt;

            // Save previous backtrace address in current backtrace
            // This is uncoalesced, but only executed once
            cells[k].bt_prev = old_val;

            // Now previous backtrace is current backtrace
            wfa_bt_prev_t bt_curr_idx = old_val;
            bt_prev_idx = prev_bt->prev;

            int idx = 1;

            while (bt_prev_idx != 0) {
                blocks++;

                // curr_Get previous backtrace
                wfa_backtrace_t* prev_bt = &offloaded_buffer_src[bt_prev_idx];

                // Store previous backtrace in the new buffer
                old_val = atomicAdd_block(last_free_idx, 1);
                offloaded_buffer_dst[old_val] = *prev_bt;
                if (old_val > original && idx == 1) {
                    //printf("old_val > original! k=%d, id=%d\n", k, idx);
                }

                // Save previous backtrace address in current backtrace
                wfa_backtrace_t* curr_bt = &offloaded_buffer_dst[bt_curr_idx];
                curr_bt->prev = old_val;

                // Now previous backtrace is current backtrace
                bt_prev_idx = prev_bt->prev;
                bt_curr_idx = old_val;

                idx++;
            }

            // Mark the limit of the chain
            wfa_backtrace_t* curr_bt = &offloaded_buffer_dst[bt_curr_idx];
            curr_bt->prev = 0;
            //if (tid % 32 == 0) printf("k=%d, blocks=%d\n", k, blocks);
        }
    }

    __syncthreads();

    if (tid == 0) printf("previous elements=%d, new elements=%d, freed element=%d\n", original, *last_free_idx, original - *last_free_idx);

    __syncthreads();
}

#endif

__device__ void pprint_offloaded_backtraces_buffer (
                                        wfa_backtrace_t* offloaded_buffer,
                                        const uint32_t* last_free_idx) {
    for (int i=0; i<*last_free_idx;) {
        printf("%04d: |", i);
        for (int j=0; j<5; j++) {
            wfa_backtrace_t* bt = &offloaded_buffer[i];
            wfa_bt_prev_t prev = bt->prev & 0x7fffffffU;
            bool marked = IS_MARKED(bt);
            char m = ' ';
            if (marked) m = 'X';
            printf(" [%04d][%04d][%c] |", i, prev, m);
            i++;
        }
        printf("\n");
    }
}

__device__ void pprint_offloaded_backtraces_chains (
                                        const wfa_backtrace_t* offloaded_buffer,
                                        const wfa_cell_t* cells,
                                        const int32_t hi,
                                        const int32_t lo,
                                        const uint32_t* last_free_idx) {
    for (int k=lo; k<=hi; k++) {
        uint4 cell = LOAD_CELL(cells[k]);
        wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);

        printf("k=%d, cell", k);

        if (bt_prev_idx != 0) {
            // Get previous backtrace
            const wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

            printf(" -> %d (0x%llx) ", bt_prev_idx, prev_bt->backtrace);
            
            // Now previous backtrace is current backtrace
            //wfa_bt_prev_t bt_curr_idx = bt_prev_idx;
            bt_prev_idx = prev_bt->prev & 0x7fffffffU;

            while (bt_prev_idx != 0) {
                printf(" -> %d (0x%llx) ", bt_prev_idx, prev_bt->backtrace);

                const wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

                //bt_curr_idx = bt_prev_idx;
                bt_prev_idx = prev_bt->prev & 0x7fffffffU;
            }

            printf(" -> %d", bt_prev_idx);
        }
        else {
            printf(" -> 0");
        }
        printf("\n");
    }
}
