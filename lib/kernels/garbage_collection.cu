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
                        const int32_t hi,
                        const int32_t lo,
                        wfa_cell_t* const cells,
                        wfa_backtrace_t* offloaded_buffer
                        ) {
    int tid = threadIdx.x;
    for (int k=lo + tid; k<=hi; k+=blockDim.x) {
        uint4 cell = LOAD_CELL(cells[k]);
        wfa_bt_prev_t bt_prev_idx = UINT4_TO_BT_PREV(cell);
        while (bt_prev_idx != 0) {
            wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

            // This race condition is not important, as all threads do the same
            // operation on the same bit (setting the highest bit to 1)
            bt_prev_idx = prev_bt->prev & 0x7fffffff;
            MARK_BACKTRACE(prev_bt);
        }
    }
}

// Assume number of threads is multiple of 32 (no half warps)
__device__ void fill_bitmap (
                            wfa_cell_t* const cells,
                            wfa_backtrace_t* const offloaded_buffer,
                            const size_t offloaded_buffer_size,
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
    for (int i=tid; i<offloaded_buffer_size; i+=blockDim.x) {
        wfa_backtrace_t* const backtrace = &offloaded_buffer[i];
        const bool is_marked = IS_MARKED(backtrace);
        UNMARK_BACKTRACE(backtrace);

        // Id inside the warp
        const int wtid = tid % 32;
        // Id of the current warp
        const int wid = tid / 32;

        // In marked there is a bitmap of all threads in a warp that have a
        // marked backtrace
        const size_t remaining_elements_of_warp = offloaded_buffer_size - (i - wtid);
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
                            (unsigned long long)__popc(marked));
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
        for (int i=1; i<bitmaps_size; i++) {
            ranks[i] += ranks[i-1];
        }
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
        printf("|");
        for (int j=0; j<25; j++) {
            wfa_bt_prev_t prev = offloaded_buffer[i].prev;
            printf(" %04d |", prev);
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

            printf(" -> %d", bt_prev_idx);
            
            // Now previous backtrace is current backtrace
            //wfa_bt_prev_t bt_curr_idx = bt_prev_idx;
            bt_prev_idx = prev_bt->prev;

            while (bt_prev_idx != 0) {
                printf(" -> %d", bt_prev_idx);

                const wfa_backtrace_t* prev_bt = &offloaded_buffer[bt_prev_idx];

                //bt_curr_idx = bt_prev_idx;
                bt_prev_idx = prev_bt->prev;
            }
        }
        else {
            printf(" -> 0");
        }
        printf("\n");
    }
}
