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

#ifndef GARBARGE_COLLECTION_CUH
#define GARBARGE_COLLECTION_CUH


// TODO: Remove magic number (31 = bits in prev - 1)
#define MARK_BACKTRACE(m_backtrace) ((m_backtrace)->prev |= (1U<<31))
#define UNMARK_BACKTRACE(m_backtrace) ((m_backtrace)->prev &= 0x7fffffff)
#define IS_MARKED(m_backtrace) ((m_backtrace)->prev >> 31)

#define GET_BITMAP_IDX(backtrace_idx) ((backtrace_idx) / bitmap_size_bits)

// If bitmap is changed to 32 bits, change those macros
#define BITMAP_POPC(bitmap_popc_macro) __popcll(bitmap_popc_macro)
#define BITMAP_FFS(bitmap_ffs_macro)   __ffsll(bitmap_ffs_macro)
#define MARKED_POPC(marked_popc_macro) __popc(marked_popc_macro)

__device__ void mark_offsets (
                        wfa_wavefront_t* const M_wavefronts,
                        wfa_wavefront_t* const I_wavefronts,
                        wfa_wavefront_t* const D_wavefronts,
                        const int active_working_set_size,
                        wfa_backtrace_t* offloaded_buffer);

__device__ void fill_bitmap (
                            wfa_cell_t* const cells,
                            wfa_backtrace_t* const offloaded_buffer,
                            const size_t offloaded_buffer_size,
                            const size_t last_free_bt_position,
                            const size_t bitmaps_size,
                            wfa_bitmap_t* const bitmaps,
                            wfa_rank_t* const ranks);

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
                                    const int active_working_set_size);

#if 0
__device__ void clean_offloaded_backtraces_buffer (
                                    const int32_t hi,
                                    const int32_t lo,
                                    uint32_t* last_free_idx_return,
                                    wfa_cell_t* cells,
                                    wfa_backtrace_t* offloaded_buffer_src,
                                    wfa_backtrace_t* offloaded_buffer_dst);
#endif

__device__ void pprint_offloaded_backtraces_buffer (
                                        wfa_backtrace_t* offloaded_buffer,
                                        const uint32_t* last_free_idx);

__device__ void pprint_offloaded_backtraces_chains (
                                        const wfa_backtrace_t* offloaded_buffer,
                                        const wfa_cell_t* cells,
                                        const int32_t hi,
                                        const int32_t lo,
                                        const uint32_t* last_free_idx);

#endif // header guard
