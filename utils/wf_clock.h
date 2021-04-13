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

#ifndef WF_CLOCK_H
#define WF_CLOCK_H

#include "logger.h"

#include <time.h>

#ifdef DEBUG

#define DEBUG_CLOCK_INIT() struct timespec wf_clock_now, wf_clock_tmstart; double wf_clock_seconds;
#define DEBUG_CLOCK_START() clock_gettime(CLOCK_REALTIME, &wf_clock_tmstart);
#define DEBUG_CLOCK_STOP(text) \
    clock_gettime(CLOCK_REALTIME, &wf_clock_now); \
    wf_clock_seconds = (double)((wf_clock_now.tv_sec+wf_clock_now.tv_nsec*1e-9) - \
                       (double)(wf_clock_tmstart.tv_sec+wf_clock_tmstart.tv_nsec*1e-9)); \
    LOG_DEBUG("%s Wall time %fs", text, wf_clock_seconds);

#else


#define DEBUG_CLOCK_INIT()
#define DEBUG_CLOCK_START()
#define DEBUG_CLOCK_STOP(text)
#endif // DEBUG

#endif
