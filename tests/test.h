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

#ifndef WFA_TEST_H
#define WFA_TEST_H

#include <stdio.h>

#define SET_TEST_NAME(name) const char* wfa_tests_test_name = name;

#define TEST_OK fprintf(stderr, "\033[32mTEST %s OK (%s)\033[0m\n", wfa_tests_test_name, __FILE__);

#define TEST_FAIL(tfail_str) { \
    fprintf(stderr, "\033[0;31mTEST %s FAILED: %s (%s:%d)\033[0m\n", \
                    wfa_tests_test_name, tfail_str, __FILE__, __LINE__); exit(-1); }

#define TEST_ASSERT(cond) if (!(cond)) TEST_FAIL("Assertion failed.")

#define CUDA_TEST_CHECK_ERR { \
    cudaError_t c_err = cudaGetLastError(); \
    if (c_err != cudaSuccess) { \
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err), \
               cudaGetErrorString(c_err), __FILE__, __LINE__); \
        TEST_FAIL("CUDA runtime error.") \
        exit(-1); \
    } \
}

#endif
