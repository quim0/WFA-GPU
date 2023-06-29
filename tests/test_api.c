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

#include "include/wfa_gpu.h"

#include "data/sequences_10K.h"
#include "data/sequences_1000.h"
#include "test.h"

SET_TEST_NAME("ALIGNMENT API")

void test_sequences_invalid_aligner_pointer () {
    bool ok = wfagpu_initialize_aligner(NULL);
    TEST_ASSERT(!ok);

    ok = wfagpu_add_sequences(NULL, sequences_10K_n100[0], sequences_10K_n100[1]);
    TEST_ASSERT(!ok);

    affine_penalties_t penalties;
    ok = wfagpu_initialize_parameters(NULL, penalties);
}

void test_sequences_invalid_penalties () {
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<200; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_10K_n100[i], sequences_10K_n100[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = -2, .o = 3, .e = 1};
    bool ok = wfagpu_initialize_parameters(&aligner, penalties);
    TEST_ASSERT(!ok);

    penalties.x = 0; penalties.o = 0; penalties.e = 0;
    ok = wfagpu_initialize_parameters(&aligner, penalties);
    TEST_ASSERT(!ok);
}

void test_sequences_10k_single_batch_x2_o3_e1 () {
    // 10K 10% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<200; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_10K_n100[i], sequences_10K_n100[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = true;

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<100; i++) {
            const int error = aligner.results[j*100 + i].error;
            if (error != -results_10K_n100_x2o3e1[i]) {
                TEST_FAIL("Incorrect result (length=10K, error=10\%, (x,o,e)=(2,3,1)");
            }
        }
    }
}

void test_sequences_10k_multi_batch_x2_o3_e1 () {
    // 10K 10% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<200; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_10K_n100[i], sequences_10K_n100[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = true;
    wfagpu_set_batch_size(&aligner, 100);

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<100; i++) {
            const int error = aligner.results[j*100 + i].error;
            if (error != -results_10K_n100_x2o3e1[i]) {
                TEST_FAIL("Incorrect result (length=10K, error=10\%, (x,o,e)=(2,3,1)");
            }
        }
    }
}

void test_sequences_10k_multi_batch_x3_o5_e2 () {
    // 10K 10% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<200; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_10K_n100[i], sequences_10K_n100[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 3, .o = 5, .e = 2};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = true;
    wfagpu_set_batch_size(&aligner, 100);

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<100; i++) {
            const int error = aligner.results[j*100 + i].error;
            if (error != -results_10K_n100_x3o5e2[i]) {
                TEST_FAIL("Incorrect result (length=10K, error=10\%, (x,o,e)=(3,5,2)");
            }
        }
    }
}

void test_sequences_10k_multi_batch_x3_o5_e2_distance_only () {
    // 10K 10% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<200; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_10K_n100[i], sequences_10K_n100[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 3, .o = 5, .e = 2};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = false;
    wfagpu_set_batch_size(&aligner, 100);

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<100; i++) {
            const int error = aligner.results[j*100 + i].error;
            if (error != -results_10K_n100_x3o5e2[i]) {
                printf("%d, %d, %d\n", i, error, -results_10K_n100_x3o5e2[i]);
                TEST_FAIL("Incorrect result (length=10K, error=10\%, (x,o,e)=(3,5,2)");
            }
        }
    }
}

void test_sequences_1000_multi_batch_x2_o3_e1 () {
    // 1000 5% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<2000; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_1000_n1000[i], sequences_1000_n1000[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = true;
    wfagpu_set_batch_size(&aligner, 2500);

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<1000; i++) {
            const int error = aligner.results[j*1000 + i].error;
            if (error != -results_1000_n1000_x2o3e1[i]) {
                TEST_FAIL("Incorrect result (length=1000, error=5\%, (x,o,e)=(2,3,1)");
            }
        }
    }
}

void test_sequences_1000_multi_batch_x5_o3_e2 () {
    // 1000 5% error
    wfagpu_aligner_t aligner = {0};
    wfagpu_initialize_aligner(&aligner);
    for (int j=0; j<10; j++) {
        for (int i=0; i<2000; i+=2) {
            wfagpu_add_sequences(&aligner, sequences_1000_n1000[i], sequences_1000_n1000[i+1]);
        }
    }

    affine_penalties_t penalties = {.x = 5, .o = 3, .e = 2};
    wfagpu_initialize_parameters(&aligner, penalties);
    aligner.alignment_options.compute_cigar = true;
    wfagpu_set_batch_size(&aligner, 2500);

    wfagpu_align(&aligner);

    for (int j=0; j<10; j++) {
        for (int i=0; i<1000; i++) {
            const int error = aligner.results[j*1000 + i].error;
            if (error != -results_1000_n1000_x5o3e2[i]) {
                TEST_FAIL("Incorrect result (length=1000, error=5\%, (x,o,e)=(3,5,2)");
            }
        }
    }
}

int main () {
    test_sequences_invalid_aligner_pointer();
    test_sequences_invalid_penalties();
    test_sequences_10k_single_batch_x2_o3_e1();
    test_sequences_10k_multi_batch_x2_o3_e1();
    test_sequences_10k_multi_batch_x3_o5_e2();
    test_sequences_10k_multi_batch_x3_o5_e2_distance_only();
    test_sequences_1000_multi_batch_x2_o3_e1();
    test_sequences_1000_multi_batch_x5_o3_e2();
    TEST_OK
}
