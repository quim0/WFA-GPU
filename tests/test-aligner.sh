#!/bin/bash

TEST_NAME="ALIGNER"
TEST_FILE="test-aligner.sh"
TEST_SUCCEED="\033[32mTEST $TEST_NAME OK ($TEST_FILE)\033[0m"

print_fail_str() {
    echo -e "\033[0;31mTEST $TEST_NAME FAILED: $1 ($TEST_FILE)\033[0m"
}

../bin/wfa.affine.gpu -i ./data/wfa.utest.seq -g 1,2,1 -e 10000 -o res1.out > out 2>&1

if ! diff -w -B -q res1.out data/results/test.score.affine.p0.alg &> /dev/null; then
    print_fail_str "Incorrect results generated (test1)."
    exit
fi

# Multi batch, with a "weird" batch size
../bin/wfa.affine.gpu -i ./data/wfa.utest.seq -g 1,2,1 -e 10000 -o res1.out > out 2>&1

if ! diff -w -B -q res1.out data/results/test.score.affine.p0.alg &> /dev/null; then
    print_fail_str "Incorrect results generated (test2)."
    exit
fi

# Low max error (test CPU recovery)
../bin/wfa.affine.gpu -i ./data/wfa.utest.seq -g 1,2,1 -e 25 -o res1.out > out 2>&1

if ! diff -w -B -q res1.out data/results/test.score.affine.p0.alg &> /dev/null; then
    print_fail_str "Incorrect results generated (test3)."
    exit
fi

# Penalties 3,1,4
../bin/wfa.affine.gpu -i ./data/wfa.utest.seq -g 3,1,4 -e 10000 -o res1.out > out 2>&1

if ! diff -w -B -q res1.out data/results/test.score.affine.p1.alg &> /dev/null; then
    print_fail_str "Incorrect results generated (test4)."
    exit
fi

# Penalties 5,3,2
../bin/wfa.affine.gpu -i ./data/wfa.utest.seq -g 5,3,2 -e 10000 -o res1.out > out 2>&1

if ! diff -w -B -q res1.out data/results/test.score.affine.p2.alg &> /dev/null; then
    print_fail_str "Incorrect results generated (test5)."
    exit
fi

rm -f res1.out

echo -e $TEST_SUCCEED
