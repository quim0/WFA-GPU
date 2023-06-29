#!/usr/bin/env bash

TEST_NAME="FASTA PARSER"
TEST_FILE="test-fasta.sh"
TEST_SUCCEED="\033[32mTEST $TEST_NAME OK ($TEST_FILE)\033[0m"

print_fail_str() {
    echo -e "\033[0;31mTEST $TEST_NAME FAILED: $1 ($TEST_FILE)\033[0m"
}

../bin/wfa.affine.gpu -Q ./data/test_hifi.query.fasta -T ./data/test_hifi.target.fasta -c > out 2>&1
retVal=$?
if [ $retVal -ne 0 ]; then
    print_fail_str "Test 1 returned $retVal."
    exit
fi

if grep -q "correct=50" out; then
    true
else
    print_fail_str "Incorrect alignments generated (test1)."
    exit
fi

../bin/wfa.affine.gpu -Q ./data/test_hifi.query.fasta -T ./data/test_hifi.target.fasta -g 5,2,5 -b 11 -c > out2 2>&1
retVal=$?
if [ $retVal -ne 0 ]; then
    print get_fail_str "Test 2 returned $retVal."
    exit
fi


../bin/wfa.affine.gpu -Q ./data/test_hifi.query.fasta -T ./data/test_hifi.target.fasta -x -c > out3 2>&1
retVal=$?
if [ $retVal -ne 0 ]; then
    print_fail_str "Test 3 returned $retVal."
    exit
fi

# TODO: CHECK VALUES

rm out out2 out3

echo -e $TEST_SUCCEED
