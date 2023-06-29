SM=80
COMPUTE=$(SM)
CC=gcc
NVCC=nvcc
WFAGPU_PATH=../

all: clean tests run-tests

tests: test-api test-packing

test-api: test_api.c
	$(CC) $< -o $@ -I $(WFAGPU_PATH)/lib/ -I $(WFAGPU_PATH) -L $(WFAGPU_PATH)/build/ -L $(WFAGPU_PATH)/external/WFA/lib/ -lwfagpu -lwfa -lm -fopenmp

test-packing: test_packing_kernel.cu
	$(NVCC) -g -G -I../lib/ -I../ ../utils/verification.c $< -L/usr/local/cuda/lib64 -L $(WFAGPU_PATH)/build/ -L $(WFAGPU_PATH)/external/WFA/lib/ -lwfagpu -lwfa -lm -Xcompiler -fopenmp -o $@

run-tests:
	for f in ./test-*; do LD_LIBRARY_PATH=../build/ ./$$f; done

clean:
	rm -f test-api test-packing
