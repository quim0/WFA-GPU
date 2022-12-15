SM=80
COMPUTE=$(SM)
CC=gcc
NVCC=nvcc
SRC_PATH=lib
BUILD_PATH=build
SRC_ALIGNER=tools/aligner.c utils/arg_handler.c utils/sequence_reader.c
SRC_LIB=$(SRC_PATH)/kernels/sequence_alignment_kernel.cu $(SRC_PATH)/kernels/sequence_alignment_kernel_aband.cu $(SRC_PATH)/kernels/sequence_distance_kernel.cu $(SRC_PATH)/kernels/sequence_distance_kernel_aband.cu $(SRC_PATH)/kernels/sequence_packing_kernel.cu $(wildcard $(SRC_PATH)/*.cu) lib/alignment_results.c lib/aligner.c utils/cigar.c utils/verification.c utils/device_query.cu
SRC_WFA_CPU=utils/wfa_cpu.c
ARGS=-I . -Ilib/
ARGS_ALIGNER=-Wall -Wno-unused-function -Lbuild/ -L/usr/local/cuda/lib64 $(ARGS)
ARGS_WFA_CPU=-Lexternal/WFA/lib/ $(ARGS) -Iexternal/WFA/ -lwfa
NVCC_OPTIONS=-O3 -maxrregcount=64 -gencode arch=compute_$(COMPUTE),code=sm_$(SM) -Xptxas -v -Xcompiler -fopenmp

aligner: wfa-cpu wfa-gpu-so $(SRC_ALIGNER)
	mkdir -p bin
	$(CC) $(SRC_ALIGNER) $(ARGS_ALIGNER) -Lexternal/WFA/lib/ -O3 -o bin/wfa.affine.gpu -lwfagpu -lwfa -lm -Wl,-rpath=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))/build/

aligner-debug: wfa-cpu wfa-gpu-debug-so $(SRC_ALIGNER)
	mkdir -p bin
	$(CC) $(SRC_ALIGNER) $(ARGS_ALIGNER) -ggdb -DDEBUG -Lexternal/WFA/lib/ -o bin/wfa.affine.gpu -lwfagpu -lwfa -lm

aligner-profile: wfa-cpu wfa-gpu-profile-so $(SRC_ALIGNER)
	mkdir -p bin
	$(CC) $(SRC_ALIGNER) $(ARGS_ALIGNER) -Lexternal/WFA/lib/ -o bin/wfa.affine.gpu -lwfagpu -lwfa -lm

run-tests:
	for f in bin/test-*; do ./$$f; done

wfa-gpu-so: $(SRC_LIB)
	mkdir -p build
	$(NVCC) $(NVCC_OPTIONS) $(ARGS) -Xcompiler -fPIC -Xcompiler -fopenmp -dc $(SRC_LIB)
	mv *.o build/
	$(NVCC) $(NVCC_OPTIONS) -shared -o build/libwfagpu.so build/*.o -lcudart

wfa-gpu-debug-so: $(SRC_LIB)
	mkdir -p build
	$(NVCC) $(NVCC_OPTIONS) -g -G -DDEBUG $(ARGS) -Xcompiler -fPIC -dc $(SRC_LIB)
	mv *.o build/
	$(NVCC) $(NVCC_OPTIONS) -shared -o build/libwfagpu.so build/*.o -lcudart

wfa-gpu-profile-so: $(SRC_LIB)
	mkdir -p build
	$(NVCC) $(NVCC_OPTIONS) -lineinfo $(ARGS) -Xcompiler -fPIC -dc $(SRC_LIB)
	mv *.o build/
	$(NVCC) $(NVCC_OPTIONS) -lineinfo -shared -o build/libwfagpu.so build/*.o -lcudart

external/WFA:
	$(MAKE) -C $@

wfa-cpu: $(SRC_WFA_CPU) external/WFA
	mkdir -p build
	$(CC) $(ARGS) $(ARGS_WFA_CPU) -O3 -fopenmp -Wall -fPIC -c $(SRC_WFA_CPU)
	mv *.o build/

wfa-gpu: $(SRC_LIB) wfa-cpu
# TODO: Not working well
	mkdir -p build
	$(NVCC) $(NVCC_OPTIONS) $(ARGS) -Xcompiler -fPIC -dc $^
	mv *.o build/
	$(NVCC) -dlink -o wfagpu.o build/*.o -lcudart
	ar cru build/libwfagpu.a build/*.o

clean:
	rm -rf build/ bin/

.PHONY: external/WFA
