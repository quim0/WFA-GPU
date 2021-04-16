CC=gcc
NVCC=nvcc
SRC_PATH=lib
BUILD_PATH=build
SRC_ALIGNER=tools/aligner.c utils/arg_handler.c utils/sequence_reader.c
SRC_LIB=$(wildcard $(SRC_PATH)/kernels/*cu) lib/sequence_packing.cu lib/batch_async.cu
ARGS=-I . -Ilib/
ARGS_ALIGNER=-Lbuild/ -L/usr/local/cuda/lib64 -I . -Ilib/
NVCC_OPTIONS=#-gencode arch=compute_70,code=sm_70

aligner: wfa-gpu-so $(SRC_ALIGNER)
	mkdir -p bin
# Link static library, could be possible to link dynamic library too
	$(CC) $(SRC_ALIGNER) $(ARGS_ALIGNER) -O3 -o bin/wfa.affine.gpu -lwfagpu
	echo "!! Before running put `pwd`/build in LD_LIBRARY_PATH env variable."

aligner-debug: wfa-gpu-debug-so $(SRC_ALIGNER)
	mkdir -p bin
	$(CC) $(SRC_ALIGNER) $(ARGS_ALIGNER) -ggdb -DDEBUG -o bin/wfa.affine.gpu -lwfagpu

wfa-gpu-so: $(SRC_LIB)
	mkdir -p build
	$(NVCC) -O2 $(NVCC_OPTIONS) $(ARGS) -Xcompiler -fPIC -dc $^
	mv *.o build/
	$(NVCC) -shared -o build/libwfagpu.so build/*.o -lcudart

wfa-gpu-debug-so: $(SRC_LIB)
	mkdir -p build
	$(NVCC) $(NVCC_OPTIONS) -g -DDEBUG $(ARGS) -Xcompiler -fPIC -dc $^
	mv *.o build/
	$(NVCC) -shared -o build/libwfagpu.so build/*.o -lcudart

wfa-gpu: $(SRC_LIB)
# TODO: Not working well
	mkdir -p build
	$(NVCC) -O2 $(NVCC_OPTIONS) $(ARGS) -Xcompiler -fPIC -dc $^
	mv *.o build/
	$(NVCC) -dlink -o wfagpu.o build/*.o -lcudart
	ar cru build/libwfagpu.a build/*.o

clean:
	rm -rf build/ bin/
