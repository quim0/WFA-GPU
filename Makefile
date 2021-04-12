CC=gcc
NVCC=nvcc
SRC=tools/aligner.c utils/arg_handler.c utils/sequence_reader.c
ARGS=-I .

aligner:
	$(CC) $(SRC) $(ARGS) -O3 -o wfa.affine.gpu
clean:
	rm -f wfa.affine.gpu
