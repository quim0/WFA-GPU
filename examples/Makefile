CC=gcc
CXX=g++
LIB_PATH=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))/../build/

all: manual auto auto-cpp

manual:
	$(CC) manual_example.c -o manual-example -I../lib/ -I.. -L ../build/ -L ../external/WFA/lib/ -lwfagpu -lwfa -lm -fopenmp -Wl,-rpath=$(LIB_PATH)

auto:
	$(CC) auto_example.c -o auto-example -I../lib/ -I.. -L ../build/ -L ../external/WFA/lib/ -lwfagpu -lwfa -lm -fopenmp -Wl,-rpath=$(LIB_PATH)

auto-cpp:
	$(CXX) auto_example.cpp -o auto-example-cpp -I../lib/ -I.. -L ../build/ -L ../external/WFA/lib/ -lwfagpu -lwfa -lm -fopenmp -Wl,-rpath=$(LIB_PATH)

clean:
	rm -f auto-example manual-example auto-example-cpp
