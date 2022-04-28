# WFA-GPU

WFA-GPU is a CUDA library to perform pairwise gap-affine global DNA sequence alignment on Nvidia GPUs.
It implements the [WFA algorithm](https://academic.oup.com/bioinformatics/article/37/4/456/5904262)

## Build

Make sure you have installed an up to date [CUDA toolkit](https://developer.nvidia.com/cuda-downloads), and a CUDA capable device (i.e. an NVidia GPU).
To compile the library and tools, run the following commands:

```
$ git clone git@github.com:quim0/WFA-GPU.git
$ cd WFA-GPU
$ ./build.sh
```

The `build.sh` script notifies if there is any missing necessary software for compiling the library and the tools.

## Using WFA-GPU in your project

This is a simple example on how to use WFA-GPU on your projects, for extended examples, see the files in `examples/`.

``` c
#include "include/wfa_gpu.h"

// ...

// Create and initialize the aligner structure
wfagpu_aligner_t aligner = {0};
wfagpu_initialize_aligner(&aligner);

// Add the sequences to align: wfagpu_add_sequences(&aligner, query, target)
wfagpu_add_sequences(&aligner, "GAATA", "GATACA");
wfagpu_add_sequences(&aligner, "CATTAATCTT", "CAGTAAT");
// ... Add as many sequence pairs as you need

// Initialize the alignment parameters (*after* the sequences have been
// added)
affine_penalties_t penalties = {.x = 2, .o = 3, .e = 1};
wfagpu_initialize_parameters(&aligner, penalties);

// Optionally set batch size (number of sequences aligned in parallel)
wfagpu_set_batch_size(&aligner, 10);

// Align all sequence pairs
wfagpu_align(&aligner);

// Get error (score) of first alignment
aligner.results[0].error
// Get ASCII CIGAR of first alignment (char*)
aligner.results[0].cigar.buffer
```

To compile, use the following options, where `$WFAGPU_PATH` is the path of this respository.

```
gcc test_wfagpu.c -o test-wfagpu -I $WFAGPU_PATH/lib/ -I $WFAGPU_PATH -L $WFAGPU_PATH/build/ -L $WFAGPU_PATH/external/WFA/lib/ -lwfagpu -lwfa
```

Then, to execute the generated binary, the OS needs to be able to find the dynamic library `$WFAGPU_PATH/build/libwfagpu.so`. A fast and easy way of doing so is adding it into the `LD_LIBRAY_PATH` environment variable (`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WFAGPU_PATH/build/`).

## Tools

WFA-GPU comes with a tool to test its functionality, it is compiled (with the instruction on section "Build") to the `bin/wfa.affine.gpu` binary.
Running the binary without any arguments lists the help menu:

```
[Input/Output]
	-i, --input-file                    (string, required) Input sequences file: File containing the sequences to align.
	-n, --num-alignments                (int) Number of alignments: Number of alignments to read from the file (default=all alignments)
	-o, --output-file                   (string) Output File: File where alignment output is saved.
	-p, --print-output                  Print: Print output to stderr
	-O, --output-verbose                Verbose output: Add the query/target information on the output
[Alignment Options]
	-g, --affine-penalties              (string, required) Affine penalties: Gap-affine penalties for the alignment, in format x,o,e
	-e, --max-distance                  (int) Maximum error allowed: Maximum error that the kernel will be able to compute (default = assume 10% error on the first sequence pair)
	-b, --batch-size                    (int) Batch size: Number of alignments per batch.
	-B, --band                          (int) Adaptative band: Wavefront band (highest and lower diagonal that will be initially computed). Use "auto" to use an automatically generated band according to other parameters.
[System]
	-c, --check                         Check: Check for alignment correctness
	-t, --threads-per-block             (int) Number of CUDA threads per alignment: Number of CUDA threads per block, each block computes one or multiple alignment
	-w, --workers                       (int) GPU workers: Number of blocks ('workers') to be running on the GPU.
```

Choosing the correct alignment and system options is key for performance. The binary tries to automatically choose adequate paramters, but the user
may have additional information to make a better choise. It is specially important to limit the maximum error supported by the kernel as much as
possible (`-e` parameter), this contrains the memory used per alignment, and helps the program to choose better block and grid sizes. Keep in mind that any alignment having an error higher than the specified with the `-e` argument will be computed on the CPU, so, if this argument is too small, performance can decrease.

For big alignments, setting a band (i.e. limiting the maximum and minimum diagonal of the wavefronts) with the `-B` argument can give significant
speedups, at the expense of potentially loosing some accuracy in corner cases.

## Troubleshooting

#### cudaErrorLaunchTimeout

When there is a screen connected, the maximum kernel time is 5 seconds. Disable the GUI or choose a smaller batch size to reduce kernel execution time.
On Ubuntu based systems, the GUI can be disabled with the command `sudo systemctl isolate multi-user.target` (keep in mind that this will close all applications on your desktop environment such as browsers, text-editors... etc).

#### Out of memory

The program is trying to use too much memory. Decrease batch size or maximum error supported by the kernel. The aligner tool stores all sequences to main memory
before starting the alignment on the GPU, if your machine does not have enough memory, it can also raise an out-of-memory error on the CPU side.

## Problems and suggestions

Open an issue on Github or contact with the main developer: Quim Aguado-Puig (quim.aguado.p@gmail.com)

## License

WFA-GPU is distributed under the MIT licence.

## Citation

Quim Aguado-Puig, Santiago Marco-Sola, Juan Carlos Moure, Christos Matzoros, David Castells-Rufas, Antonio Espinosa, Miquel Moreto. WFA-GPU: Gap-affine pairwise alignment using GPUs. bioRxiv (2022). DOI [2022.04.18.488374](https://doi.org/10.1101/2022.04.18.488374)
