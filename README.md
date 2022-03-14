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

TODO

## Tools

WFA-GPU comes with a tool to test its functionality, it is compiled (with the instruction on section "Build") to the `bin/wfa.affine.gpu` binary.
Running the binary without any arguments lists the help menu:

```
[Input/Output]
	-i, --input                         (string, required) Input sequences file: File containing the sequences to align.
	-n, --num-alignments                (int) Number of alignments: Number of alignments to read from the file (default=all alignments)
[Alignment Options]
	-g, --affine-penalties              (string, required) Affine penalties: Gap-affine penalties for the alignment, in format x,o,e
	-e, --max-distance                  (int) Maximum error allowed: Maximum error that the kernel will be able to compute (default = maximum possible error of first alignment)
	-b, --batch-size                    (int) Batch size: Number of alignments per batch.
	-B, --band                          (int) Band: Wavefront band (highest and lower diagonal that will be computed).
[System]
	-c, --check                         Check: Check for alignment correctness
	-t, --threads-per-block             (int) Number of CUDA threads per alginment: Number of CUDA threads per block, each block computes one or multiple alignment
	-w, --workers                       (int) GPU workers: Number of blocks ('workers') to be running on the GPU.
```

Choosing the correct alignment and system options is key for performance. The binary tries to automatically choose adequate paramters, but the user
may have additional information to make a better choise. It is specially important to limit the maximum error supported by the kernel as much as
possible (`-e` parameter), this contrains the memory used per alignment, and helps the program to choose better block and grid sizes. Keep in mind that any alignment having an error higher than the specified with the `-e` argument will be computed on the CPU, so, if this argument is too small, performance can decrease.

For big alignments, setting a band (i.e. limiting the maximum and minimum diagonal of the wavefronts) with the `-B` argument can give significant
speedups, at the expense of potentially loosing some accuracy in corner cases.

## Examples

TODO

## Possible errors

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
