#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

# Check nvcc/CUDA
NVCC=$(command -v nvcc)
if [ -z "$NVCC" ]; then
    NVCC=$(command -v $CUDA_PATH/bin/nvcc)
    if [ -z "$NVCC" ]; then
        echo "nvcc not found, install CUDA and try again (https://developer.nvidia.com/cuda-downloads)"
        echo "If you have CUDA installed in a path different than $CUDA_PATH, change the \$CUDA_PATH variable in this script."
        exit 1
    fi
fi
echo "Detected nvcc: $NVCC"

# Check compute capability
CAPABILITY_BIN=wfagpu_test_cuda_capability
NAME_BIN=wfagpu_test_cuda_name

$NVCC tools/cuda_capability.cu utils/device_query.cu -I. -o $CAPABILITY_BIN
$NVCC tools/cuda_name.cu utils/device_query.cu -I. -o $NAME_BIN

OUTPUT=$(./$CAPABILITY_BIN)
if [ $? -ne 0 ]; then
    echo "No CUDA devices found on this machine. Aborting."
    exit 1
fi

MAJOR_CAP=${OUTPUT:0:1}
MINOR_CAP=${OUTPUT:1:1}

echo "Detected device $(./$NAME_BIN) with compute capability $MAJOR_CAP.$MINOR_CAP"

rm $CAPABILITY_BIN
rm $NAME_BIN

# Check gcc
GCC=$(command -v gcc)
if [ -z "$GCC" ]; then
    echo "gcc not found. Aborting."
    exit 1
fi
echo "Detected gcc: $GCC"

# Check if the compute capability of the GPU is supported by the CUDA version
# installed
output=$(nvcc --run -arch=compute_$MAJOR_CAP$MINOR_CAP -code=sm_$MAJOR_CAP$MINOR_CAP 2>&1)

if [[ $output == *"compute"* ]]; then
    if [ $MINOR_CAP -ne 0 ]; then
        echo "Compute capability $MAJOR_CAP.$MINOR_CAP is NOT supported.  Downgrading to $MAJOR_CAP.0"
        MINOR_CAP=0
        output=$(nvcc --run -arch=compute_$MAJOR_CAP$MINOR_CAP -code=sm_$MAJOR_CAP$MINOR_CAP 2>&1)
        if [[ $output == *"compute"* ]]; then
            echo "ERROR: Your CUDA version do not support the compute capability of your GPU ($MAJOR_CAP.$MINOR_CAP), please update CUDA."
            exit
        fi
    else
        echo "ERROR: Your CUDA version do not support the compute capability of your GPU ($MAJOR_CAP.$MINOR_CAP), please update CUDA."
        exit
    fi
fi

# Build
make CC=$GCC NVCC=$NVCC SM=$MAJOR_CAP$MINOR_CAP clean aligner
