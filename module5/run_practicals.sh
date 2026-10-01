#!/usr/bin/env bash
# Module 5 practical activity: run constant_memory and constant_memory2 in their
# initial state, then with modified parameters (all set at compile time with -D):
#   KERNEL_LOOP  - iterations of the read loop in each kernel
#                  (constant_memory2: also the __constant__ array length, max 16384)
#   NUM_ELEMENTS - threads launched for the timing comparison
#   WORK_SIZE    - elements checked against the host result
set -e
cd "$(dirname "$0")/shared_constant_memory"

# build_run <source> <name> [-DNAME=value ...]
build_run() {
    local src=$1 name=$2
    shift 2
    echo "----- $src $* -----"
    nvcc "$@" -o "$name" "$src"
    ./"$name"
    echo
}

echo "===== constant_memory.cu (literal vs. __constant__ vs. __device__) ====="
build_run constant_memory.cu cm_default
build_run constant_memory.cu cm_small   -DWORK_SIZE=64     -DNUM_ELEMENTS=16384   -DKERNEL_LOOP=1024
build_run constant_memory.cu cm_large   -DWORK_SIZE=4096   -DNUM_ELEMENTS=1048576 -DKERNEL_LOOP=262144

echo "===== constant_memory2.cu (__constant__ array vs. __device__ array) ====="
build_run constant_memory2.cu cm2_default
build_run constant_memory2.cu cm2_small -DWORK_SIZE=64     -DNUM_ELEMENTS=16384   -DKERNEL_LOOP=1024
build_run constant_memory2.cu cm2_large -DWORK_SIZE=4096   -DNUM_ELEMENTS=1048576 -DKERNEL_LOOP=16384
