#!/usr/bin/env bash
# Module 4 practical activities: run host_memory and global_memory at several sizes.
set -e
cd "$(dirname "$0")"

SIZES="256 4096 65536 1048576"

echo "===== Host memory (saxpy: y = 2*x + y) ====="
nvcc -o host_memory host_memory.cu
for n in $SIZES; do
    ./host_memory "$n" 1.0 2.0
    echo
done
# Different source values for the host arrays
./host_memory 4096 3.0 5.0
echo

echo "===== Global memory (interleaved vs non-interleaved) ====="
for n in $SIZES; do
    nvcc -DNUM_ELEMENTS=${n}u -o global_memory_$n global_memory.cu
    ./global_memory_$n
    echo
done
