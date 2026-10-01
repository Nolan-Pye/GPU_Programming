#!/usr/bin/env bash
# Module 5 shared memory lab: run shared_memory.cu in its initial configuration,
# then with modified parameters (all set at compile time with -D):
#   NUM_ELEMENTS  - values sorted (multiple of MAX_NUM_LISTS, <= 4096 for 48 KB smem)
#   MAX_NUM_LISTS - interleaved lists == threads in the block
#   MERGE_VARIANT - 1 single-thread, 5 reduction, 6 atomicMin, 9 two-level atomicMin
#   INPUT_MODE    - 0 ascending (original), 1 descending, 2 random
set -e
cd "$(dirname "$0")/shared_constant_memory"

# build_run <name> [-DNAME=value ...]
build_run() {
    local name=$1
    shift
    echo "----- shared_memory.cu $* -----"
    nvcc "$@" -o "sm_$name" shared_memory.cu
    ./"sm_$name"
    echo
}

echo "===== Initial parameters (2048 elements, 16 lists, merge1, ascending input) ====="
build_run default

echo "===== Input order ====="
build_run desc -DINPUT_MODE=1
build_run rand -DINPUT_MODE=2

echo "===== Merge variants (random input) ====="
for m in 1 5 6 9; do
    build_run merge$m -DINPUT_MODE=2 -DMERGE_VARIANT=$m
done

echo "===== Data set size (16 lists, merge6, random input) ====="
for n in 256 1024 4096; do
    build_run n$n -DINPUT_MODE=2 -DMERGE_VARIANT=6 -DNUM_ELEMENTS=$n
done

echo "===== Number of lists / threads (2048 elements, random input) ====="
for l in 4 8 32 64 128; do
    build_run l${l}_m9 -DINPUT_MODE=2 -DMERGE_VARIANT=9 -DMAX_NUM_LISTS=$l
    build_run l${l}_m1 -DINPUT_MODE=2 -DMERGE_VARIANT=1 -DMAX_NUM_LISTS=$l
    build_run l${l}_m6 -DINPUT_MODE=2 -DMERGE_VARIANT=6 -DMAX_NUM_LISTS=$l
done

echo "===== Too large: 8192 elements needs 96 KB static shared memory (expected to fail) ====="
nvcc -DNUM_ELEMENTS=8192 -o sm_too_big shared_memory.cu 2>&1 | grep -i -m3 "shared\|error" || true
