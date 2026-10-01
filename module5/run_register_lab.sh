#!/usr/bin/env bash
# Module 5 register lab: run register.cu in its initial configuration, then with
# modified parameters (all set at compile time with -D):
#   KERNEL_LOOP - number of elements (one thread each)
#   KERNEL_SIZE - threads per block
#   WORK_ITERS  - steps per element in the register-vs-global loop kernels
set -e
cd "$(dirname "$0")/register_memory"

# build_run <name> [-DNAME=value ...]
build_run() {
    local name=$1
    shift
    echo "----- register.cu $* -----"
    nvcc "$@" -o "reg_$name" register.cu
    ./"reg_$name"
    echo
}

echo "===== Register usage per kernel (ptxas) ====="
nvcc -Xptxas -v -o reg_default register.cu 2>&1 | grep -E "Compiling entry|registers" || true
echo

echo "===== Initial parameters (2048 elements, 128 threads/block) ====="
build_run default

echo "===== Data set size (128 threads/block) ====="
for n in 256 65536 1048576 16777216; do
    build_run n$n -DKERNEL_LOOP=$n
done

echo "===== Threads per block (1M elements) ====="
for t in 32 256 1024; do
    build_run t$t -DKERNEL_LOOP=1048576 -DKERNEL_SIZE=$t
done

echo "===== Work per element: register vs. global (1M elements) ====="
for w in 1 128 1024; do
    build_run w$w -DKERNEL_LOOP=1048576 -DWORK_ITERS=$w
done

echo "===== Invalid launch: 2048 threads/block (expected CUDA error) ====="
build_run t2048 -DKERNEL_SIZE=2048 || true
