#!/usr/bin/env bash
# Module 6 practical activity: run stream_example.cu in its initial state, then
# with modified parameters (runtime args):
#   stream_example [array_size] [threads_per_block] [num_streams]
set -e
cd "$(dirname "$0")"

nvcc -o stream_example stream_example.cu

run() {
    echo "----- stream_example $* -----"
    ./stream_example "$@"
    echo
}

echo "===== Initial state (1M ints, 1 thread/block, 1 stream) ====="
run

echo "===== Block size (1M ints, 1 stream) ====="
for t in 32 256 1024; do run 1048576 $t 1; done

echo "===== Data set size (256 threads/block, 1 stream) ====="
for n in 4096 65536 16777216 67108864; do run $n 256 1; done

echo "===== Number of streams: copy/compute overlap (256 threads/block) ====="
for s in 2 4 8; do run 1048576 256 $s; done
for s in 4 16; do run 16777216 256 $s; done
run 67108864 256 8
