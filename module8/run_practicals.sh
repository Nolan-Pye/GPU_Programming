#!/usr/bin/env bash
# Module 8 practical activities: build the NPP box filter, Thrust, and
# PageRank (nvGRAPH sample ported to cuSPARSE) examples, run each in its
# initial state, then with modified parameters.
#
# A standard Linux CUDA toolkit needs nothing extra. If the library headers/libs
# or the host compiler live elsewhere, pass extra nvcc flags, e.g.
#   NVCC_EXTRA="-I/path/include -L/path/lib" ./run_practicals.sh
set -e
cd "$(dirname "$0")"

NVCC="nvcc -O2 -arch=native -std=c++17 ${NVCC_EXTRA:-}"

$NVCC -I../common/inc -Icommon/UtilNPP -o boxFilterNPP/boxFilterNPP boxFilterNPP/boxFilterNPP.cpp \
      -lnppif -lnppisu -lnppc
$NVCC -x cu -o thrust_examples/thrust_example thrust_examples/thrust_example.cpp
$NVCC -I../common/inc -o nvgraph_examples/pagerank_cusparse nvgraph_examples/pagerank_cusparse.cu -lcusparse

run() {
    echo "----- $* -----"
    ./"$@"
    echo
}

LENA=../common/data/Lena.pgm
mkdir -p boxFilterNPP/out

echo "################ NPP: box filter ################"
echo "===== Initial state (Lena.pgm 512x512, 5x5 mask) ====="
run boxFilterNPP/boxFilterNPP -input=$LENA -output=boxFilterNPP/out/Lena_boxFilter.pgm
echo "===== Mask size on Lena (odd sizes) ====="
for m in 3 7 9 15 31; do
    run boxFilterNPP/boxFilterNPP -input=$LENA -mask=$m -output=boxFilterNPP/out/Lena_box$m.pgm
done
echo "===== Even mask sizes (anchor no longer centered) ====="
for m in 4 8; do
    run boxFilterNPP/boxFilterNPP -input=$LENA -mask=$m -output=boxFilterNPP/out/Lena_box$m.pgm
done
echo "===== Larger synthetic images, 5x5 mask ====="
for n in 1024 2048 4096 8192; do
    run boxFilterNPP/boxFilterNPP -size=$n -output=boxFilterNPP/out/synthetic_$n.pgm
done
echo "===== Large image and large mask ====="
run boxFilterNPP/boxFilterNPP -size=4096 -mask=15 -output=boxFilterNPP/out/synthetic_4096_box15.pgm

echo "################ Thrust: host/device vectors, random data, algorithms ################"
echo "===== Initial state ====="
run thrust_examples/thrust_example
echo "===== Random data, small; same seed twice, then a different seed ====="
run thrust_examples/thrust_example 1000 1234
run thrust_examples/thrust_example 1000 1234
run thrust_examples/thrust_example 1000 42
echo "===== Larger data sets ====="
for n in 1000000 16777216 33554432; do run thrust_examples/thrust_example $n; done

echo "################ PageRank: nvGRAPH sample ported to cuSPARSE + Thrust ################"
echo "(nvgraph_Pagerank.cpp itself needs nvGRAPH, removed in CUDA 11; not built)"
echo "===== Initial state (6-vertex sample graph, alpha 0.85 then 0.90) ====="
run nvgraph_examples/pagerank_cusparse
echo "===== Damping factor on the sample graph ====="
for a in 0.5 0.99; do run nvgraph_examples/pagerank_cusparse -alpha=$a; done
echo "===== Tolerance on the sample graph ====="
run nvgraph_examples/pagerank_cusparse -tol=1e-3
run nvgraph_examples/pagerank_cusparse -tol=1e-9
echo "===== Random graphs, average out-degree 8 ====="
for n in 10000 100000 1000000 4000000; do run nvgraph_examples/pagerank_cusparse -n=$n -deg=8; done
echo "===== Edge density at 1M vertices ====="
for d in 2 32; do run nvgraph_examples/pagerank_cusparse -n=1000000 -deg=$d; done
echo "===== High damping on a large graph ====="
run nvgraph_examples/pagerank_cusparse -n=1000000 -deg=8 -alpha=0.99
