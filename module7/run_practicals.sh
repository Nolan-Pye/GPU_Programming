#!/usr/bin/env bash
# Module 7 practical activities: build the cuFFT, cuBLAS, cuSPARSE, cuSOLVER and
# cuRAND examples, run each in its initial state, then with modified parameters.
#
# A standard Linux CUDA toolkit needs nothing extra. If the library headers/libs
# or the host compiler live elsewhere, pass extra nvcc flags, e.g.
#   NVCC_EXTRA="-I/path/include -L/path/lib" ./run_practicals.sh
set -e
cd "$(dirname "$0")"

NVCC="nvcc -O2 -arch=native ${NVCC_EXTRA:-}"

$NVCC -o cufft_example    cufft_example.cu    -lcufft
$NVCC -o cublas_example   cublas_example.cu   -lcublas
$NVCC -o cusparse_example cusparse_example.cu -lcusparse -lcublas
$NVCC -o cusolver_example cusolver_example.cu -lcusolver -lcublas
$NVCC -o curand_example   curand_example.cu   -lcurand

run() {
    echo "----- $* -----"
    ./"$@"
    echo
}

echo "################ cuFFT: 2D convolution via FFT ################"
echo "===== Initial state (5x5 all-ones signal and filter) ====="
run cufft_example
echo "===== Original multiply kernel, exactly as written (no 1/(N*N) scaling) ====="
run cufft_example 5 ones orig
echo "===== Original kernel on non-trivial (random) input: the multiply bug shows up ====="
run cufft_example 8 random orig || true
echo "===== Fixed kernel on random input, growing N ====="
for n in 8 64 256 1024 4096; do run cufft_example $n random; done
echo "===== Non-power-of-two sizes (1000 = 2^3*5^3, 1009 is prime) ====="
for n in 1000 1009; do run cufft_example $n random; done

echo "################ cuBLAS: SGEMM C = A*B ################"
echo "===== Initial state (A 2x9, B 9x2) ====="
run cublas_example -noprompt
echo "===== Small square (printed) ====="
run cublas_example 4 4 4
echo "===== Square matrices, growing size ====="
for n in 128 512 1024 2048 4096; do run cublas_example $n $n $n; done
echo "===== Skinny shapes (same 2*M*N*K flops budget, different arithmetic intensity) ====="
run cublas_example 4096 16 4096
run cublas_example 16 65536 16

echo "################ cuSPARSE + cuBLAS: conjugate gradient ################"
echo "===== Initial state (N = 1048576, diag in [10,11)) ====="
run cusparse_example
echo "===== Matrix size ====="
for n in 1024 65536 16777216; do run cusparse_example $n; done
echo "===== Conditioning: smaller diagonal -> more iterations (N = 1048576) ====="
for d in 4 2.5 2.1; do run cusparse_example 1048576 $d; done
echo "===== Looser tolerance ====="
run cusparse_example 1048576 2.1 1e-2

echo "################ cuSOLVER + cuBLAS: QR solve of A*x = b ################"
echo "===== Initial state (hard-coded 3x3) ====="
run cusolver_example
echo "===== Random systems, growing m ====="
for m in 4 64 256 1024 2048 4096; do run cusolver_example $m; done

echo "################ cuRAND: device and host API ################"
echo "===== Initial state (25 blocks x 20 threads, [0,100), time seed) ====="
run curand_example
echo "===== Fixed seed, run twice: identical output ====="
run curand_example 4 8 100 1234
run curand_example 4 8 100 1234
echo "===== Different range: dice rolls in [0,6) ====="
run curand_example 4 8 6 1234
echo "===== Larger data sets ====="
run curand_example 4096 256 100 1234
run curand_example 16384 256 1000 1234
run curand_example 65536 256 1000 1234
