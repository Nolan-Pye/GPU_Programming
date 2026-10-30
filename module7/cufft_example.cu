/*
 * 2D circular convolution through the frequency domain with cuFFT:
 *     result = IFFT2( FFT2(signal) .* FFT2(filter) ) / (N*N)
 *
 * Usage: cufft_example [N] [ones|random] [fixed|orig]
 *
 * With no arguments it runs the original configuration: a 5x5 signal and a
 * 5x5 filter that are both all ones.
 *
 *   N            - grid is N x N complex values
 *   ones|random  - input fill: all ones (original) or uniform random in [0,1)
 *   fixed|orig   - pointwise multiply kernel: "fixed" uses a correct complex
 *                  multiply and normalizes the inverse FFT by 1/(N*N);
 *                  "orig" runs the original kernel and launch exactly as
 *                  written (<<<N, N>>>, no normalization, and a bug that
 *                  overwrites a[i].x before a[i].y is computed)
 *
 * The result is checked against a direct CPU circular convolution (every
 * point for small N, a random sample for large N). Matrices are printed for
 * N <= 8.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <cufft.h>
#include <fstream>

using namespace std;
typedef float2 Complex;


//Found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

#define CUFFT_CHECK(call) do { cufftResult r_ = (call); if (r_ != CUFFT_SUCCESS) { \
        printf("cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

/* The original kernel, unchanged. a[i].x is overwritten on the first line, so
   the second line uses the *new* real part. That is only harmless when every
   spectrum value is purely real (e.g. the all-ones input). There is also no
   bounds check, so the launch must cover exactly N*N threads. */
__global__ void ComplexMUL_original(Complex *a, Complex *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}

/* a[i] = a[i] * b[i] * scale. Both inputs are read into registers before
   either output is written. scale = 1/(N*N) undoes cuFFT's unnormalized
   inverse transform. */
__global__ void ComplexMUL(Complex *a, Complex *b, int size, float scale)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    if (i < size)
    {
        Complex x = a[i];
        Complex y = b[i];
        a[i].x = (x.x * y.x - x.y * y.y) * scale;
        a[i].y = (x.x * y.y + x.y * y.x) * scale;
    }
}

void printGrid(const Complex *g, int N)
{
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << g[i+j].x << " ";
        }
        cout << endl;
    }
}

/* Direct circular convolution at one output point: O(N^2) work */
double convolvePoint(const Complex *s, const Complex *f, int N, int r, int c)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += (double)s[i * N + j].x * f[((r - i + N) % N) * N + ((c - j + N) % N)].x;
    return sum;
}


int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 5;
    bool randomInput = (argc > 2) && strcmp(argv[2], "random") == 0;
    bool original = (argc > 3) && strcmp(argv[3], "orig") == 0;
    int SIZE = N*N;

    if (N < 1 || (original && N > 1024)) {
        printf("Usage: %s [N] [ones|random] [fixed|orig]  (orig launches <<<N,N>>> so needs N <= 1024)\n", argv[0]);
        return EXIT_FAILURE;
    }
    printf("Config: %d x %d grid, %s input, %s multiply kernel\n\n", N, N,
           randomInput ? "random" : "all-ones", original ? "original" : "fixed");

    srand(1234);
    Complex *fg = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fg[i].x = randomInput ? (float)rand() / RAND_MAX : 1;
        fg[i].y = 0;
    }
    Complex *fig = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fig[i].x = randomInput ? (float)rand() / RAND_MAX : 1; //
        fig[i].y = 0;
    }
    if (N <= 8)
    {
        printGrid(fg, N);
        cout << "----------------" << endl;
        printGrid(fig, N);
        cout << "----------------" << endl;
    }

    size_t mem_size = sizeof(Complex)* SIZE;


    cufftComplex *d_signal;
    CUDA_CHECK(cudaMalloc((void **) &d_signal, mem_size));
    CUDA_CHECK(cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice));

    cufftComplex *d_filter_kernel;
    CUDA_CHECK(cudaMalloc((void **)&d_filter_kernel, mem_size));
    CUDA_CHECK(cudaMemcpy(d_filter_kernel, fig, mem_size, cudaMemcpyHostToDevice));

    // CUFFT plan: cuFFT picks an algorithm for this size/type and allocates
    // its work area once; the same plan is reused for every transform below.
    auto planStart = chrono::high_resolution_clock::now();
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, N, N, CUFFT_C2C));
    size_t workSize = 0;
    CUFFT_CHECK(cufftGetSize(plan, &workSize));
    double planMs = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - planStart).count();

    cudaEvent_t ev[4];
    for (int e = 0; e < 4; e++) CUDA_CHECK(cudaEventCreate(&ev[e]));

    // Transform signal and filter
    printf("Transforming signal cufftExecC2C\n");
    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD));
    CUDA_CHECK(cudaEventRecord(ev[1]));

    printf("Launching Complex multiplication<<< >>>\n");
    if (original)
    {
        ComplexMUL_original <<< N, N >>>(d_signal, d_filter_kernel);
    }
    else
    {
        int threads = 256;
        int blocks = (SIZE + threads - 1) / threads;
        ComplexMUL <<< blocks, threads >>>(d_signal, d_filter_kernel, SIZE, 1.0f / SIZE);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));
    CUDA_CHECK(cudaEventRecord(ev[3]));

    Complex *result = new Complex[SIZE];
    CUDA_CHECK(cudaMemcpy(result, d_signal, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost));

    if (N <= 8)
    {
        printGrid(result, N);
    }

    float fwdMs, mulMs, invMs;
    CUDA_CHECK(cudaEventElapsedTime(&fwdMs, ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&mulMs, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&invMs, ev[2], ev[3]));

    // Compare against a direct convolution. The original path never divides
    // by N*N, so its output is SIZE times the true convolution.
    int samples = (SIZE <= 64 * 64) ? SIZE : (int)fmax(16.0, (double)(1 << 26) / SIZE);
    double maxErr = 0.0, maxRef = 0.0;
    for (int s = 0; s < samples; s++)
    {
        int idx = (samples == SIZE) ? s : (int)(((long long)rand() * RAND_MAX + rand()) % SIZE);
        double ref = convolvePoint(fg, fig, N, idx / N, idx % N);
        double got = original ? (double)result[idx].x / SIZE : (double)result[idx].x;
        maxErr = fmax(maxErr, fabs(got - ref));
        maxErr = fmax(maxErr, fabs(original ? (double)result[idx].y / SIZE : (double)result[idx].y));
        maxRef = fmax(maxRef, fabs(ref));
    }
    double relErr = maxErr / (maxRef > 0 ? maxRef : 1.0);

    printf("\nTimings (N=%d, %d complex points, %.1f MB per array):\n", N, SIZE, mem_size / 1048576.0);
    printf("  plan creation        : %9.3f ms (host clock, cuFFT work area %.1f MB)\n", planMs, workSize / 1048576.0);
    printf("  2 forward FFTs       : %9.3f ms\n", fwdMs);
    printf("  pointwise multiply   : %9.3f ms\n", mulMs);
    printf("  inverse FFT          : %9.3f ms\n", invMs);
    printf("Check vs direct convolution (%d point%s): max rel error = %.3e -> %s\n",
           samples, samples == 1 ? "" : "s", relErr, relErr < 1e-4 ? "PASS" : "FAIL");
    if (original)
        printf("  (orig path: raw output is %d x the convolution because the inverse FFT is unnormalized)\n", SIZE);

    for (int e = 0; e < 4; e++) cudaEventDestroy(ev[e]);
    delete[] result;
    delete[] fg;
    delete[] fig;
    cufftDestroy(plan);
    //cufftDestroy(plan2);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);

    return relErr < 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
