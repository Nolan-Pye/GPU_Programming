// CUDA Threads and Blocks assignment — Module 3
//
// Runs the same simple element-wise array operation four ways: on the CPU
// and on the GPU, each once with no branching and once with a conditional
// branch added, to compare (1) CPU vs. GPU throughput and (2) the cost of
// a branch on each architecture (warp divergence on the GPU).
//
// Usage: assignment.exe <totalThreads> <blockSize>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Monotonic wall-clock time in milliseconds (QueryPerformanceCounter on
// Windows, clock_gettime on Linux — same call site either way).
static double nowMs(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int haveFreq = 0;
    LARGE_INTEGER counter;
    if (!haveFreq) {
        QueryPerformanceFrequency(&freq);
        haveFreq = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

// c[i] = a[i] * b[i] + a[i], no branching.
__global__ void gpuBaselineKernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] * b[idx] + a[idx];
}

// Same algorithm, but every other thread takes a different arithmetic path.
// Because the branch is on idx parity, every warp splits 50/50 and serializes
// both halves — worst-case divergence.
__global__ void gpuBranchingKernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        c[idx] = a[idx] * b[idx] + a[idx];
    } else {
        c[idx] = a[idx] - b[idx] * b[idx];
    }
}

void cpuBaseline(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] * b[i] + a[i];
    }
}

// Mirrors gpuBranchingKernel's branch, so the CPU and GPU cost of the same
// conditional can be compared directly.
void cpuBranching(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            c[i] = a[i] * b[i] + a[i];
        } else {
            c[i] = a[i] - b[i] * b[i];
        }
    }
}

float maxAbsDiff(const float* x, const float* y, int n)
{
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(x[i] - y[i]);
        if (diff > worst) worst = diff;
    }
    return worst;
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = totalThreads / blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    const int n = totalThreads;
    printf("Running with n=%d elements, numBlocks=%d, blockSize=%d\n", n, numBlocks, blockSize);

    // host buffers — the CPU functions below use n, not blockSize/numBlocks;
    // there's no equivalent CPU concept for those.
    size_t bytes = (size_t)n * sizeof(float);
    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* cCpuBase = (float*)malloc(bytes);
    float* cCpuBranch = (float*)malloc(bytes);
    float* cGpuBase = (float*)malloc(bytes);
    float* cGpuBranch = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(n - i) * 0.25f;
    }

    // device buffers
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));
    float gpuBaseMs = 0.0f, gpuBranchMs = 0.0f;

    // Warm-up launch: the first kernel launch on a fresh context pays a
    // one-time driver/JIT cost that has nothing to do with the algorithm.
    // Absorb that cost here so the two timed kernels below are compared
    // fairly against each other.
    gpuBaselineKernel<<<numBlocks, blockSize>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- GPU baseline (no branching) ---
    CUDA_CHECK(cudaEventRecord(startEvt));
    gpuBaselineKernel<<<numBlocks, blockSize>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaEventSynchronize(stopEvt));
    CUDA_CHECK(cudaEventElapsedTime(&gpuBaseMs, startEvt, stopEvt));
    CUDA_CHECK(cudaMemcpy(cGpuBase, dC, bytes, cudaMemcpyDeviceToHost));

    // --- GPU branching ---
    CUDA_CHECK(cudaEventRecord(startEvt));
    gpuBranchingKernel<<<numBlocks, blockSize>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaEventSynchronize(stopEvt));
    CUDA_CHECK(cudaEventElapsedTime(&gpuBranchMs, startEvt, stopEvt));
    CUDA_CHECK(cudaMemcpy(cGpuBranch, dC, bytes, cudaMemcpyDeviceToHost));

    // --- CPU baseline (no branching) ---
    double cpuBaseStart = nowMs();
    cpuBaseline(a, b, cCpuBase, n);
    double cpuBaseMs = nowMs() - cpuBaseStart;

    // --- CPU branching ---
    double cpuBranchStart = nowMs();
    cpuBranching(a, b, cCpuBranch, n);
    double cpuBranchMs = nowMs() - cpuBranchStart;

    // correctness sanity check — CPU and GPU should agree on both variants
    float baseDiff = maxAbsDiff(cCpuBase, cGpuBase, n);
    float branchDiff = maxAbsDiff(cCpuBranch, cGpuBranch, n);

    printf("CPU baseline:   %8.4f ms\n", cpuBaseMs);
    printf("GPU baseline:   %8.4f ms   (max diff vs CPU: %g)\n", gpuBaseMs, baseDiff);
    printf("CPU branching:  %8.4f ms\n", cpuBranchMs);
    printf("GPU branching:  %8.4f ms   (max diff vs CPU: %g)\n", gpuBranchMs, branchDiff);

    // append this run's numbers to a CSV so results across multiple
    // invocations (different thread counts / block sizes) can be charted
    const char* csvPath = "results.csv";
    FILE* csv = fopen(csvPath, "r");
    bool needsHeader = (csv == nullptr);
    if (csv) fclose(csv);

    csv = fopen(csvPath, "a");
    if (csv) {
        if (needsHeader) {
            fprintf(csv, "total_threads,block_size,num_blocks,cpu_baseline_ms,gpu_baseline_ms,cpu_branching_ms,gpu_branching_ms\n");
        }
        fprintf(csv, "%d,%d,%d,%f,%f,%f,%f\n",
                totalThreads, blockSize, numBlocks, cpuBaseMs, gpuBaseMs, cpuBranchMs, gpuBranchMs);
        fclose(csv);
    } else {
        fprintf(stderr, "Warning: could not open %s to record results\n", csvPath);
    }

    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(a);
    free(b);
    free(cCpuBase);
    free(cCpuBranch);
    free(cGpuBase);
    free(cGpuBranch);

    return 0;
}
