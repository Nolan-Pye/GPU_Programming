// CUDA Memory assignment — Module 5
//
// Counts how often each letter a-z appears in a text file (War and Peace
// by default). The same counting code runs once per CUDA memory space
// (host, global, constant, shared, register) plus one kernel that combines
// them all. Each variant is timed with cudaEvents and must match a CPU
// reference histogram computed in host memory exactly.
//
// Usage: memory_assignment [totalThreads] [blockSize] [textFile]

#include <chrono>
#include <cstring>
#include "common.h"
#include "histogram_kernels.cuh"
#include "text_utils.h"

// All buffers used by one run.
struct Buffers {
    Text text;                     // pageable host memory (malloc)
    signed char table[TABLE_SIZE]; // byte -> letter bin, host copy
    unsigned int cpuHist[NUM_BINS];
    unsigned int gpuHist[NUM_BINS];
    unsigned char* pinnedText;     // pinned + mapped host memory
    unsigned char* mappedText;     // device-side alias of pinnedText
    unsigned char* devText;        // device global memory
    signed char* devTable;         // device global memory
    unsigned int* devHist;         // device global memory
};

// Parses [totalThreads] [blockSize] [textFile] and rounds the thread count
// up to a whole number of blocks.
static LaunchConfig parseArgs(int argc, char** argv)
{
    LaunchConfig cfg{DEFAULT_THREADS, DEFAULT_BLOCK_SIZE, 0,
                     DEFAULT_TEXT_FILE};
    if (argc >= 2) cfg.totalThreads = atoi(argv[1]);
    if (argc >= 3) cfg.blockSize = atoi(argv[2]);
    if (argc >= 4) cfg.textFile = argv[3];

    if (cfg.blockSize < 1 || cfg.blockSize > MAX_BLOCK_SIZE ||
        cfg.totalThreads < 1) {
        fprintf(stderr, "Usage: %s [totalThreads] [blockSize 1-%d] "
                "[textFile]\n", argv[0], MAX_BLOCK_SIZE);
        exit(EXIT_FAILURE);
    }
    if (cfg.totalThreads < MIN_THREADS) {
        printf("Note: assignment asks for at least %d threads\n",
               MIN_THREADS);
    }
    cfg.numBlocks = (cfg.totalThreads + cfg.blockSize - 1) / cfg.blockSize;
    if (cfg.numBlocks * cfg.blockSize != cfg.totalThreads) {
        cfg.totalThreads = cfg.numBlocks * cfg.blockSize;
        printf("Warning: thread count not divisible by block size; "
               "rounded up to %d\n", cfg.totalThreads);
    }
    return cfg;
}

static void allocBuffers(Buffers& b)
{
    size_t n = b.text.size;
    CUDA_CHECK(cudaHostAlloc((void**)&b.pinnedText, n,
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&b.mappedText,
                                        b.pinnedText, 0));
    memcpy(b.pinnedText, b.text.data, n);

    CUDA_CHECK(cudaMalloc((void**)&b.devText, n));
    CUDA_CHECK(cudaMalloc((void**)&b.devTable, TABLE_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&b.devHist,
                          NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(b.devTable, b.table, TABLE_SIZE,
                          cudaMemcpyHostToDevice));
}

static void freeBuffers(Buffers& b)
{
    free(b.text.data);
    CUDA_CHECK(cudaFreeHost(b.pinnedText));
    CUDA_CHECK(cudaFree(b.devText));
    CUDA_CHECK(cudaFree(b.devTable));
    CUDA_CHECK(cudaFree(b.devHist));
}

// Times `op` (any sequence of async GPU work) averaged over TIMING_ITERS
// runs, after one untimed warm-up run.
template <typename Op>
static float timeGpuMs(Op op)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    op();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < TIMING_ITERS; ++it) {
        op();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / TIMING_ITERS;
}

// Times the host-memory CPU reference, averaged over TIMING_ITERS runs.
static float timeCpuMs(Buffers& b)
{
    auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < TIMING_ITERS; ++it) {
        cpuHistogram(b.text.data, b.text.size, b.table, b.cpuHist);
    }
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> ms = stop - start;
    return ms.count() / TIMING_ITERS;
}

// Host memory comparison: host->device copy from pageable vs pinned memory.
// Also leaves devText populated for the device-memory variants.
static void reportTransfers(Buffers& b)
{
    size_t n = b.text.size;
    float pageableMs = timeGpuMs([&] {
        CUDA_CHECK(cudaMemcpy(b.devText, b.text.data, n,
                              cudaMemcpyHostToDevice));
    });
    float pinnedMs = timeGpuMs([&] {
        CUDA_CHECK(cudaMemcpy(b.devText, b.pinnedText, n,
                              cudaMemcpyHostToDevice));
    });
    double gb = (double)n / 1e9;
    printf("\nHost->device copy of the text:\n");
    printf("  pageable host memory: %9.4f ms  (%6.2f GB/s)\n",
           pageableMs, gb / (pageableMs / 1000.0));
    printf("  pinned host memory:   %9.4f ms  (%6.2f GB/s)\n",
           pinnedMs, gb / (pinnedMs / 1000.0));
}

// Runs and times one variant (histogram reset + kernel), then reports
// whether its counts exactly match the CPU reference.
static bool runVariant(Variant v, Buffers& b, const LaunchConfig& cfg,
                       float* avgMs)
{
    const size_t histBytes = NUM_BINS * sizeof(unsigned int);
    const unsigned char* text =
        (v == Variant::HostMapped) ? b.mappedText : b.devText;

    *avgMs = timeGpuMs([&] {
        CUDA_CHECK(cudaMemsetAsync(b.devHist, 0, histBytes));
        launchVariant(v, text, b.text.size, b.devTable, b.devHist, cfg);
    });
    CUDA_CHECK(cudaMemcpy(b.gpuHist, b.devHist, histBytes,
                          cudaMemcpyDeviceToHost));
    return memcmp(b.gpuHist, b.cpuHist, histBytes) == 0;
}

static void printRow(const char* name, const char* memory, float ms,
                     float globalMs, const char* status)
{
    printf("  %-14s %-31s %9.4f %7.2fx  %s\n", name, memory, ms,
           globalMs / ms, status);
}

// Runs every variant and prints a timing table relative to global memory.
static bool runAllVariants(Buffers& b, const LaunchConfig& cfg)
{
    const int count = sizeof(ALL_VARIANTS) / sizeof(ALL_VARIANTS[0]);
    float ms[count];
    bool pass[count];
    float globalMs = 1.0f;
    for (int i = 0; i < count; ++i) {
        pass[i] = runVariant(ALL_VARIANTS[i], b, cfg, &ms[i]);
        if (ALL_VARIANTS[i] == Variant::Global) globalMs = ms[i];
    }
    float cpuMs = timeCpuMs(b);

    bool allPass = true;
    printf("\nHistogram timings (avg of %d runs, identical counting code)"
           ":\n", TIMING_ITERS);
    printf("  %-14s %-31s %9s %8s  %s\n", "variant", "memory used", "ms",
           "vs glob", "check");
    printRow("cpu", "text/table/hist: host", cpuMs, globalMs, "reference");
    for (int i = 0; i < count; ++i) {
        allPass = allPass && pass[i];
        printRow(variantName(ALL_VARIANTS[i]),
                 variantMemory(ALL_VARIANTS[i]), ms[i], globalMs,
                 pass[i] ? "PASS" : "FAIL");
    }
    return allPass;
}

static void printDeviceInfo(const LaunchConfig& cfg, size_t textBytes)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Text: %s (%zu bytes)\n", cfg.textFile, textBytes);
    printf("Threads: %d   Block size: %d   Blocks: %d   "
           "Bytes/thread: ~%zu\n", cfg.totalThreads, cfg.blockSize,
           cfg.numBlocks, (textBytes + cfg.totalThreads - 1) /
           (size_t)cfg.totalThreads);
    if (!prop.canMapHostMemory) {
        fprintf(stderr, "Device cannot map host memory\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    LaunchConfig cfg = parseArgs(argc, argv);
    // Must precede any call that creates the CUDA context.
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    Buffers b;
    b.text = loadText(cfg.textFile);
    printDeviceInfo(cfg, b.text.size);
    buildLetterTable(b.table);
    cpuHistogram(b.text.data, b.text.size, b.table, b.cpuHist);

    allocBuffers(b);
    uploadConstantTable(b.table);
    reportTransfers(b);

    bool ok = runAllVariants(b, cfg);
    printLetterChart(b.cpuHist);
    freeBuffers(b);
    printf("\n%s\n", ok ? "All GPU variants match the CPU counts exactly."
                        : "ERROR: some variants do not match.");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
