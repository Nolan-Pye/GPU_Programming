// Letter-histogram kernels, one per CUDA memory space. The per-byte work
// lives in a single template, countByte(), so every variant runs identical
// counting code and timing differences come only from where the text, the
// lookup table, and the histogram are stored.

#include "histogram_kernels.cuh"

// ---- Constant memory: read-only, cached, broadcast to a whole warp ----
__constant__ signed char cLetterBin[TABLE_SIZE];

// ---- Lookup-table "views": same operator[] over different memory ----

struct GlobalTable {
    const signed char* p;  // device global memory
    __device__ int operator[](unsigned char c) const { return p[c]; }
};

struct ConstantTable {
    __device__ int operator[](unsigned char c) const
    {
        return cLetterBin[c];
    }
};

// Atomic histogram over whichever memory `bins` points to: global memory
// (shared by the whole grid) or shared memory (private to one block).
struct AtomicHist {
    unsigned int* bins;
    __device__ void add(int bin) const { atomicAdd(&bins[bin], 1u); }
};

// The one piece of counting code every variant shares.
template <typename T>
__device__ __forceinline__ void countByte(unsigned char c, const T& table,
                                          const AtomicHist& hist)
{
    int bin = table[c];
    if (bin != NOT_A_LETTER) {
        hist.add(bin);
    }
}

__device__ __forceinline__ size_t globalThreadId()
{
    return (size_t)blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t gridThreads()
{
    return (size_t)gridDim.x * blockDim.x;
}

// Grid-stride loop: any thread count covers the whole text, one byte per
// step. Neighbouring threads read neighbouring bytes (coalesced).
template <typename T>
__device__ void countBytes(const unsigned char* text, size_t begin,
                           size_t n, const T& table, const AtomicHist& h)
{
    for (size_t i = begin + globalThreadId(); i < n; i += gridThreads()) {
        countByte(text[i], table, h);
    }
}

// Counts the four bytes packed in one 32-bit register.
template <typename T>
__device__ __forceinline__ void countWord(unsigned int word, const T& table,
                                          const AtomicHist& h)
{
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        countByte((unsigned char)(word >> (8 * b)), table, h);
    }
}

// Register variant of the loop: each step loads 16 bytes with one uint4
// load into four registers and counts them from there. Leftover bytes
// (n not a multiple of 16) go through the plain byte loop.
template <typename T>
__device__ void countVectors(const unsigned char* text, size_t n,
                             const T& table, const AtomicHist& h)
{
    const uint4* vec = reinterpret_cast<const uint4*>(text);
    size_t nVec = n / sizeof(uint4);
    for (size_t i = globalThreadId(); i < nVec; i += gridThreads()) {
        uint4 v = vec[i];
        countWord(v.x, table, h);
        countWord(v.y, table, h);
        countWord(v.z, table, h);
        countWord(v.w, table, h);
    }
    countBytes(text, nVec * sizeof(uint4), n, table, h);
}

// ---- Shared-memory histogram helpers ----

__device__ void clearSharedHist(unsigned int* sHist)
{
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        sHist[b] = 0;
    }
    __syncthreads();
}

// Adds this block's private counts into the global histogram: NUM_BINS
// global atomics per block instead of one per letter.
__device__ void flushSharedHist(const unsigned int* sHist,
                                unsigned int* hist)
{
    __syncthreads();
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (sHist[b] != 0) {
            atomicAdd(&hist[b], sHist[b]);
        }
    }
}

// ---- Kernels ----

// Reads `text` from the pointer it is given: device global memory for
// Variant::Global, mapped host memory for Variant::HostMapped.
__global__ void globalKernel(const unsigned char* text, size_t n,
                             const signed char* table, unsigned int* hist)
{
    countBytes(text, 0, n, GlobalTable{table}, AtomicHist{hist});
}

__global__ void constantKernel(const unsigned char* text, size_t n,
                               unsigned int* hist)
{
    countBytes(text, 0, n, ConstantTable{}, AtomicHist{hist});
}

__global__ void sharedKernel(const unsigned char* text, size_t n,
                             const signed char* table, unsigned int* hist)
{
    __shared__ unsigned int sHist[NUM_BINS];
    clearSharedHist(sHist);
    countBytes(text, 0, n, GlobalTable{table}, AtomicHist{sHist});
    flushSharedHist(sHist, hist);
}

__global__ void registerKernel(const unsigned char* text, size_t n,
                               const signed char* table, unsigned int* hist)
{
    countVectors(text, n, GlobalTable{table}, AtomicHist{hist});
}

// All device memory spaces in one kernel: text from global memory into
// registers, constant lookup table, shared per-block histogram.
__global__ void combinedKernel(const unsigned char* text, size_t n,
                               unsigned int* hist)
{
    __shared__ unsigned int sHist[NUM_BINS];
    clearSharedHist(sHist);
    countVectors(text, n, ConstantTable{}, AtomicHist{sHist});
    flushSharedHist(sHist, hist);
}

// ---- Host-side API ----

const char* variantName(Variant v)
{
    switch (v) {
    case Variant::HostMapped: return "host (mapped)";
    case Variant::Global:     return "global";
    case Variant::Constant:   return "constant";
    case Variant::Shared:     return "shared";
    case Variant::Register:   return "register";
    case Variant::Combined:   return "combined";
    }
    return "?";
}

const char* variantMemory(Variant v)
{
    switch (v) {
    case Variant::HostMapped: return "text: pinned host, rest global";
    case Variant::Global:     return "text/table/hist: global";
    case Variant::Constant:   return "table: constant";
    case Variant::Shared:     return "hist: shared per block";
    case Variant::Register:   return "text: 16B loads into registers";
    case Variant::Combined:   return "regs + constant + shared";
    }
    return "?";
}

void uploadConstantTable(const signed char* hostTable)
{
    CUDA_CHECK(cudaMemcpyToSymbol(cLetterBin, hostTable,
                                  TABLE_SIZE * sizeof(signed char)));
}

void launchVariant(Variant v, const unsigned char* text, size_t n,
                   const signed char* table, unsigned int* hist,
                   const LaunchConfig& cfg)
{
    dim3 grid(cfg.numBlocks), block(cfg.blockSize);
    switch (v) {
    case Variant::HostMapped:  // same kernel, host pointer
    case Variant::Global:
        globalKernel<<<grid, block>>>(text, n, table, hist);
        break;
    case Variant::Constant:
        constantKernel<<<grid, block>>>(text, n, hist);
        break;
    case Variant::Shared:
        sharedKernel<<<grid, block>>>(text, n, table, hist);
        break;
    case Variant::Register:
        registerKernel<<<grid, block>>>(text, n, table, hist);
        break;
    case Variant::Combined:
        combinedKernel<<<grid, block>>>(text, n, hist);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}
