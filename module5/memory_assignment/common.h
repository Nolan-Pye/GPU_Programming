// Shared constants and error-checking helper for the Memory assignment.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// One histogram bin per letter a-z (case-insensitive).
constexpr int NUM_BINS = 26;
// Lookup-table entry for any byte that is not an ASCII letter.
constexpr signed char NOT_A_LETTER = -1;
// One entry per possible byte value.
constexpr int TABLE_SIZE = 256;

// Command-line defaults and limits.
constexpr int DEFAULT_THREADS = 1 << 16;
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int MIN_THREADS = 64;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr const char* DEFAULT_TEXT_FILE = "war_and_peace.txt";

// Timed launches per variant (averaged), after one untimed warm-up.
constexpr int TIMING_ITERS = 20;

// Launch geometry and input file, from the command line.
struct LaunchConfig {
    int totalThreads;
    int blockSize;
    int numBlocks;
    const char* textFile;
};
