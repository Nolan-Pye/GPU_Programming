// Shared constants and error-checking helper for the CUDA Streams and
// Events assignment.
#pragma once

#include <cstddef>
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

// Command-line defaults and limits.
constexpr int DEFAULT_BLOCK_X = 16;
constexpr int DEFAULT_BLOCK_Y = 16;
constexpr int DEFAULT_BATCH_SIZE = 32;
constexpr int MAX_BLOCK_THREADS = 1024;
constexpr const char* PHOTO_DIR = "images";
constexpr const char* OUTPUT_DIR = "output";

// Gaussian blur: 2 * BLUR_RADIUS + 1 taps per pass, standard deviation
// BLUR_SIGMA pixels. The radius sets how much arithmetic each kernel does.
constexpr int BLUR_RADIUS = 12;
constexpr int BLUR_TAPS = 2 * BLUR_RADIUS + 1;
constexpr float BLUR_SIGMA = 6.0f;

// Stream counts tried by the "N streams" variants (the batch is split
// into one group of photos per stream).
constexpr int STREAM_COUNTS[] = {1, 2, 4, 8};
constexpr int NUM_STREAM_COUNTS =
    sizeof(STREAM_COUNTS) / sizeof(STREAM_COUNTS[0]);
constexpr int MAX_STREAMS = STREAM_COUNTS[NUM_STREAM_COUNTS - 1];

// Every variant gets one untimed warm-up run, then this many timed runs.
constexpr int NUM_TIMED_RUNS = 3;

// The CPU check allows each colour channel to differ by this much: host
// and device may round the floating-point weighted sum differently.
constexpr int VERIFY_TOLERANCE = 1;

// Launch geometry and batch size, from the command line.
struct LaunchConfig {
    int blockX;     // threads per block along x (image columns)
    int blockY;     // threads per block along y (image rows)
    int batchSize;  // photos blurred per run
};
