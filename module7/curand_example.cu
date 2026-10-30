/*
 * Usage: curand_example [blocks] [threads_per_block] [max] [seed]
 *
 * With no arguments it runs the original configuration: 25 blocks of 20
 * threads, each thread drawing one random int in [0, 100), seeded from the
 * clock.
 *
 *   blocks            - number of blocks (N in the original)
 *   threads_per_block - threads per block (M in the original, 1..1024)
 *   max               - numbers are drawn in [0, max)
 *   seed              - fixed seed (repeatable output); omit to seed from time()
 *
 * Small grids print every number like the original; large grids print summary
 * statistics and a histogram check instead. A final section generates the same
 * amount of data with the cuRAND *host* API for comparison.
 */

#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define MAX 100
#define N 25
#define M 20

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

#define CURAND_CHECK(call) do { curandStatus_t s_ = (call); if (s_ != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              thread_idx, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[thread_idx]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers, unsigned int max) {
  /* curand works like rand - except that it takes a state as a parameter */
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  numbers[thread_idx] = curand(&states[thread_idx]) % max;
}

/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random0(int* result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(1, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % MAX;
}

/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random1(unsigned int seed, int* result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % MAX;
}

/* Mean, min, max and a chi-square uniformity statistic over [0, max) */
void printStats(const unsigned int *nums, int count, unsigned int max) {
  unsigned int *hist = (unsigned int *)calloc(max, sizeof(unsigned int));
  double sum = 0.0;
  unsigned int lo = max, hi = 0;
  for (int i = 0; i < count; i++) {
    unsigned int v = nums[i];
    sum += v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    hist[v]++;
  }
  double expected = (double)count / max;
  double chi2 = 0.0;
  for (unsigned int b = 0; b < max; b++) {
    double d = hist[b] - expected;
    chi2 += d * d / expected;
  }
  /* For a uniform source chi2 has mean (max-1) and std dev sqrt(2*(max-1)) */
  double dof = max - 1.0;
  printf("  count=%d  min=%u  max=%u  mean=%.3f (ideal %.3f)\n",
         count, lo, hi, sum / count, (max - 1) / 2.0);
  printf("  chi-square=%.1f over %u bins (ideal %.0f +/- %.1f): %s\n",
         chi2, max, dof, sqrt(2.0 * dof),
         fabs(chi2 - dof) < 4.0 * sqrt(2.0 * dof) ? "looks uniform" : "NOT uniform");
  free(hist);
}

int main(int argc, char **argv) {
  int blocks = (argc > 1) ? atoi(argv[1]) : N;
  int threads = (argc > 2) ? atoi(argv[2]) : M;
  unsigned int max = (argc > 3) ? (unsigned int)atoi(argv[3]) : MAX;
  unsigned int seed = (argc > 4) ? (unsigned int)strtoul(argv[4], NULL, 10) : (unsigned int)time(NULL);
  int total = blocks * threads;

  if (blocks < 1 || threads < 1 || threads > 1024 || max < 1) {
    printf("Usage: %s [blocks] [threads_per_block(1..1024)] [max] [seed]\n", argv[0]);
    return EXIT_FAILURE;
  }
  printf("Config: %d blocks x %d threads = %d numbers in [0, %u), seed=%u%s\n\n",
         blocks, threads, total, max, seed, (argc > 4) ? " (fixed)" : " (from time)");

  /* allocate an int on the GPU */
  int* gpu_x0;
  CUDA_CHECK(cudaMalloc((void**) &gpu_x0, sizeof(int)));

  /* invoke the GPU to initialize all of the random states */
  random0<<<1, 1>>>(gpu_x0);
  CUDA_CHECK(cudaGetLastError());

  /* copy the random number back */
  int x0;
  CUDA_CHECK(cudaMemcpy(&x0, gpu_x0, sizeof(int), cudaMemcpyDeviceToHost));

  printf("random0 (seed fixed at 1)   : Random number = %d.\n", x0);

  /* free the memory we allocated */
  cudaFree(gpu_x0);

   /* allocate an int on the GPU */
  int* gpu_x1;
  CUDA_CHECK(cudaMalloc((void**) &gpu_x1, sizeof(int)));

  /* invoke the GPU to initialize all of the random states */
  random1<<<1, 1>>>(seed, gpu_x1);
  CUDA_CHECK(cudaGetLastError());

  /* copy the random number back */
  int x1;
  CUDA_CHECK(cudaMemcpy(&x1, gpu_x1, sizeof(int), cudaMemcpyDeviceToHost));

  printf("random1 (seed=%u): Random number = %d.\n\n", seed, x1);

  /* free the memory we allocated */
  cudaFree(gpu_x1);

  cudaEvent_t start, stop;
  float initMs, genMs;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

   /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  CUDA_CHECK(cudaMalloc((void**) &states, total * sizeof(curandState_t)));

  /* invoke the GPU to initialize all of the random states */
  CUDA_CHECK(cudaEventRecord(start));
  init<<<blocks, threads>>>(seed, states);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&initMs, start, stop));

  /* allocate an array of unsigned ints on the CPU and GPU (heap, not stack,
     so large grids don't overflow the stack) */
  unsigned int *cpu_nums = (unsigned int *)malloc(total * sizeof(unsigned int));
  unsigned int* gpu_nums;
  CUDA_CHECK(cudaMalloc((void**) &gpu_nums, total * sizeof(unsigned int)));

  /* invoke the kernel to get some random numbers */
  CUDA_CHECK(cudaEventRecord(start));
  randoms<<<blocks, threads>>>(states, gpu_nums, max);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&genMs, start, stop));

  /* copy the random numbers back */
  CUDA_CHECK(cudaMemcpy(cpu_nums, gpu_nums, total * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  /* print them out (one row per block) when the grid is small */
  if (blocks <= 32 && threads <= 32) {
    for (int i = 0; i < blocks; i++) {
        for (int j =0; j < threads; j++){
            printf("%u ", cpu_nums[i * threads + j]);
        }
        printf("\n");
    }
  }

  printf("Device API (curand_init + curand per thread, XORWOW state %d bytes):\n",
         (int)sizeof(curandState_t));
  printf("  init kernel     : %8.3f ms\n", initMs);
  printf("  generate kernel : %8.3f ms\n", genMs);
  printStats(cpu_nums, total, max);

  /* Same amount of data through the host API: the library launches its own
     kernels to fill a device buffer, no per-thread state to manage. */
  curandGenerator_t gen;
  unsigned int *gpu_raw;
  CUDA_CHECK(cudaMalloc((void**) &gpu_raw, total * sizeof(unsigned int)));
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

  CUDA_CHECK(cudaEventRecord(start));
  CURAND_CHECK(curandGenerate(gen, gpu_raw, total));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float firstMs;
  CUDA_CHECK(cudaEventElapsedTime(&firstMs, start, stop));

  CUDA_CHECK(cudaEventRecord(start));
  CURAND_CHECK(curandGenerate(gen, gpu_raw, total));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float hostMs;
  CUDA_CHECK(cudaEventElapsedTime(&hostMs, start, stop));

  CUDA_CHECK(cudaMemcpy(cpu_nums, gpu_raw, total * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < total; i++) cpu_nums[i] %= max;

  printf("Host API (curandGenerate, XORWOW):\n");
  printf("  first call (incl. state setup) : %8.3f ms\n", firstMs);
  printf("  second call                    : %8.3f ms\n", hostMs);
  printStats(cpu_nums, total, max);

  /* free the memory we allocated for the states and numbers */
  CURAND_CHECK(curandDestroyGenerator(gen));
  cudaFree(gpu_raw);
  cudaFree(states);
  cudaFree(gpu_nums);
  free(cpu_nums);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
