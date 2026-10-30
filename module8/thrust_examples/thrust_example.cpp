#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/equal.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// Module 8 practical: with no arguments this is the original example. With
//   thrust_example N [seed]
// it also generates N random ints in [0, 1000) on the device, stores them in
// device and host vectors, and times Thrust algorithms against the CPU/STL.
// Build as CUDA (nvcc -x cu) so the device backend is used.

// Random value for element i: each element gets its own engine, seeded and
// advanced to position i, so the result is independent of thread order and
// identical on host and device.
struct random_int
{
  unsigned int seed;
  __host__ __device__ random_int(unsigned int s) : seed(s) {}
  __host__ __device__ int operator()(unsigned int i) const
  {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<int> dist(0, 999);
    rng.discard(i);
    return dist(rng);
  }
};

struct less_than
{
  int limit;
  __host__ __device__ less_than(int l) : limit(l) {}
  __host__ __device__ bool operator()(int x) const { return x < limit; }
};

// Simple GPU timer around a block of Thrust calls (they run on the default
// stream; the stop event waits for them).
struct gpu_timer
{
  cudaEvent_t start, stop;
  gpu_timer() { cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start); }
  ~gpu_timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  float ms()
  {
    float t;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    return t;
  }
};

double cpu_ms_since(std::chrono::high_resolution_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

void random_demo(size_t n, unsigned int seed)
{
  typedef std::chrono::high_resolution_clock clk;
  std::cout << "\n=== Random data: N = " << n << ", seed = " << seed << " ===" << std::endl;

  // warm up the CUDA context so it is not charged to the first timing
  cudaFree(0);

  // --- generation + parallel storage ---
  float t_alloc, t_gen;
  thrust::device_vector<int> D;
  {
    gpu_timer t;
    D.resize(n);          // cudaMalloc + a fill kernel (value-initialization)
    t_alloc = t.ms();
  }
  {
    gpu_timer t;
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(n),
                      D.begin(), random_int(seed));
    t_gen = t.ms();
  }

  thrust::host_vector<int> Hcpu(n);
  clk::time_point t0 = clk::now();
  std::transform(thrust::counting_iterator<unsigned int>(0),
                 thrust::counting_iterator<unsigned int>(n),
                 Hcpu.begin(), random_int(seed));
  double c_gen = cpu_ms_since(t0);

  // bulk copy device -> host (one cudaMemcpy)
  float t_copy;
  thrust::host_vector<int> H;
  {
    gpu_timer t;
    H = D;
    t_copy = t.ms();
  }

  // element-wise access D[i]: every read is its own cudaMemcpy
  size_t k = std::min<size_t>(n, 1000);
  long long s = 0;
  t0 = clk::now();
  for (size_t i = 0; i < k; ++i) s += D[i];
  double c_elem = cpu_ms_since(t0);

  bool same = thrust::equal(H.begin(), H.end(), Hcpu.begin());
  std::cout << "first values: ";
  for (size_t i = 0; i < std::min<size_t>(n, 8); ++i) std::cout << H[i] << " ";
  std::cout << "\nhost and device generators agree: " << (same ? "yes" : "NO") << std::endl;

  printf("  %-34s %10s %10s\n", "operation", "GPU ms", "CPU ms");
  printf("  %-34s %10.3f %10s\n", "device_vector allocate+fill", t_alloc, "-");
  printf("  %-34s %10.3f %10.3f\n", "generate random ints", t_gen, c_gen);
  printf("  %-34s %10.3f %10s  (%.2f GB/s)\n", "copy D -> H (bulk)", t_copy, "-",
         n * sizeof(int) / (t_copy / 1e3) / 1e9);
  printf("  %-34s %10s %10.3f  (%.1f us per element, %zu reads)\n", "read D[i] one at a time", "-",
         c_elem, c_elem * 1e3 / k, k);

  // --- algorithms on the stored data ---
  float g;
  double c;
  long long gsum, csum;
  {
    gpu_timer t;
    gsum = thrust::reduce(D.begin(), D.end(), 0LL, thrust::plus<long long>());
    g = t.ms();
  }
  t0 = clk::now();
  csum = std::accumulate(H.begin(), H.end(), 0LL);
  c = cpu_ms_since(t0);
  printf("  %-34s %10.3f %10.3f  sum=%lld mean=%.2f %s\n", "reduce (sum)", g, c, gsum,
         (double)gsum / n, gsum == csum ? "match" : "MISMATCH");

  long long gcount, ccount;
  {
    gpu_timer t;
    gcount = thrust::count_if(D.begin(), D.end(), less_than(100));
    g = t.ms();
  }
  t0 = clk::now();
  ccount = std::count_if(H.begin(), H.end(), [](int x) { return x < 100; });
  c = cpu_ms_since(t0);
  printf("  %-34s %10.3f %10.3f  %lld (%.2f%%) %s\n", "count_if (x < 100)", g, c, gcount,
         100.0 * gcount / n, gcount == ccount ? "match" : "MISMATCH");

  int gmin, gmax;
  {
    gpu_timer t;
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> mm =
        thrust::minmax_element(D.begin(), D.end());
    gmin = *mm.first;
    gmax = *mm.second;
    g = t.ms();
  }
  t0 = clk::now();
  std::pair<thrust::host_vector<int>::iterator, thrust::host_vector<int>::iterator> cmm =
      std::minmax_element(H.begin(), H.end());
  c = cpu_ms_since(t0);
  printf("  %-34s %10.3f %10.3f  min=%d max=%d %s\n", "minmax_element", g, c, gmin, gmax,
         (gmin == *cmm.first && gmax == *cmm.second) ? "match" : "MISMATCH");

  {
    gpu_timer t;
    thrust::sort(D.begin(), D.end());
    g = t.ms();
  }
  t0 = clk::now();
  std::sort(H.begin(), H.end());
  c = cpu_ms_since(t0);
  // thrust::equal needs both ranges in the same memory space, so bring the
  // device result back first
  thrust::host_vector<int> Dsorted = D;
  bool sorted_same = thrust::equal(Dsorted.begin(), Dsorted.end(), H.begin());
  printf("  %-34s %10.3f %10.3f  %s\n", "sort", g, c, sorted_same ? "match" : "MISMATCH");
}

int main(int argc, char **argv)
{
  // H has storage for 4 integers
  thrust::host_vector<int> H(4);

  // initialize individual elements
  H[0] = 14;
  H[1] = 20;
  H[2] = 38;
  H[3] = 46;

  // H.size() returns the size of vector H
  std::cout << "H has size " << H.size() << std::endl;

  // print contents of H
  for(int i = 0; i < H.size(); i++)
  {
    std::cout << "H[" << i << "] = " << H[i] << std::endl;
  }

  // resize H
  H.resize(2);

  std::cout << "H now has size " << H.size() << std::endl;

  // Copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;

  // elements of D can be modified
  D[0] = 99;
  D[1] = 88;

  // print contents of D
  for(int i = 0; i < D.size(); i++)
  {
    std::cout << "D[" << i << "] = " << D[i] << std::endl;
  }

  if (argc > 1)
  {
    size_t n = strtoull(argv[1], NULL, 10);
    unsigned int seed = argc > 2 ? (unsigned int)strtoul(argv[2], NULL, 10) : 1234u;
    random_demo(n, seed);
  }

  // H and D are automatically destroyed when the function returns
  return 0;
}
