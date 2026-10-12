/*https://cdac.in/index.aspx?id=ev_hpc_gpu-comp-nvidia-cuda-streams#hetr-cuda-prog-cuda-streams*/

/*
 * Usage: stream_example [array_size] [threads_per_block] [num_streams]
 *
 * With no arguments it runs the original configuration: 1024*1024 ints,
 * 1 thread per block (one block per element) and a single stream.
 *
 *   array_size        - number of ints to add
 *   threads_per_block - block size for the kernel launch (1..1024)
 *   num_streams       - the array is split into this many chunks; each chunk's
 *                       H2D copies, kernel and D2H copy are issued into its own
 *                       stream so copies in one stream can overlap compute in
 *                       another
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define sizeOfArray (1024*1024)

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

__global__ void arrayAddition(int *device_a, int *device_b, int *device_result, int n)
{

	int threadId = threadIdx.x + blockIdx.x * blockDim.x ;

	if (threadId < n)
        device_result[threadId]= device_a[threadId]+device_b[threadId];
}

/* Issue copy-in, kernel, copy-out for every chunk, each chunk in its own stream */
void issueWork(int *host_a, int *host_b, int *host_result,
               int *device_a, int *device_b, int *device_result,
               int n, int threadsPerBlock, cudaStream_t *streams, int numStreams)
{
  int chunk = (n + numStreams - 1) / numStreams;

  for (int s = 0; s < numStreams; s++)
  {
    int offset = s * chunk;
    if (offset >= n) break;
    int count = (offset + chunk <= n) ? chunk : n - offset;
    size_t bytes = count * sizeof(int);
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemcpyAsync(device_a + offset, host_a + offset, bytes, cudaMemcpyHostToDevice, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(device_b + offset, host_b + offset, bytes, cudaMemcpyHostToDevice, streams[s]));

    /*Kernel call*/
    arrayAddition <<<blocks, threadsPerBlock, 0, streams[s]>>>(device_a + offset, device_b + offset, device_result + offset, count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(host_result + offset, device_result + offset, bytes, cudaMemcpyDeviceToHost, streams[s]));
  }
}

int main ( int argc, char **argv )
{
  int n = (argc > 1) ? atoi(argv[1]) : sizeOfArray;
  int threadsPerBlock = (argc > 2) ? atoi(argv[2]) : 1;
  int numStreams = (argc > 3) ? atoi(argv[3]) : 1;

  if (n < 1 || threadsPerBlock < 1 || threadsPerBlock > 1024 || numStreams < 1)
  {
    printf("Usage: %s [array_size>=1] [threads_per_block 1..1024] [num_streams>=1]\n", argv[0]);
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop;
  int *host_a, *host_b, *host_result;
  int *device_a, *device_b, *device_result;
  int whichDevice;

  /* The original passed the device *count* to cudaGetDeviceProperties, which is
     one past the last valid device index; ask for the current device instead. */
  CUDA_CHECK(cudaGetDevice( &whichDevice));
  CUDA_CHECK(cudaGetDeviceProperties( &prop, whichDevice));

  cudaEvent_t start, stop;
  float elapsedTime;

  CUDA_CHECK(cudaEventCreate( &start ));
  CUDA_CHECK(cudaEventCreate( &stop ));

  cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));
  for (int s = 0; s < numStreams; s++)
    CUDA_CHECK(cudaStreamCreate(&streams[s]));

  CUDA_CHECK(cudaMalloc( ( void**)& device_a, n * sizeof ( *device_a ) ));
  CUDA_CHECK(cudaMalloc( ( void**)& device_b, n * sizeof ( *device_b ) ));
  CUDA_CHECK(cudaMalloc( ( void**)& device_result, n * sizeof ( *device_result ) ));

  /* Pinned (page-locked) host memory - required for cudaMemcpyAsync to be truly
     asynchronous and overlap with other work */
  CUDA_CHECK(cudaHostAlloc((void **)&host_a, n*sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void **)&host_b, n*sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void **)&host_result, n*sizeof(int), cudaHostAllocDefault));

  for(int index = 0; index < n; index++)
  {
   host_a[index] = rand()%10;
   host_b[index] = rand()%10;
  }

  /* Untimed warm-up pass so one-time costs (lazy module load, first-touch of
     the buffers) don't land in the measurement */
  issueWork(host_a, host_b, host_result, device_a, device_b, device_result,
            n, threadsPerBlock, streams, numStreams);
  CUDA_CHECK(cudaDeviceSynchronize());

  /* start/stop are recorded on the legacy default stream (0), which waits for
     all other (blocking) streams, so they bracket the work in every stream */
  CUDA_CHECK(cudaEventRecord(start, 0));

  issueWork(host_a, host_b, host_result, device_a, device_b, device_result,
            n, threadsPerBlock, streams, numStreams);

  for (int s = 0; s < numStreams; s++)
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  int errors = 0;
  for (int index = 0; index < n; index++)
    if (host_result[index] != host_a[index] + host_b[index]) errors++;

  /* With one stream, also time each stage separately by recording events into
     the stream between the operations */
  float h2dTime = 0, kernelTime = 0, d2hTime = 0;
  if (numStreams == 1)
  {
    cudaEvent_t e[4];
    for (int i = 0; i < 4; i++) CUDA_CHECK(cudaEventCreate(&e[i]));
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(e[0], streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, n * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(device_b, host_b, n * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaEventRecord(e[1], streams[0]));
    arrayAddition <<<blocks, threadsPerBlock, 0, streams[0]>>>(device_a, device_b, device_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e[2], streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(host_result, device_result, n * sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaEventRecord(e[3], streams[0]));
    CUDA_CHECK(cudaEventSynchronize(e[3]));

    CUDA_CHECK(cudaEventElapsedTime(&h2dTime, e[0], e[1]));
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, e[1], e[2]));
    CUDA_CHECK(cudaEventElapsedTime(&d2hTime, e[2], e[3]));
    for (int i = 0; i < 4; i++) CUDA_CHECK(cudaEventDestroy(e[i]));
  }

  printf("*********** CDAC - Tech Workshop : hyPACK-2013 \n");
  printf("\n Device : %s (%d async copy engine(s))\n", prop.name, prop.asyncEngineCount);
  printf("\n Size of array : %d \n", n);
  printf(" Threads per block : %d (%d blocks)\n", threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock);
  printf(" Streams : %d \n", numStreams);
  printf("\n Time taken: %3.3f ms \n", elapsedTime);
  if (numStreams == 1)
    printf("   H2D copy %.3f ms | kernel %.3f ms | D2H copy %.3f ms\n", h2dTime, kernelTime, d2hTime);
  printf(" Result check: %s (%d mismatches)\n", errors ? "FAILED" : "PASSED", errors);

  for (int s = 0; s < numStreams; s++)
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  free(streams);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_result);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_result);

  return errors ? EXIT_FAILURE : 0;
}
