#include <stdio.h>
#include <stdlib.h>

//From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

// Usage: host_memory [N] [x_value] [y_value]
// N should be a multiple of 256 (default 1<<20).
int main(int argc, char **argv)
{
  int N = (argc > 1) ? atoi(argv[1]) : 1<<20;
  float x_val = (argc > 2) ? atof(argv[2]) : 1.0f;
  float y_val = (argc > 3) ? atof(argv[3]) : 2.0f;
  float expected = 2.0f*x_val + y_val;  // saxpy with a = 2.0f
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = x_val;
    y[i] = y_val;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on N elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(y[i]-expected));
  }
  // Print only the first few values; printing all N is very slow
  for (int i = 0; i < 4 && i < N; i++)
    printf("y[%d]=%f\n",i,y[i]);
  printf("N=%d, x=%f, y=%f, expected y=%f\n", N, x_val, y_val, expected);
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
