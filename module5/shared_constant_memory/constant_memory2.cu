/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// Override at compile time, e.g. nvcc -DKERNEL_LOOP=1024 -DWORK_SIZE=4096
#ifndef WORK_SIZE
#define WORK_SIZE 256
#endif

typedef unsigned short int u16;
typedef unsigned int u32;

// Also the size of the __constant__ array: 4096 * 4 bytes = 16KB.
// Constant memory is limited to 64KB, so KERNEL_LOOP must be <= 16384.
#ifndef KERNEL_LOOP
#define KERNEL_LOOP 4096
#endif

// Number of elements used by the timing comparison in gpu_kernel()
#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS (128*1024)
#endif

__constant__ u32 const_data_gpu[KERNEL_LOOP];
__device__ static u32 gmem_data_gpu[KERNEL_LOOP];
static u32 const_data_host[KERNEL_LOOP];

__global__ void const_test_gpu_gmem(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d = gmem_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= gmem_data_gpu[0];
			d |= gmem_data_gpu[1];
			d &= gmem_data_gpu[2];
			d |= gmem_data_gpu[3];
		}

		data[tid] = d;
	}
}


__global__ void const_test_gpu_const(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d = const_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= const_data_gpu[0];
			d |= const_data_gpu[1];
			d &= const_data_gpu[2];
			d |= const_data_gpu[3];
		}

		data[tid] = d;
	}
}

__host__ void wait_exit(void)
{
	char ch;

	printf("\nPress any key to exit");
	ch = getchar();
}

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if(cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s",prefix,cudaGetErrorString(cudaGetLastError()),postfix);
		cudaDeviceReset();
		wait_exit();
		exit(1);
	}
}

__host__ void generate_rand_data(u32 * host_data_ptr)
{
	for(u32 i=0; i < KERNEL_LOOP; i++)
	{
		host_data_ptr[i] = (u32) rand();
	}
}

__host__ void gpu_kernel(void)
{
	const u32 num_elements = NUM_ELEMENTS;
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads-1))/num_threads;
	const u32 num_bytes = num_elements * sizeof(u32);
	int max_device_num;
	const int max_runs = 6;

	cudaGetDeviceCount(&max_device_num);

	for(int device_num=0; device_num < max_device_num; device_num++)
	{
		cudaSetDevice(device_num);

		u32 * data_gpu;
		cudaEvent_t kernel_start1, kernel_stop1;
		cudaEvent_t kernel_start2, kernel_stop2;
		float delta_time1 = 0.0F, delta_time2 = 0.0F;
		struct cudaDeviceProp device_prop;
		char device_prefix[261];

		cudaMalloc(&data_gpu, num_bytes);
		cudaEventCreate(&kernel_start1);
		cudaEventCreate(&kernel_start2);
		cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync);

		cudaGetDeviceProperties(&device_prop, device_num);
		sprintf(device_prefix, "ID: %d %s:", device_num, device_prop.name);

		for(int num_test=0; num_test < max_runs; num_test++)
		{
			generate_rand_data(const_data_host);

			// Load the same random data into both constant and global memory
			// before either kernel runs (originally gmem was filled one test late)
			cudaMemcpyToSymbol(const_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));
			cudaMemcpyToSymbol(gmem_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);
			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_start1,0);

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_stop1,0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1);

			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_start2,0);

			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_stop2,0);
			cudaEventSynchronize(kernel_stop2);
			cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2);

			if(delta_time1 > delta_time2)
			{
				printf("\n%sConstant version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time1-delta_time2, delta_time1, delta_time2);
			}
			else
			{
				printf("\n%sGMEM version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time2-delta_time1, delta_time1, delta_time2);
			}

		}

		cudaEventDestroy(kernel_start1);
		cudaEventDestroy(kernel_start2);
		cudaEventDestroy(kernel_stop1);
		cudaEventDestroy(kernel_stop2);
		cudaFree(data_gpu);

		cudaDeviceReset();
		printf("\n");
	}
//	wait_exit();	// blocks on getchar(); disabled so the script can run unattended
}

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	u32 *d = NULL;
	int i;
	const u32 num_threads = 256;
	// <<<1, WORK_SIZE>>> fails above 1024 threads, so use multiple blocks
	const u32 num_blocks = (WORK_SIZE + num_threads - 1) / num_threads;
	unsigned int *idata = (unsigned int *) malloc(sizeof(unsigned int) * WORK_SIZE);
	unsigned int *odata = (unsigned int *) malloc(sizeof(unsigned int) * WORK_SIZE);

	for (i = 0; i < WORK_SIZE; i++){
		idata[i] = (unsigned int) i;
	}

	// The __constant__ array is zero until it is loaded from the host.
	// Originally it was never loaded here, so every output was 0.
	srand(1);
	generate_rand_data(const_data_host);
	cudaMemcpyToSymbol(const_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));
	cudaMemcpyToSymbol(gmem_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));

	// Host reference: the same loop the kernels run, using the same data
	u32 expected = const_data_host[0];
	for (i = 0; i < KERNEL_LOOP; i++) {
		expected ^= const_data_host[0];
		expected |= const_data_host[1];
		expected &= const_data_host[2];
		expected |= const_data_host[3];
	}

	printf("WORK_SIZE = %d, KERNEL_LOOP = %d (constant array = %u bytes), blocks = %u, threads = %u\n",
			WORK_SIZE, KERNEL_LOOP, (unsigned int) sizeof(const_data_gpu), num_blocks, num_threads);
	printf("const_data[0..3] = 0x%08X 0x%08X 0x%08X 0x%08X\n",
			const_data_host[0], const_data_host[1], const_data_host[2], const_data_host[3]);

	cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE);

	const char *names[2] = { "constant", "gmem" };
	for (int k = 0; k < 2; k++) {
		cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
		if (k == 0) const_test_gpu_const<<<num_blocks, num_threads>>>(d, WORK_SIZE);
		if (k == 1) const_test_gpu_gmem<<<num_blocks, num_threads>>>(d, WORK_SIZE);

		cudaDeviceSynchronize();	// Wait for the GPU launched work to complete
		cuda_error_check("Error ", " returned from kernel!");
		cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

		int mismatches = 0;
		for (i = 0; i < WORK_SIZE; i++){
			if (odata[i] != expected) mismatches++;
		}
		// Print only the first few values; printing every element is slow
		for (i = 0; i < 4 && i < WORK_SIZE; i++){
			printf("[%s] Input value: %u, device output: 0x%08X, host output: 0x%08X\n",
					names[k], idata[i], odata[i], expected);
		}
		printf("[%s] %d/%d outputs match host\n\n", names[k], WORK_SIZE - mismatches, WORK_SIZE);
	}

	cudaFree((void*) d);
	free(idata);
	free(odata);
	cudaDeviceReset();


}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();
	gpu_kernel();	// constant vs. global memory timing (was never called)

	return 0;
}
