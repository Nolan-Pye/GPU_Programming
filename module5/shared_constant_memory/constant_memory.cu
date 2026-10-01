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
#include <assert.h>

// Override at compile time, e.g. nvcc -DKERNEL_LOOP=1024 -DWORK_SIZE=4096
#ifndef KERNEL_LOOP
#define KERNEL_LOOP 65536
#endif

#ifndef WORK_SIZE
#define WORK_SIZE 256
#endif

// Number of elements used by the timing comparison in gpu_kernel()
#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS (128 * 1024)
#endif

typedef unsigned short int u16;
typedef unsigned int u32;

__constant__  static const unsigned int const_data_01 = 0x55555555;
__constant__  static const unsigned int const_data_02 = 0x77777777;
__constant__  static const unsigned int const_data_03 = 0x33333333;
__constant__  static const unsigned int const_data_04 = 0x11111111;

__global__ void const_test_gpu_literal(u32 * data,
		const u32 num_elements) {
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements) {
		u32 d = 0x55555555;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= 0x55555555;
			d |= 0x77777777;
			d &= 0x33333333;
			d |= 0x11111111;
		}

		data[tid] = d;
	}
}

__global__ void const_test_gpu_const(unsigned int * const data, const unsigned int num_elements) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements) {
		unsigned int d = const_data_01;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= const_data_01;
			d |= const_data_02;
			d &= const_data_03;
			d |= const_data_04;
		}

		data[tid] = d;
	}
}

__device__  static unsigned int data_01 = 0x55555555;
__device__  static unsigned int data_02 = 0x77777777;
__device__  static unsigned int data_03 = 0x33333333;
__device__  static unsigned int data_04 = 0x11111111;

__global__ void const_test_gpu_gmem(unsigned int * const data, const unsigned int num_elements) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < num_elements) {
		unsigned int d = data_01;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= data_01;
			d |= data_02;
			d &= data_03;
			d |= data_04;
		}

		data[tid] = d;
	}
}

__host__ void gpu_kernel(void) {
	const unsigned int num_elements = NUM_ELEMENTS;
	const unsigned int num_threads = 256;
	const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;
	const unsigned int num_bytes = num_elements * sizeof(unsigned int);
	int max_device_num;
	const int max_runs = 6;

	cudaGetDeviceCount(&max_device_num);

	for (int device_num = 0; device_num < max_device_num; device_num++) {
		cudaSetDevice(device_num);

		for (int num_test = 0; num_test < max_runs; num_test++) {
			unsigned int * data_gpu;
			cudaEvent_t kernel_start1, kernel_stop1;
			cudaEvent_t kernel_start2, kernel_stop2;
			cudaEvent_t kernel_start3, kernel_stop3;
			float delta_time1 = 0.0f, delta_time2 = 0.0F, delta_time3 = 0.0f;
			struct cudaDeviceProp device_prop;
			char device_prefix[261];

			cudaMalloc(&data_gpu, num_bytes);
			cudaEventCreate(&kernel_start1);
			cudaEventCreate(&kernel_start2);
			cudaEventCreate(&kernel_start3);
			cudaEventCreateWithFlags(&kernel_stop3, cudaEventBlockingSync);
			
					cudaEventCreateWithFlags(&kernel_stop1,
							cudaEventBlockingSync);
			
					cudaEventCreateWithFlags(&kernel_stop2,
							cudaEventBlockingSync);

			cudaGetDeviceProperties(&device_prop, device_num);
			sprintf(device_prefix, "ID: %d %s:", device_num, device_prop.name);

			const_test_gpu_literal<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal startup  kernel!");

			cudaEventRecord(kernel_start1, 0);
			const_test_gpu_literal<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal runtime  kernel!");

			cudaEventRecord(kernel_stop1, 0);
			cudaEventSynchronize(kernel_stop1);
			
					cudaEventElapsedTime(&delta_time1, kernel_start1,
							kernel_stop1);

			// Warm-up launch, then time a second launch (same as the literal test)
			const_test_gpu_const<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal startup  kernel!");

			// kernel_start2 was never recorded originally, so delta_time2 was invalid
			cudaEventRecord(kernel_start2, 0);
			const_test_gpu_const<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);
			cudaEventRecord(kernel_stop2, 0);
			cudaEventSynchronize(kernel_stop2);
			
					cudaEventElapsedTime(&delta_time2, kernel_start2,
							kernel_stop2);

			// Same test reading the values from __device__ (global) memory
			const_test_gpu_gmem<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);
			cudaEventRecord(kernel_start3, 0);
			const_test_gpu_gmem<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);
			cudaEventRecord(kernel_stop3, 0);
			cudaEventSynchronize(kernel_stop3);
			cudaEventElapsedTime(&delta_time3, kernel_start3, kernel_stop3);

			// delta_time1 = literal, delta_time2 = constant (the labels were swapped)
			if (delta_time1 > delta_time2) {
				printf(
						"\n%sConstant version is faster by: %.3fms (Const=%.3fms vs. Literal=%.3fms, GMEM=%.3fms)",
						device_prefix, delta_time1 - delta_time2, delta_time2,
						delta_time1, delta_time3);
			} else {
				printf(
						"\n%sLiteral version is faster by: %.3fms (Const=%.3fms vs. Literal=%.3fms, GMEM=%.3fms)",
						device_prefix, delta_time2 - delta_time1, delta_time2,
						delta_time1, delta_time3);
			}

			cudaEventDestroy(kernel_start1);
			cudaEventDestroy(kernel_start2);
			cudaEventDestroy(kernel_stop1);
			cudaEventDestroy(kernel_stop2);
			cudaEventDestroy(kernel_start3);
			cudaEventDestroy(kernel_stop3);
			cudaFree(data_gpu);
		}

		cudaDeviceReset();
		printf("\n");
	}
//	wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	u32 *data = NULL;
	const u32 num_threads = 256;
	// Round up so a WORK_SIZE below/not a multiple of 256 still launches
	const u32 num_blocks = (WORK_SIZE + num_threads - 1)/num_threads;

	unsigned int *idata = (unsigned int *) malloc(sizeof(unsigned int) * WORK_SIZE);
	unsigned int *odata = (unsigned int *) malloc(sizeof(unsigned int) * WORK_SIZE);
	int i;
	for (i = 0; i < WORK_SIZE; i++){
		idata[i] = (unsigned int) i;
	}

	cudaMalloc((void** ) &data, sizeof(int) * WORK_SIZE);
	
	cudaMemcpy(data, idata, sizeof(unsigned int) * WORK_SIZE, cudaMemcpyHostToDevice);

	// Host reference: run the same loop the kernels run.
	// ((0x55555555 ^ 0x55555555) | 0x77777777) & 0x33333333 | 0x11111111 = 0x33333333
	u32 expected = 0x55555555;
	for (i = 0; i < KERNEL_LOOP; i++) {
		expected ^= 0x55555555;
		expected |= 0x77777777;
		expected &= 0x33333333;
		expected |= 0x11111111;
	}

	printf("WORK_SIZE = %d, KERNEL_LOOP = %d, blocks = %u, threads = %u\n",
			WORK_SIZE, KERNEL_LOOP, num_blocks, num_threads);

	// Each thread overwrites its input with the result, so the literal,
	// __constant__ and __device__ kernels should all produce the same values.
	const char *names[3] = { "literal", "constant", "gmem" };
	for (int k = 0; k < 3; k++) {
		cudaMemcpy(data, idata, sizeof(unsigned int) * WORK_SIZE, cudaMemcpyHostToDevice);
		if (k == 0) const_test_gpu_literal<<<num_blocks,num_threads>>>(data, WORK_SIZE);
		if (k == 1) const_test_gpu_const<<<num_blocks,num_threads>>>(data, WORK_SIZE);
		if (k == 2) const_test_gpu_gmem<<<num_blocks,num_threads>>>(data, WORK_SIZE);
		cudaDeviceSynchronize();	// Wait for the GPU launched work to complete
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Kernel error: %s\n", cudaGetErrorString(err));
		}

		cudaMemcpy(odata, data, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

		int mismatches = 0;
		for (i = 0; i < WORK_SIZE; i++) {
			if (odata[i] != expected) mismatches++;
		}
		// Print only the first few values; printing every element is slow
		for (i = 0; i < 4 && i < WORK_SIZE; i++) {
			printf("[%s] Input value: %u, device output: 0x%08X\n", names[k], idata[i], odata[i]);
		}
		printf("[%s] %d/%d outputs match host value 0x%08X\n\n",
				names[k], WORK_SIZE - mismatches, WORK_SIZE, expected);
	}

	cudaFree((void* ) data);
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
	gpu_kernel();	// literal vs. constant vs. global timing (was never called)

	return 0;
}
