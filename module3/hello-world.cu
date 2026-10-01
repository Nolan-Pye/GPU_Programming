// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world
//
// Module 3 practical: runs the kernel with several different thread counts,
// block sizes, and numbers of blocks.
//   ./hello-world                        -> runs the 5 built-in configurations
//   ./hello-world <num_blocks> <block_size> -> runs a single configuration

#include <stdio.h>
#include <stdlib.h>

__global__
void hello(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub(unsigned int num_blocks, unsigned int block_size)
{
	const unsigned int array_size = num_blocks * block_size;
	const size_t array_size_in_bytes = sizeof(unsigned int) * array_size;

	unsigned int *cpu_block = (unsigned int *)malloc(array_size_in_bytes);
	unsigned int *cpu_thread = (unsigned int *)malloc(array_size_in_bytes);

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	cudaMalloc((void **)&gpu_block, array_size_in_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_in_bytes);

	/* Execute our kernel */
	hello<<<num_blocks, block_size>>>(gpu_block, gpu_thread);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy( cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_thread, gpu_thread, array_size_in_bytes, cudaMemcpyDeviceToHost );

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	printf("=== %u threads = %u block(s) x %u threads/block ===\n",
			array_size, num_blocks, block_size);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < array_size; i++)
	{
		printf("Global thread: %4u - Block: %3u - Thread in block: %4u\n",
				i, cpu_block[i], cpu_thread[i]);
	}
	printf("\n");

	free(cpu_block);
	free(cpu_thread);
}

int main(int argc, char **argv)
{
	if (argc == 3)
	{
		main_sub((unsigned int)atoi(argv[1]), (unsigned int)atoi(argv[2]));
		return EXIT_SUCCESS;
	}

	/* {num_blocks, block_size}: 5 different total thread counts */
	const unsigned int configs[][2] = {
		{ 1, 16},	/*  16 threads: the original program */
		{ 4,  8},	/*  32 threads: many small blocks */
		{ 2, 32},	/*  64 threads: one warp per block */
		{ 3, 64},	/* 192 threads: odd number of blocks */
		{ 8, 32},	/* 256 threads: eight one-warp blocks */
	};

	for (unsigned int i = 0; i < sizeof(configs) / sizeof(configs[0]); i++)
	{
		main_sub(configs[i][0], configs[i][1]);
	}

	return EXIT_SUCCESS;
}
