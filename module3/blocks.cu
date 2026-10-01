// Module 3 practical: runs the kernel with several different thread counts,
// block sizes, and numbers of blocks.
//   ./blocks                                   -> runs the 5 built-in configurations
//   ./blocks <total_threads> <threads_per_block> -> runs a single configuration

#include <stdio.h>
#include <stdlib.h>

__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0(unsigned int array_size, unsigned int num_threads)
{
	const size_t array_size_in_bytes = sizeof(unsigned int) * array_size;

	/* Declare two host arrays of array_size each */
	unsigned int *cpu_block = (unsigned int *)malloc(array_size_in_bytes);
	unsigned int *cpu_thread = (unsigned int *)malloc(array_size_in_bytes);

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	cudaMalloc((void **)&gpu_block, array_size_in_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_in_bytes);

	const unsigned int num_blocks = array_size/num_threads;

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread);

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
			array_size, num_blocks, num_threads);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < array_size; i++)
	{
		printf("Thread: %4u - Block: %3u\n",cpu_thread[i],cpu_block[i]);
	}
	printf("\n");

	free(cpu_block);
	free(cpu_thread);
}

int main(int argc, char **argv)
{
	if (argc == 3)
	{
		const unsigned int total = (unsigned int)atoi(argv[1]);
		const unsigned int per_block = (unsigned int)atoi(argv[2]);
		if (per_block == 0 || total % per_block != 0)
		{
			printf("total_threads must be a non-zero multiple of threads_per_block\n");
			return EXIT_FAILURE;
		}
		main_sub0(total, per_block);
		return EXIT_SUCCESS;
	}

	/* {total_threads, threads_per_block}: 5 different total thread counts */
	const unsigned int configs[][2] = {
		{ 64,  16},	/*   4 blocks of 16 */
		{128,  32},	/*   4 blocks of one warp each */
		{256,  16},	/*  16 blocks of 16: the original program */
		{512, 128},	/*   4 blocks of 128 */
		{1024, 256},	/*   4 blocks of 256 */
	};

	for (unsigned int i = 0; i < sizeof(configs) / sizeof(configs[0]); i++)
	{
		main_sub0(configs[i][0], configs[i][1]);
	}

	return EXIT_SUCCESS;
}
