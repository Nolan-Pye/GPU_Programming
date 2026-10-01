// Module 3 practical: runs the 2D kernel with several different thread counts,
// block sizes, numbers of blocks, and grid dimensions (including the original
// 1x4 and 2x2 grid layouts).
//   ./grids                                        -> runs the 6 built-in configurations
//   ./grids <grid_x> <grid_y> <block_x> <block_y>  -> runs a single configuration

#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id_2d_A(
				unsigned int * const block_x,
				unsigned int * const block_y,
				unsigned int * const thread_x,
				unsigned int * const thread_y,
				unsigned int * const calc_thread,
				unsigned int * const x_thread,
				unsigned int * const y_thread,
				unsigned int * const grid_dimx,
				unsigned int * const block_dimx,
				unsigned int * const grid_dimy,
				unsigned int * const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread_x[thread_idx] = threadIdx.x;
	thread_y[thread_idx] = threadIdx.y;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

#define NUM_ARRAYS 11

void run_config(const char *name, const dim3 blocks, const dim3 threads)
{
	/* The array covers exactly the threads launched: one element per thread */
	const unsigned int array_size_x = blocks.x * threads.x;
	const unsigned int array_size_y = blocks.y * threads.y;
	const unsigned int array_size = array_size_x * array_size_y;
	const size_t array_size_in_bytes = array_size * sizeof(unsigned int);

	unsigned int *cpu[NUM_ARRAYS];
	unsigned int *gpu[NUM_ARRAYS];
	for (int a = 0; a < NUM_ARRAYS; a++)
	{
		cpu[a] = (unsigned int *)malloc(array_size_in_bytes);
		cudaMalloc((void **)&gpu[a], array_size_in_bytes);
	}

	/* Execute our kernel */
	what_is_my_id_2d_A<<<blocks, threads>>>(gpu[0], gpu[1], gpu[2], gpu[3], gpu[4],
			gpu[5], gpu[6], gpu[7], gpu[8], gpu[9], gpu[10]);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	/* Copy back the gpu results to the CPU, then free the GPU arrays */
	for (int a = 0; a < NUM_ARRAYS; a++)
	{
		cudaMemcpy(cpu[a], gpu[a], array_size_in_bytes, cudaMemcpyDeviceToHost);
		cudaFree(gpu[a]);
	}

	const unsigned int *cpu_block_x = cpu[0];
	const unsigned int *cpu_block_y = cpu[1];
	const unsigned int *cpu_thread_x = cpu[2];
	const unsigned int *cpu_thread_y = cpu[3];
	const unsigned int *cpu_calc_thread = cpu[4];
	const unsigned int *cpu_xthread = cpu[5];
	const unsigned int *cpu_ythread = cpu[6];
	const unsigned int *cpu_grid_dimx = cpu[7];
	const unsigned int *cpu_block_dimx = cpu[8];
	const unsigned int *cpu_grid_dimy = cpu[9];
	const unsigned int *cpu_block_dimy = cpu[10];

	printf("\n=== %s: grid %ux%u blocks, block %ux%u threads = %u threads (array %ux%u) ===\n",
			name, blocks.x, blocks.y, threads.x, threads.y, array_size,
			array_size_x, array_size_y);

	/* Iterate through the arrays and print */
	for (unsigned int i = 0; i < array_size; i++)
	{
		printf("CT: %4u BKX: %2u BKY: %2u TIDX: %2u TIDY: %2u YTID: %3u XTID: %3u GDX: %2u BDX: %2u GDY: %2u BDY: %2u\n",
				cpu_calc_thread[i], cpu_block_x[i], cpu_block_y[i], cpu_thread_x[i], cpu_thread_y[i],
				cpu_ythread[i], cpu_xthread[i], cpu_grid_dimx[i], cpu_block_dimx[i],
				cpu_grid_dimy[i], cpu_block_dimy[i]);
	}

	for (int a = 0; a < NUM_ARRAYS; a++)
	{
		free(cpu[a]);
	}
}

int main(int argc, char **argv)
{
	if (argc == 5)
	{
		const dim3 blocks(atoi(argv[1]), atoi(argv[2]));
		const dim3 threads(atoi(argv[3]), atoi(argv[4]));
		run_config("Custom", blocks, threads);
		return EXIT_SUCCESS;
	}

	/* Total thread count = (32 * 4) threads/block * (1 * 4) blocks = 512 */
	run_config("Kernel 0 (1x4 rect)", dim3(1, 4), dim3(32, 4));

	/* Total thread count = (16 * 8) threads/block * (2 * 2) blocks = 512 */
	run_config("Kernel 1 (2x2 square)", dim3(2, 2), dim3(16, 8));

	/* Total thread count = 8 * 4 * (2 * 1) = 64 */
	run_config("Kernel 2 (2x1 wide)", dim3(2, 1), dim3(8, 4));

	/* Total thread count = 8 * 4 * (3 * 3) = 288 */
	run_config("Kernel 3 (3x3 square)", dim3(3, 3), dim3(8, 4));

	/* Total thread count = 8 * 4 * (4 * 2) = 256 */
	run_config("Kernel 4 (4x2 wide)", dim3(4, 2), dim3(8, 4));

	/* Total thread count = 8 * 8 * (4 * 4) = 1024 */
	run_config("Kernel 5 (4x4 square)", dim3(4, 4), dim3(8, 8));

	return EXIT_SUCCESS;
}
