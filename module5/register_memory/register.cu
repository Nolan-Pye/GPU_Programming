#include <stdio.h>
#include <stdlib.h>

// Tunable parameters - override at compile time, e.g.
//   nvcc -DKERNEL_LOOP=1048576 -DKERNEL_SIZE=256 -DWORK_ITERS=64
// KERNEL_LOOP : number of elements (one thread per element)
// KERNEL_SIZE : threads per block (<= 1024)
// WORK_ITERS  : multiply/add steps per element in the register-vs-global
//               comparison kernels
// PRINT_ALL   : define to print every element like the original code
#ifndef KERNEL_LOOP
#define KERNEL_LOOP 2048
#endif

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 128
#endif

#ifndef WORK_ITERS
#define WORK_ITERS 32
#endif

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

__host__ void wait_exit(void)
{
        char ch;

        printf("\nPress any key to exit");
        ch = getchar();
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < KERNEL_LOOP; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
        }
}

__global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                unsigned int d_tmp = data[tid];
                d_tmp = d_tmp * 2;
                data[tid] = d_tmp;
        }
}

// Register version of a longer computation: load once into a register, do all
// WORK_ITERS steps on the register, then store once.
__global__ void test_gpu_register_loop(unsigned int * const data, const unsigned int num_elements)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                unsigned int d_tmp = data[tid];
                for(unsigned int i = 0; i < WORK_ITERS; i++)
                {
                        d_tmp = d_tmp * 3 + i;
                }
                data[tid] = d_tmp;
        }
}

// Same computation done directly on global memory. volatile forces a real
// load and store every iteration instead of letting the compiler keep the
// value in a register.
__global__ void test_gpu_global_loop(volatile unsigned int * const data, const unsigned int num_elements)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                for(unsigned int i = 0; i < WORK_ITERS; i++)
                {
                        data[tid] = data[tid] * 3 + i;
                }
        }
}

__host__ unsigned int host_loop(unsigned int d)
{
        for(unsigned int i = 0; i < WORK_ITERS; i++)
        {
                d = d * 3 + i;
        }
        return d;
}

// Runs one kernel on a fresh copy of the input; returns elapsed ms.
__host__ float time_kernel(int which, unsigned int * data_gpu, const unsigned int * host_in,
                unsigned int * host_out, unsigned int num_blocks, unsigned int num_threads,
                unsigned int num_elements, unsigned int num_bytes)
{
        cudaEvent_t start, stop;
        float ms = 0.0f;
        CUDA_CHECK(cudaMemcpy(data_gpu, host_in, num_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        if (which == 0)
                test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);
        else if (which == 1)
                test_gpu_register_loop <<<num_blocks, num_threads>>>(data_gpu, num_elements);
        else
                test_gpu_global_loop <<<num_blocks, num_threads>>>(data_gpu, num_elements);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());        // Wait for the GPU launched work to complete
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(host_out, data_gpu, num_bytes, cudaMemcpyDeviceToHost));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
}

__host__ void gpu_kernel(void)
{
        const unsigned int num_elements = KERNEL_LOOP;
        const unsigned int num_threads = KERNEL_SIZE;
        const unsigned int num_blocks = (num_elements + num_threads - 1)/num_threads;
        const unsigned int num_bytes = num_elements * sizeof(unsigned int);

        unsigned int * data_gpu;

        // Heap instead of stack: large KERNEL_LOOP values overflow the stack
        unsigned int * host_packed_array = (unsigned int *) malloc(num_bytes);
        unsigned int * host_packed_array_output = (unsigned int *) malloc(num_bytes);

        printf("KERNEL_LOOP (elements) = %u, KERNEL_SIZE (threads/block) = %u, blocks = %u, WORK_ITERS = %u\n",
                num_elements, num_threads, num_blocks, (unsigned int)WORK_ITERS);

        CUDA_CHECK(cudaMalloc(&data_gpu, num_bytes));

        generate_rand_data(host_packed_array);

        // 1) Original kernel: load to register, double, store
        float ms = time_kernel(0, data_gpu, host_packed_array, host_packed_array_output,
                        num_blocks, num_threads, num_elements, num_bytes);

        unsigned int mismatches = 0;
        for (unsigned int i = 0; i < num_elements; i++){
#ifdef PRINT_ALL
                printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);
#else
                if (i < 4)
                        printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);
#endif
                if (host_packed_array_output[i] != host_packed_array[i] * 2) mismatches++;
        }
        printf("[x2 register]      %.4f ms  mismatches: %u -> %s\n", ms, mismatches, mismatches ? "FAIL" : "PASS");

        // 2) and 3) WORK_ITERS steps kept in a register vs. done in global memory
        const char * names[2] = { "[loop in register]", "[loop in global]  " };
        for (int k = 1; k <= 2; k++)
        {
                ms = time_kernel(k, data_gpu, host_packed_array, host_packed_array_output,
                                num_blocks, num_threads, num_elements, num_bytes);
                mismatches = 0;
                for (unsigned int i = 0; i < num_elements; i++)
                {
                        if (host_packed_array_output[i] != host_loop(host_packed_array[i])) mismatches++;
                }
                printf("%s %.4f ms  mismatches: %u -> %s\n", names[k-1], ms, mismatches, mismatches ? "FAIL" : "PASS");
        }

        cudaFree((void* ) data_gpu);
        free(host_packed_array);
        free(host_packed_array_output);
        cudaDeviceReset();
//        wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_kernel();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}
