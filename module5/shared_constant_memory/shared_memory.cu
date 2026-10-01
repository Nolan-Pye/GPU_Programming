#include <stdio.h>
#include <stdlib.h>

typedef unsigned short int u16;
typedef unsigned int u32;

// Tunable parameters - override at compile time, e.g.
//   nvcc -DNUM_ELEMENTS=4096 -DMAX_NUM_LISTS=32 -DMERGE_VARIANT=5 -DINPUT_MODE=2
// NUM_ELEMENTS : total values to sort; must be a multiple of MAX_NUM_LISTS.
//                The kernel uses 3 * NUM_ELEMENTS * 4 bytes of static shared
//                memory, so NUM_ELEMENTS <= 4096 (48 KB limit).
// MAX_NUM_LISTS: number of interleaved lists == threads per block (<= 1024).
//                Must be a power of two for merge variants 5 and 9.
// MERGE_VARIANT: 1 = single thread, 5 = parallel reduction,
//                6 = atomicMin, 9 = two-level atomicMin
// INPUT_MODE   : 0 = ascending (original), 1 = descending, 2 = random
#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS 2048
#endif

#ifndef MAX_NUM_LISTS
#define MAX_NUM_LISTS 16
#endif

#ifndef MERGE_VARIANT
#define MERGE_VARIANT 1
#endif

#ifndef INPUT_MODE
#define INPUT_MODE 0
#endif

#if (NUM_ELEMENTS % MAX_NUM_LISTS) != 0
#error "NUM_ELEMENTS must be a multiple of MAX_NUM_LISTS"
#endif

__host__ void cpu_sort(u32 * const data, const u32 num_elements)
{
	static u32 cpu_tmp_0[NUM_ELEMENTS];
	static u32 cpu_tmp_1[NUM_ELEMENTS];

	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for(u32 i=0; i<num_elements; i++)
		{
			const u32 d = data[i];
			if((d & bit_mask) > 0)
			{
				cpu_tmp_1[base_cnt_1] = d;
				base_cnt_1++;
			}
			else
			{
				cpu_tmp_0[base_cnt_0] = d;
				base_cnt_0++;
			}
		}

		// Copy data back to the source
		// First the zero list, then the one list
		for(u32 i=0; i<base_cnt_0; i++)
		{
			data[i] = cpu_tmp_0[i];
		}
		for(u32 i = 0; i<base_cnt_1; i++)
		{
			data[base_cnt_0+i] = cpu_tmp_1[i];
		}
	}
}

__device__ void radix_sort(u32 * const sort_tmp,
				const u32 num_lists,
				const u32 num_elements,
				const u32 tid,
				u32 * const sort_tmp_0,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<num_elements; i+=num_lists)
		{
			const u32 elem = sort_tmp[i+tid];
			const u32 bit_mask = (1 << bit);
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1+tid] = elem;
				base_cnt_1+=num_lists;
			}
			else
			{
				sort_tmp_0[base_cnt_0+tid] = elem;
				base_cnt_0+=num_lists;
			}
		}
		
		// Copy data back to source - first the zero list
		for(u32 i=0;i<base_cnt_0;i+=num_lists)
		{
			sort_tmp[i+tid] = sort_tmp_0[i+tid];
		}
		
		//Copy data back to source - then the one list
		for(u32 i=0;i<base_cnt_1; i+=num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}
	}
	__syncthreads();
}

__device__ void radix_sort2(u32 * const sort_tmp,
				const u32 num_lists,
				const u32 num_elements,
				const u32 tid,
				u32 * const sort_tmp_0,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<num_elements; i+=num_lists)
		{
			const u32 elem = sort_tmp[i+tid];
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1+tid] = elem;
				base_cnt_1+=num_lists;
			}
			else
			{
				// Fixed: write the zero list in place (base_cnt_0 <= i, so this
				// never overwrites an unread element). The original wrote to
				// sort_tmp_0, which radix_sort2 never copies back.
				sort_tmp[base_cnt_0+tid] = elem;
				base_cnt_0+=num_lists;
			}
		}
		
		//Copy data back to source - then the one list
		for(u32 i=0;i<base_cnt_1; i+=num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}
	}
	__syncthreads();
}

u32 find_min(const u32 * const src_array,
		u32 * const list_indexes,
		const u32 num_lists,
		const u32 num_elements_per_list)
{
	u32 min_val = 0xFFFFFFF;
	u32 min_idx = 0;
	// Iterate over each of the lists
	for(u32 i=0; i<num_lists; i++)
	{
		// If the current list ahs already been emptied
		// then ignore it
		if(list_indexes[i] < num_elements_per_list)
		{
			const u32 src_idx = i + (list_indexes[i] * num_lists);

			const u32 data = src_array[src_idx];
	
			if(data <= min_val)
			{
				min_val = data;
				min_idx = i;
			}
		}
	}
	list_indexes[min_idx]++;
	return min_val;
}

void merge_array(const u32 * const src_array,
			u32 * const dest_array,
			const u32 num_lists,
			const u32 num_elements)
{
	const u32 num_elements_per_list = (num_elements / num_lists);

	unsigned int list_indexes[MAX_NUM_LISTS];

	for(u32 list=0; list < MAX_NUM_LISTS; list++)
	{
		list_indexes[list] = 0;
	}

	for(u32 i=0; i<num_elements; i++)
	{
		dest_array[i] = find_min(src_array,
					list_indexes,
					num_lists,
					num_elements_per_list);
	}
}

__device__ void copy_data_to_shared(const u32 * const data,
									u32 * const sort_tmp,
									const u32 num_lists,
									const u32 num_elements,
									const u32 tid)
{
	// Copy data into temp store
	// Fixed: stride by num_lists so each thread copies only its own list
	for(u32 i = 0; i<num_elements; i+=num_lists)
	{
		sort_tmp[i+tid] = data[i+tid];
	}
	__syncthreads();
}


// Uses a single thread for merge
__device__ void merge_array1(const u32 * const src_array,
							u32 * const dest_array,
							const u32 num_lists,
							const u32 num_elements,
							const u32 tid)
{
	__shared__ u32 list_indexes[MAX_NUM_LISTS];

	// Multiple threads
	list_indexes[tid] = 0;
	__syncthreads();

	// Single threaded
	if(tid == 0)
	{
		const u32 num_elements_per_list = (num_elements / num_lists);

		for (u32 i = 0; i < num_elements; i++)
		{
			u32 min_val = 0xFFFFFFFF;
			u32 min_idx = 0;

			// Iterate over each of the lists
			for(u32 list=0; list<num_lists;list++)
			{
				//If the current list has already been emptied then ignored it
				if(list_indexes[list] < num_elements_per_list)
				{
					const u32 src_idx = list + (list_indexes[list] * num_lists);

					const u32 data = src_array[src_idx];

					if(data <= min_val)
					{
						min_val = data;
						min_idx = list;
					}
				}
			}
			list_indexes[min_idx]++;
			dest_array[i] = min_val;
		}
	}
}

// Uses multiple threads for merge
// Deals with multiple identical entries in the data
__device__ void merge_array6(const u32 * const src_array,
								u32 * const dest_array,
								const u32 num_lists,
								const u32 num_elements,
								const u32 tid)
{
	const u32 num_elements_per_list = (num_elements / num_lists);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	list_indexes[tid] = 0;

	//Wait for list_indexes[tid] to be cleared
	__syncthreads();

	//Iterate over all elements
	for(u32 i=0; i<num_elements; i++)
	{
		//Create a value shared with other threads
		__shared__ u32 min_val;
		__shared__ u32 min_tid;

		// Use a temp register for work purposes
		u32 data;

		//If the current list has not already been
		//emptied then read from it, else ignore it
		if(list_indexes[tid] < num_elements_per_list)
		{
			//Work out from the list_index, the index into
			// the linear array
			const u32 src_idx = tid + (list_indexes[tid] * num_lists);

			//Read the data from the list for the given
			// thread
			data = src_array[src_idx];
		}
		else
		{
			data = 0xFFFFFFFF;
		}

		//Have thread zero clear the min values
		if(tid == 0)
		{
			// Write a very large value so the first
			// thread wins with the min
			min_val = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}

		// Wait for all threads
		__syncthreads();

		// Have every thread try to store it's value into
		// min_val. Only the thread with the lowest value
		// will win.
		atomicMin(&min_val, data);

		//Make sure all threads have taken their turn
		__syncthreads();

		// If this thread was the one with the minimum
		if(min_val == data)
		{
			// Check for equal values
			// Lowest tid wins, and does the write
			atomicMin(&min_tid, tid);
		}

		// Make sure all threads have taken their turn.
		__syncthreads();

		// If this thread has the lowest tid
		if(tid == min_tid)
		{
			// Increment the list pointer for this thread
			list_indexes[tid]++;

			// Store the winning value
			dest_array[i] = data;
		}
	}
}

// Uses multiple threads for reduction type merge
__device__ void merge_array5(const u32 * const src_array,
								u32 * const dest_array,
								const u32 num_lists,
								const u32 num_elements,
								const u32 tid)
{
	const u32 num_elements_per_list = (num_elements / num_lists);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	__shared__ u32 reduction_val[MAX_NUM_LISTS];
	__shared__ u32 reduction_idx[MAX_NUM_LISTS];

	//Clear the working sets
	list_indexes[tid] = 0;
	reduction_val[tid] = 0;
	reduction_idx[tid] = 0;
	__syncthreads();

	for(u32 i=0; i<num_elements; i++)
	{
		// We need (num_lists / 2) active threads
		u32 tid_max = num_lists >> 1;

		u32 data;

		// If the current list has already been
		// emptied then ignore it
		if(list_indexes[tid] < num_elements_per_list)
		{
			const u32 src_idx = tid + (list_indexes[tid] * num_lists);

			data = src_array[src_idx];
		}
		else
		{
			data = 0xFFFFFFFF;
		}

		reduction_val[tid] = data;
		reduction_idx[tid] = tid;

		__syncthreads();

		while(tid_max != 0)
		{
			if(tid < tid_max)
			{
				const u32 val2_idx = tid + tid_max;

				const u32 val2 = reduction_val[val2_idx];

				if(reduction_val[tid] > val2)
				{
					reduction_val[tid] = val2;
					reduction_idx[tid] = reduction_idx[val2_idx];
				}
			}
			tid_max >>= 1;

			__syncthreads();
		}
		if(tid == 0)
		{
			list_indexes[reduction_idx[0]]++;

			dest_array[i] = reduction_val[0];
		}

		__syncthreads();
	}
}

#define REDUCTION_SIZE 8
#define REDUCTION_SIZE_BIT_SHIFT 3
// At least 1 so fewer than REDUCTION_SIZE lists still compiles
#define MAX_ACTIVE_REDUCTIONS (((MAX_NUM_LISTS) / REDUCTION_SIZE) > 0 ? ((MAX_NUM_LISTS) / REDUCTION_SIZE) : 1)

__device__ void merge_array9(const u32 * const src_array,
								u32 * const dest_array,
								const u32 num_lists,
								const u32 num_elements,
								const u32 tid)
{
	u32 data = src_array[tid];

	const u32 s_idx = tid >> REDUCTION_SIZE_BIT_SHIFT;

	const u32 num_reductions = num_lists >> REDUCTION_SIZE_BIT_SHIFT;
	const u32 num_elements_per_list = (num_elements / num_lists);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	list_indexes[tid] = 0;

	for(u32 i=0; i<num_elements; i++)
	{
		__shared__ u32 min_val[MAX_ACTIVE_REDUCTIONS];
		__shared__ u32 min_tid;

		if(tid < num_lists)
		{
			min_val[s_idx] = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}

		__syncthreads();

		atomicMin(&min_val[s_idx], data);

		if(num_reductions > 0)
		{
			__syncthreads();

			if(tid < num_reductions)
			{
				atomicMin(&min_val[0], min_val[tid]);
			}

			__syncthreads();
		}

		if(min_val[0] == data)
		{
			atomicMin(&min_tid, tid);
		}

		__syncthreads();

		if(tid == min_tid)
		{
			list_indexes[tid]++;

			dest_array[i] = data;

			if(list_indexes[tid] < num_elements_per_list)
			{
				data = src_array[tid + (list_indexes[tid] * num_lists)];
			}
			else
			{
				data = 0xFFFFFFFF;
			}
		}
		__syncthreads();
	}
}

__global__ void gpu_sort_array_array(u32 * const data,
					const u32 num_lists,
					const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ u32 sort_tmp[NUM_ELEMENTS];
	__shared__ u32 sort_tmp_0[NUM_ELEMENTS];
	__shared__ u32 sort_tmp_1[NUM_ELEMENTS];

	copy_data_to_shared(data, sort_tmp, num_lists,
				num_elements, tid);

	radix_sort2(sort_tmp, num_lists, num_elements, tid, sort_tmp_0, sort_tmp_1);

#if MERGE_VARIANT == 5
	merge_array5(sort_tmp, data, num_lists, num_elements, tid);
#elif MERGE_VARIANT == 6
	merge_array6(sort_tmp, data, num_lists, num_elements, tid);
#elif MERGE_VARIANT == 9
	merge_array9(sort_tmp, data, num_lists, num_elements, tid);
#else
	merge_array1(sort_tmp, data, num_lists, num_elements, tid);
#endif
}

void execute_host_functions()
{

}

static const char *merge_name(void)
{
	switch(MERGE_VARIANT)
	{
		case 5: return "merge_array5 (parallel reduction)";
		case 6: return "merge_array6 (atomicMin)";
		case 9: return "merge_array9 (two-level atomicMin)";
		default: return "merge_array1 (single thread)";
	}
}

void execute_gpu_functions()
{
	u32 *d = NULL;
	static u32 idata[NUM_ELEMENTS], odata[NUM_ELEMENTS], expected[NUM_ELEMENTS];
	u32 i;

	srand(617);
	for (i = 0; i < NUM_ELEMENTS; i++){
#if INPUT_MODE == 1
		idata[i] = (u32) (NUM_ELEMENTS - 1 - i);
#elif INPUT_MODE == 2
		idata[i] = (u32) (rand() % (NUM_ELEMENTS * 4));
#else
		idata[i] = (u32) i;
#endif
		expected[i] = idata[i];
	}
	cpu_sort(expected, NUM_ELEMENTS);

	printf("NUM_ELEMENTS=%u MAX_NUM_LISTS(threads)=%u INPUT_MODE=%d merge=%s\n",
		(u32)NUM_ELEMENTS, (u32)MAX_NUM_LISTS, INPUT_MODE, merge_name());
	printf("Static shared memory for sort buffers: %u bytes\n",
		(u32)(3 * NUM_ELEMENTS * sizeof(u32)));

	cudaMalloc((void** ) &d, sizeof(u32) * NUM_ELEMENTS);
	
	cudaMemcpy(d, idata, sizeof(u32) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Call GPU kernels
	// Fixed: one thread per list. The original launched NUM_ELEMENTS (2048)
	// threads, which exceeds the 1024 threads/block limit, so the launch
	// failed silently and the (already sorted) input was copied back unchanged.
	cudaEventRecord(start);
	gpu_sort_array_array<<<1, MAX_NUM_LISTS>>>(d,MAX_NUM_LISTS,NUM_ELEMENTS);
	cudaEventRecord(stop);

	cudaError_t err = cudaGetLastError();
	if (err == cudaSuccess) err = cudaDeviceSynchronize();	// Wait for the GPU launched work to complete
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		cudaFree(d);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	
	cudaMemcpy(odata, d, sizeof(u32) * NUM_ELEMENTS, cudaMemcpyDeviceToHost);

	u32 mismatches = 0;
	for (i = 0; i < NUM_ELEMENTS; i++) {
#ifdef VERBOSE
		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);
#else
		if (i < 8 || i >= NUM_ELEMENTS - 4)
			printf("  [%4u] input %6u -> device %6u (cpu %6u)\n", i, idata[i], odata[i], expected[i]);
		else if (i == 8)
			printf("  ...\n");
#endif
		if (odata[i] != expected[i]) mismatches++;
	}
	printf("Kernel time: %.3f ms   Mismatches vs CPU radix sort: %u -> %s\n",
		ms, mismatches, mismatches == 0 ? "PASS" : "FAIL");
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree((void* ) d);
	cudaDeviceReset();

}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return 0;
}
