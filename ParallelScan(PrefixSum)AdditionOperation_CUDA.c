// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 64

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Combine localized result array together by adding the cumulative total
// of previous blocks to each output block.
__global__ void combine(float * input, float * output, int len, float * sums) {
	unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
		
	// If the global index is outside array range
	if (global_index >= len) {
		return;
	}
	
	float total = 0;
	for (int i = 0; i < blockIdx.x; i++) {
		
		total += sums[i];
		if (threadIdx.x == 0) {
			printf("\n[%d-%d] - Sum of block %i is %f giving total %f", blockIdx.x, threadIdx.x, i, sums[i], total);
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		printf("\n[%d-%d] - Final total = %f (output) + %f (cumulative)", blockIdx.x, threadIdx.x, output[global_index], total);
	}
	
	output[global_index] += total;
	
	if (threadIdx.x == 0) {
		printf("\n[%d-%d] Final = %f ", blockIdx.x, threadIdx.x, output[global_index]);
	}
	
	__syncthreads();
}


__global__ void scan(float * input, float * output, int len, float * sums) {
	__shared__ float result[2 * BLOCK_SIZE];
	
	// Determine global index into input/output array
	unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Load first element
	if (global_index < len) {
		result[threadIdx.x] = input[global_index];
	} else {
		result[threadIdx.x] = 0;
	}
	
	// Load second element
		if (global_index + blockDim.x < len) {
		result[threadIdx.x + blockDim.x] = input[global_index + blockDim.x];
	} else {
		result[threadIdx.x + blockDim.x] = 0;
	}
	
	// Wait for all threads to finish loading data.
	__syncthreads();
	
	// Phase 1 - Reduction phase
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < BLOCK_SIZE) {
			result[index] += result[index - stride];
		}
		
		// Wait for all threads to finish current iteration.
		__syncthreads();
	}
	
	// Phase 2 - Post reduction reversal
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		// Make sure previous iterations have finished and previous reduction step has completed.
		__syncthreads();
				
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		// If index is valid data range write result
		if (index + stride < BLOCK_SIZE) {
			result[index + stride] += result[index];
		}		
	}
	
	// Wait for all threads to post reduction reversal phase
	__syncthreads();
	
	// Write elements from shared memory to global memory.
	if (global_index < len) {
		output[global_index] = result[threadIdx.x];
	}
	
	// If final thread in block save sum
	if (threadIdx.x + 1 == blockDim.x) {
		printf("\n[%i-%i] Final element is %f [%u]", blockIdx.x, threadIdx.x, output[global_index], global_index);
		sums[blockIdx.x] = output[global_index];
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	float * deviceSums;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

	// Use Ceiling calcuation to get sufficent blocks to cover all elements.
	int numBlocks = (numElements - 1) / BLOCK_SIZE + 1;
	wbLog(TRACE, "The number of blocks is ", numBlocks);
	
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceSums, numBlocks*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Initialize the grid and block dimensions
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
		
    wbTime_start(Compute, "Performing CUDA computation");
	
	// Scan each section of the array into a localized result
	scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceSums);
  	cudaDeviceSynchronize();
	
	// Add the cumulative sum of previous sections into each localized result
	// to get the final values.
	combine<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceSums);
  	cudaDeviceSynchronize();
	
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
	cudaFree(deviceSums);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}